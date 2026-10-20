"""Download, format, tokenize, and persist ATHENA's sequential task datasets."""

from __future__ import annotations

import logging
import os
from functools import partial
from pathlib import Path
from typing import Any

import yaml
from datasets import Dataset, DatasetDict, load_dataset
from transformers import AutoTokenizer


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "config.yaml"
LOGGER = logging.getLogger("athena.data_pipeline")

TASKS = [
    {
        "name": "task_1_medical",
        "domain": "Medical",
        "sources": [
            ("medalpaca/medical_meadow_medqa", None),
            ("pubmed_qa", "pqa_labeled"),
        ],
        "formatter": "medical",
    },
    {
        "name": "task_2_legal_finance",
        "domain": "Financial/Legal",
        "sources": [
            ("zeroshot/twitter-financial-news-sentiment", None),
            ("financial_phrasebank", "sentences_allagree"),
        ],
        "formatter": "finance",
    },
    {
        "name": "task_3_superglue_boolq",
        "domain": "General Reasoning / SuperGLUE BoolQ",
        "sources": [
            ("aps/super_glue", "boolq"),
        ],
        "formatter": "boolq",
    },
]


def load_config() -> dict[str, Any]:
    with CONFIG_PATH.open(encoding="utf-8") as config_file:
        config = yaml.safe_load(config_file)
    if not isinstance(config, dict):
        raise ValueError(f"Expected a YAML mapping in {CONFIG_PATH}")
    return config


def load_first_available(sources: list[tuple[str, str | None]]) -> tuple[DatasetDict | Dataset, str]:
    errors = []
    for dataset_name, dataset_config in sources:
        try:
            LOGGER.info("Loading public dataset %s (config=%s)", dataset_name, dataset_config)
            if dataset_config is None:
                loaded = load_dataset(dataset_name, trust_remote_code=False)
            else:
                loaded = load_dataset(dataset_name, dataset_config, trust_remote_code=False)
            return loaded, dataset_name
        except Exception as error:
            errors.append(f"{dataset_name}: {error}")
            LOGGER.warning("Could not load %s: %s", dataset_name, error)
    details = "\n".join(errors)
    raise RuntimeError(f"No configured public dataset source could be loaded:\n{details}")


def choose_splits(
    loaded: DatasetDict | Dataset,
    train_limit: int,
    val_limit: int,
    seed: int,
) -> tuple[Dataset, Dataset]:
    if isinstance(loaded, Dataset):
        train_source = loaded
        val_source = None
    else:
        train_key = next((key for key in ("train", "training") if key in loaded), None)
        if train_key is None:
            train_key = next(iter(loaded.keys()), None)
        if train_key is None:
            raise ValueError("The loaded dataset has no splits")
        train_source = loaded[train_key]
        val_key = next((key for key in ("validation", "test", "dev") if key in loaded), None)
        val_source = loaded[val_key] if val_key else None

    if val_source is not None and len(val_source) > 0:
        train_source = train_source.shuffle(seed=seed)
        val_source = val_source.shuffle(seed=seed)
        train = train_source.select(range(min(train_limit, len(train_source))))
        val = val_source.select(range(min(val_limit, len(val_source))))
    else:
        shuffled = train_source.shuffle(seed=seed)
        available_val = min(val_limit, max(0, len(shuffled) - 1))
        available_train = min(train_limit, max(0, len(shuffled) - available_val))
        train = shuffled.select(range(available_train))
        val_start = len(shuffled) - available_val
        val = shuffled.select(range(val_start, val_start + available_val))

    if len(train) == 0 or len(val) == 0:
        raise ValueError(
            f"Not enough rows for non-empty splits (train={len(train)}, val={len(val)})"
        )
    return train, val


def _string(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, dict):
        return "\n".join(f"{key}: {item}" for key, item in value.items())
    if isinstance(value, (list, tuple)):
        return "\n".join(str(item) for item in value)
    return str(value)


def _first(row: dict[str, Any], keys: tuple[str, ...], default: Any = "") -> Any:
    for key in keys:
        value = row.get(key)
        if value is not None and value != "":
            return value
    return default


def format_medical(row: dict[str, Any]) -> dict[str, str]:
    instruction = _first(row, ("instruction", "question", "prompt"), "Answer the medical question.")
    context = _first(row, ("input", "context", "passage"))
    answer = _first(row, ("output", "answer", "long_answer", "response"))
    if not context and row.get("options"):
        context = row["options"]
    return {"text": _format_prompt(instruction, context, answer)}


def format_finance(row: dict[str, Any]) -> dict[str, str]:
    text = _first(row, ("text", "sentence", "statement", "content"))
    label = _first(row, ("label", "sentiment", "target"), "unknown")
    if isinstance(label, int):
        label_names = {0: "negative", 1: "neutral", 2: "positive"}
        label = label_names.get(label, str(label))
    return {
        "text": _format_prompt(
            "Classify the financial statement's sentiment.", text, _string(label)
        )
    }


def format_boolq(row: dict[str, Any]) -> dict[str, str]:
    label = row.get("label", -1)
    if isinstance(label, bool):
        answer = "Yes" if label else "No"
    elif label in (0, 1):
        answer = "Yes" if label == 1 else "No"
    else:
        answer = _string(label)
    return {
        "text": _format_prompt(
            _string(row.get("question", "Answer the question based on the passage.")),
            _string(row.get("passage", "")),
            answer,
        )
    }


def _format_prompt(instruction: Any, context: Any, answer: Any) -> str:
    return (
        "### Instruction:\n"
        f"{_string(instruction)}\n\n"
        "### Context:\n"
        f"{_string(context)}\n\n"
        "### Response:\n"
        f"{_string(answer)}<|end_of_text|>"
    )


FORMATTERS = {
    "medical": format_medical,
    "finance": format_finance,
    "boolq": format_boolq,
}


def tokenize_function(
    examples: dict[str, list[str]], tokenizer: Any, max_length: int
) -> dict[str, list[list[int]]]:
    encoded = tokenizer(
        examples["text"],
        truncation=True,
        max_length=max_length,
        padding="max_length",
    )
    encoded["labels"] = [
        [token_id if attention else -100 for token_id, attention in zip(ids, mask)]
        for ids, mask in zip(encoded["input_ids"], encoded["attention_mask"])
    ]
    return encoded


def format_and_tokenize(
    dataset: Dataset, formatter: Any, tokenizer: Any, max_length: int, num_proc: int
) -> Dataset:
    original_columns = dataset.column_names
    formatted = dataset.map(formatter, remove_columns=original_columns)
    text_columns = formatted.column_names
    token_mapper = partial(tokenize_function, tokenizer=tokenizer, max_length=max_length)
    process_count = max(1, min(num_proc, os.cpu_count() or 1))
    try:
        return formatted.map(
            token_mapper,
            batched=True,
            remove_columns=text_columns,
            num_proc=process_count if process_count > 1 else None,
            desc="Tokenizing formatted samples",
        )
    except Exception as error:
        if process_count <= 1:
            raise
        LOGGER.warning("Multiprocess tokenization failed; retrying in one process: %s", error)
        return formatted.map(
            token_mapper,
            batched=True,
            remove_columns=text_columns,
            desc="Tokenizing formatted samples",
        )


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )
    config = load_config()
    output_dir = (PROJECT_ROOT / config["output_dir"]).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    LOGGER.info("Loading tokenizer %s", config["tokenizer_name"])
    tokenizer = AutoTokenizer.from_pretrained(config["tokenizer_name"])
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    LOGGER.info("Tokenizer ready; maximum sequence length is %s", config["max_seq_length"])

    for task_index, task in enumerate(TASKS, start=1):
        LOGGER.info("Starting task %d/3: %s", task_index, task["domain"])
        loaded, source_name = load_first_available(task["sources"])
        train_raw, val_raw = choose_splits(
            loaded,
            config["samples_per_task_train"],
            config["samples_per_task_val"],
            config["seed"] + task_index,
        )
        formatter = FORMATTERS[task["formatter"]]
        task_dir = output_dir / task["name"]
        for split_name, raw_split in (("train", train_raw), ("val", val_raw)):
            LOGGER.info(
                "Formatting and tokenizing %s/%s (%d rows) from %s",
                task["name"],
                split_name,
                len(raw_split),
                source_name,
            )
            processed = format_and_tokenize(
                raw_split,
                formatter,
                tokenizer,
                config["max_seq_length"],
                config.get("num_proc", 1),
            )
            split_dir = task_dir / split_name
            processed.save_to_disk(str(split_dir))
            LOGGER.info("Saved %d rows to %s", len(processed), split_dir)
        LOGGER.info("[SUCCESS] Finished %s", task["name"])

    LOGGER.info("[SUCCESS] All tasks saved under %s", output_dir)


if __name__ == "__main__":
    main()