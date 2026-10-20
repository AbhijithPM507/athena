"""Validate saved ATHENA datasets and print one decoded sample per task."""

from __future__ import annotations

import logging
from pathlib import Path

import yaml
from datasets import load_from_disk
from transformers import AutoTokenizer


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TASKS = (
    "task_1_medical",
    "task_2_legal_finance",
    "task_3_superglue_boolq",
)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    with (PROJECT_ROOT / "config.yaml").open(encoding="utf-8") as config_file:
        config = yaml.safe_load(config_file)
    output_dir = (PROJECT_ROOT / config["output_dir"]).resolve()
    tokenizer = AutoTokenizer.from_pretrained(config["tokenizer_name"])
    sequence_length = config["max_seq_length"]

    for task_name in TASKS:
        logging.info("Checking %s", task_name)
        for split_name in ("train", "val"):
            dataset_path = output_dir / task_name / split_name
            dataset = load_from_disk(str(dataset_path))
            print(f"[INFO] {task_name}/{split_name}: {len(dataset)} samples")
            if len(dataset) == 0:
                raise AssertionError(f"Dataset is empty: {dataset_path}")
            for column in ("input_ids", "attention_mask", "labels"):
                if column not in dataset.column_names:
                    raise AssertionError(f"Missing {column} in {dataset_path}")
                if any(len(row) != sequence_length for row in dataset[column]):
                    raise AssertionError(
                        f"{dataset_path} has a {column} row with length other "
                        f"than {sequence_length}"
                    )

        sample = load_from_disk(str(output_dir / task_name / "train"))[0]
        visible_labels = [token_id for token_id in sample["labels"] if token_id != -100]
        masked_count = sum(token_id == -100 for token_id in sample["labels"])
        padding_positions = [
            index for index, attention in enumerate(sample["attention_mask"]) if attention == 0
        ]
        print(f"[INFO] {task_name} decoded sample:")
        print(tokenizer.decode(visible_labels, skip_special_tokens=False))
        print(f"[INFO] Masked padding tokens: {masked_count}")
        if any(sample["labels"][index] != -100 for index in padding_positions):
            raise AssertionError(f"Padding labels are not masked in {task_name}")
        if any(
            label == -100
            for label, attention in zip(sample["labels"], sample["attention_mask"])
            if attention == 1
        ):
            raise AssertionError(f"A non-padding label is masked in {task_name}")

    print("[SUCCESS] All saved datasets passed shape and label-mask checks.")


if __name__ == "__main__":
    main()