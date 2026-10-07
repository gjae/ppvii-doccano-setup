from datasets import load_dataset
import json


def main():
    TEST_SPLIT = "test"
    TRAIN_SPLIT = "train"

    splits = [TEST_SPLIT, TRAIN_SPLIT]

    for split in splits:
        dataset = load_dataset("IEETA/SPACCC-Spanish-NER")

        # Si quieres guardarlo en formato JSONL para Doccano:
        with open(f"{split}_pharmaconer_huggingface.jsonl", "w", encoding="utf-8") as f:
            print(f"El dataset {split} tiene {dataset.num_rows} filas: ", dataset)


if __name__ == "__main__":
    main()
