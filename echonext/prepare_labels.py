import os
import pandas as pd
from pathlib import Path

BASE_DIR = Path(os.getcwd())
METADATA_PATH = BASE_DIR / "Demo_Data" / "echonext_metadata_100k.csv"
OUTPUT_FILE = BASE_DIR / "Demo_Data" / "EchoNext_MVP1_labels.csv"
TARGET_COL = 'shd_moderate_or_greater_flag'


def main():
    if not METADATA_PATH.exists():
        print(f"Ошибка: Метаданные не найдены по пути {METADATA_PATH}")
        return

    df = pd.read_csv(METADATA_PATH)

    df_clean = df.dropna(subset=[TARGET_COL]).copy()
    df_clean[TARGET_COL] = df_clean[TARGET_COL].astype(int)

    labels_df = df_clean[['ecg_key', 'split', TARGET_COL]]

    # Исправление: приведение ключа к единому стандарту
    labels_df = labels_df.rename(columns={'ecg_key': 'patient_id'})

    labels_df.to_csv(OUTPUT_FILE, index=False)
    print(f"Метки успешно сохранены: {OUTPUT_FILE.name}")


if __name__ == "__main__":
    main()