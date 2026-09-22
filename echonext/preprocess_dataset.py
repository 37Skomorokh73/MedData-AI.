import os
from pathlib import Path
import pandas as pd
import numpy as np
import wfdb
from scipy.signal import resample_poly, butter, sosfiltfilt

BASE_DIR = Path(os.getcwd())
# Используем тестовую метадату (для демо-версии берем файл EchoNext)
PTBXL_ROOT = BASE_DIR / "Demo_Data"
OUTPUT_DIR = BASE_DIR / "Demo_Data" / "ptbxl_processed"
RECORDS_DIR = OUTPUT_DIR / "records"

TARGET_FS = 250
LOWCUT = 0.5
HIGHCUT = 40.0
FILTER_ORDER = 4


def preprocess_ecg(signal: np.ndarray, original_fs: int) -> np.ndarray:
    if original_fs == 500 and TARGET_FS == 250:
        processed = resample_poly(signal, up=1, down=2, axis=0).astype(np.float32)
    elif original_fs == TARGET_FS:
        processed = signal.astype(np.float32)
    else:
        raise ValueError(f"Неподдерживаемая частота: {original_fs}")

    nyquist = 0.5 * TARGET_FS
    sos = butter(FILTER_ORDER, [LOWCUT / nyquist, HIGHCUT / nyquist], btype="bandpass", output="sos")
    processed = sosfiltfilt(sos, processed, axis=0).astype(np.float32)

    mean_per_lead = np.mean(processed, axis=0, keepdims=True)
    std_per_lead = np.std(processed, axis=0, keepdims=True)
    std_per_lead = np.where(std_per_lead < 1e-8, 1.0, std_per_lead)

    return ((processed - mean_per_lead) / std_per_lead).astype(np.float32)


def main():
    RECORDS_DIR.mkdir(parents=True, exist_ok=True)

    # В демо-версии мы просто обрабатываем файлы, которые уже лежат в RECORDS_DIR,
    # поэтому этот скрипт можно использовать как шаблон для полной базы.
    # Для текущей структуры (где файлы уже готовы) этот шаг можно пропустить при запуске.
    print(f"Директория для обработки готова: {RECORDS_DIR}")
    print("В демо-режиме файлы .npy уже скопированы в эту папку.")


if __name__ == "__main__":
    main()