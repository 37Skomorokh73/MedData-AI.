import os
import sys
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from pathlib import Path

BASE_DIR = Path(os.getcwd())
ECG_DIR = BASE_DIR / "Demo_Data" / "ptbxl_processed" / "records"
OUTPUT_FILE = BASE_DIR / "Demo_Data" / "ECG_embeddings.csv"

# Подключение архитектуры EchoNext-Mini
ECHONEXT_DIR = BASE_DIR / "Demo_Data" / "cradlenet"
sys.path.append(str(BASE_DIR / "Demo_Data"))
WEIGHTS_PATH = ECHONEXT_DIR / "weights.pt"

try:
    from cradlenet.models.resnet1d_tabular import ResNet1dWithTabular
except ImportError:
    print("Ошибка импорта: Убедитесь, что папка cradlenet лежит в Demo_Data.")
    sys.exit(1)


def initialize_extractor(device: torch.device) -> nn.Module:
    model = ResNet1dWithTabular(base_filters=16, tabular_features=7)

    if WEIGHTS_PATH.exists():
        checkpoint = torch.load(WEIGHTS_PATH, map_location=device, weights_only=True)
        model.load_state_dict(checkpoint.get("model", checkpoint), strict=False)
    else:
        print(f"Предупреждение: Веса не найдены по пути {WEIGHTS_PATH}")

    if hasattr(model, 'fc'):
        model.fc = nn.Identity()

    model.to(device)
    model.eval()
    return model


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = initialize_extractor(device)
    ecg_files = list(ECG_DIR.glob('*.npy'))

    if not ecg_files:
        print(f"Файлы ЭКГ не найдены в {ECG_DIR}")
        return

    all_embeddings, patient_ids = [], []

    with torch.no_grad():
        for filepath in ecg_files:
            # Исправление: очистка ID от префикса 'ecg_' для точного совпадения с разметкой
            clean_id = str(int(filepath.stem.replace("ecg_", "")))

            signal = np.load(filepath)
            if signal.shape == (2500, 12):
                signal = signal.T

            tensor = torch.tensor(signal, dtype=torch.float32).unsqueeze(0).to(device)
            dummy_tabular = torch.zeros((1, 7), dtype=torch.float32).to(device)

            embedding = model((tensor, dummy_tabular))
            all_embeddings.append(embedding.cpu().numpy().flatten())
            patient_ids.append(clean_id)

    emb_size = len(all_embeddings[0])
    df_embeddings = pd.DataFrame(all_embeddings, columns=[f"emb_{j}" for j in range(emb_size)])
    df_embeddings.insert(0, 'patient_id', patient_ids)

    df_embeddings.to_csv(OUTPUT_FILE, index=False)
    print(f"Экстракция завершена. Файл сохранен: {OUTPUT_FILE.name}")


if __name__ == "__main__":
    main()