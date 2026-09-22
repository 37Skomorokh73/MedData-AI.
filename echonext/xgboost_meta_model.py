import os
import pandas as pd
import xgboost as xgb
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, average_precision_score, classification_report

BASE_DIR = Path(os.getcwd())
EMBEDDINGS_PATH = BASE_DIR / "Demo_Data" / "ECG_embeddings.csv"
LABELS_PATH = BASE_DIR / "Demo_Data" / "EchoNext_MVP1_labels.csv"


def main():
    if not EMBEDDINGS_PATH.exists() or not LABELS_PATH.exists():
        print(f"Ошибка: Необходимые файлы не найдены в {BASE_DIR / 'Demo_Data'}")
        return

    print("Загрузка данных...")
    df_emb = pd.read_csv(EMBEDDINGS_PATH)
    df_labels = pd.read_csv(LABELS_PATH)

    df_emb['patient_id'] = df_emb['patient_id'].astype(str)
    df_labels['patient_id'] = df_labels['patient_id'].astype(str)

    df_full = pd.merge(df_emb, df_labels, on='patient_id', how='inner')
    if df_full.empty:
        print("Ошибка: Слияние вернуло пустой результат. Проверьте индексы patient_id.")
        return

    print(f"Размер объединенного датасета: {len(df_full)} записей")

    feature_cols = [col for col in df_full.columns if col.startswith('emb_')]
    X = df_full[feature_cols]
    y = df_full['shd_moderate_or_greater_flag']

    # Если в демо-версии мало файлов, отключаем стратификацию, чтобы не было ошибки
    stratify_param = y if len(y) > 10 else None

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=stratify_param
    )

    model = xgb.XGBClassifier(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=4,
        eval_metric='auc',
        random_state=42
    )

    print("Старт обучения мета-модели...")
    model.fit(X_train, y_train)

    y_pred_proba = model.predict_proba(X_test)[:, 1]
    y_pred_class = model.predict(X_test)

    print("\nРезультаты валидации (Метрики):")
    try:
        print(f"AUROC: {roc_auc_score(y_test, y_pred_proba):.4f}")
        print(f"AUPRC: {average_precision_score(y_test, y_pred_proba):.4f}")
        print("\nОтчет по классификации:")
        print(classification_report(y_test, y_pred_class))
    except ValueError:
        print(
            "В тестовой выборке представлен только один класс. Загрузите больше данных для вычисления метрик ROC AUC.")


if __name__ == "__main__":
    main()