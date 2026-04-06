# validation.py
# -*- coding: utf-8 -*-
"""
Модуль валидации моделей анализа тональности.

Обеспечивает честную оценку качества моделей на отложенной тестовой выборке,
которая не участвовала в обучении и подборе гиперпараметров.

Функции:
    - split_labeled_data: стратифицированное разбиение размеченных данных
    - evaluate_baseline: оценка классической модели (TF-IDF + LR)
    - evaluate_rubert: оценка дообученного RuBERT
    - generate_validation_report: создание HTML-отчёта с метриками и матрицами ошибок
    - run_full_validation: запуск полной процедуры валидации

Зависимости:
    pip install scikit-learn pandas matplotlib seaborn
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from io import BytesIO
import base64
from typing import Tuple, List, Dict, Any

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import joblib

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset

from preprocessing import clean_text, lemmatize
from sentiment_models import RuBertSentimentModel
from exceptions import ValidationError, ModelError
from utils import logger, ensure_directory_exists

# ----------------------------------------------------------------------
# 1. Разбиение данных
# ----------------------------------------------------------------------
def split_labeled_data(
    csv_path: str = "labeled_data.csv",
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    random_state: int = 42
) -> Tuple[List[str], List[int], List[str], List[int], List[str], List[int]]:
    """
    Загружает размеченные данные и разбивает на train/val/test.

    Args:
        csv_path: путь к CSV с колонками 'text' и 'label'
        train_ratio: доля обучающей выборки
        val_ratio: доля валидационной выборки
        test_ratio: доля тестовой выборки
        random_state: seed для воспроизводимости

    Returns:
        (train_texts, train_labels, val_texts, val_labels, test_texts, test_labels)

    Raises:
        ValidationError: при некорректных данных или невозможности разбить
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Файл {csv_path} не найден. Сначала выполните разметку.")

    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        raise ValidationError(f"Не удалось прочитать {csv_path}: {e}")

    df = df.dropna(subset=['text', 'label'])
    df = df[df['text'].str.strip() != ""]

    if len(df) < 10:
        raise ValidationError(f"Недостаточно размеченных данных: {len(df)} примеров (нужно минимум 10)")

    texts = df['text'].tolist()
    labels = df['label'].tolist()

    # Сначала отделяем тестовую выборку
    train_val_texts, test_texts, train_val_labels, test_labels = train_test_split(
        texts, labels,
        test_size=test_ratio,
        random_state=random_state,
        stratify=labels
    )

    # Затем из оставшихся отделяем валидационную
    val_ratio_adjusted = val_ratio / (train_ratio + val_ratio)
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        train_val_texts, train_val_labels,
        test_size=val_ratio_adjusted,
        random_state=random_state,
        stratify=train_val_labels
    )

    logger.info(f"Разбиение данных: train={len(train_texts)}, val={len(val_texts)}, test={len(test_texts)}")
    return train_texts, train_labels, val_texts, val_labels, test_texts, test_labels


# ----------------------------------------------------------------------
# 2. Оценка Baseline модели (TF‑IDF + Logistic Regression)
# ----------------------------------------------------------------------
def evaluate_baseline(
    train_texts: List[str],
    train_labels: List[int],
    test_texts: List[str],
    test_labels: List[int],
    vectorizer_path: str = "models/tfidf_vectorizer.pkl",
    model_path: str = "models/baseline_model.pkl"
) -> Dict[str, Any]:
    """
    Обучает или загружает baseline модель (TF‑IDF + LR) и оценивает на тесте.

    Returns:
        dict с метриками: accuracy, macro_f1, weighted_f1, per_class_f1,
        classification_report (dict), predictions
    """
    try:
        # Предобработка
        train_cleaned = [clean_text(t) for t in train_texts]
        train_lemmatized = [lemmatize(t) for t in train_cleaned]
        test_cleaned = [clean_text(t) for t in test_texts]
        test_lemmatized = [lemmatize(t) for t in test_cleaned]

        ensure_directory_exists(os.path.dirname(vectorizer_path))

        # Обучение или загрузка
        if os.path.exists(vectorizer_path) and os.path.exists(model_path):
            logger.info("Загрузка существующей baseline модели...")
            vectorizer = joblib.load(vectorizer_path)
            model = joblib.load(model_path)
        else:
            logger.info("Обучение baseline модели...")
            vectorizer = TfidfVectorizer(max_features=5000)
            X_train = vectorizer.fit_transform(train_lemmatized)
            model = LogisticRegression(max_iter=1000, class_weight='balanced')
            model.fit(X_train, train_labels)
            joblib.dump(vectorizer, vectorizer_path)
            joblib.dump(model, model_path)

        # Предсказание на тесте
        X_test = vectorizer.transform(test_lemmatized)
        preds = model.predict(X_test)

        # Метрики
        accuracy = accuracy_score(test_labels, preds)
        precision, recall, f1, _ = precision_recall_fscore_support(
            test_labels, preds, average=None, labels=[0, 1, 2]
        )
        macro_f1 = np.mean(f1)
        weighted_f1 = precision_recall_fscore_support(test_labels, preds, average='weighted')[2]

        report_dict = classification_report(test_labels, preds, output_dict=True, labels=[0, 1, 2])

        logger.info(f"Baseline: accuracy={accuracy:.3f}, macro_f1={macro_f1:.3f}")

        return {
            'accuracy': accuracy,
            'macro_f1': macro_f1,
            'weighted_f1': weighted_f1,
            'per_class': {'neg': f1[0], 'neutral': f1[1], 'pos': f1[2]},
            'classification_report': report_dict,
            'predictions': preds
        }
    except Exception as e:
        logger.error(f"Ошибка в evaluate_baseline: {e}", exc_info=True)
        raise ModelError(f"Не удалось выполнить оценку baseline: {e}")


# ----------------------------------------------------------------------
# 3. Оценка RuBERT модели (с дообучением)
# ----------------------------------------------------------------------
def evaluate_rubert(
    train_texts: List[str],
    train_labels: List[int],
    val_texts: List[str],
    val_labels: List[int],
    test_texts: List[str],
    test_labels: List[int],
    fine_tuned_model_path: str = "./fine_tuned_model"
) -> Dict[str, Any]:
    """
    Дообучает RuBERT на train+val (если модель не существует) или загружает,
    затем оценивает на тесте.

    Returns:
        dict с метриками (аналогично evaluate_baseline)
    """
    try:
        # Если модель уже дообучена, загружаем и предсказываем
        if os.path.exists(fine_tuned_model_path):
            logger.info(f"Загрузка дообученной модели из {fine_tuned_model_path}")
            rubert = RuBertSentimentModel(model_path=fine_tuned_model_path)
            results = rubert.predict_batch(test_texts)
            preds = [0 if r[0] == 'negative' else 1 if r[0] == 'neutral' else 2 for r in results]
        else:
            logger.info("Дообучение RuBERT на train+val...")
            # Объединяем train и val для обучения
            train_val_texts = train_texts + val_texts
            train_val_labels = train_labels + val_labels

            # Токенизация
            model_name = "sunny3/rubert-conversational-sentiment-balanced"
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=3)

            def tokenize(texts, labels):
                encodings = tokenizer(texts, truncation=True, padding=True, max_length=128)
                return Dataset.from_dict({
                    'input_ids': encodings['input_ids'],
                    'attention_mask': encodings['attention_mask'],
                    'labels': labels
                })

            train_dataset = tokenize(train_val_texts, train_val_labels)
            val_dataset = tokenize(val_texts, val_labels)

            training_args = TrainingArguments(
                output_dir='./rubert_val_results',
                num_train_epochs=3,
                per_device_train_batch_size=8,
                per_device_eval_batch_size=16,
                warmup_steps=100,
                weight_decay=0.01,
                eval_strategy="epoch",
                save_strategy="epoch",
                load_best_model_at_end=True,
                metric_for_best_model="f1",
                logging_dir='./logs',
            )

            def compute_metrics(eval_pred):
                logits, labels = eval_pred
                preds = np.argmax(logits, axis=-1)
                f1 = precision_recall_fscore_support(labels, preds, average='weighted')[2]
                acc = accuracy_score(labels, preds)
                return {'accuracy': acc, 'f1': f1}

            trainer = Trainer(
                model=model,
                args=training_args,
                train_dataset=train_dataset,
                eval_dataset=val_dataset,
                compute_metrics=compute_metrics,
            )
            trainer.train()

            # Сохраняем лучшую модель
            os.makedirs(fine_tuned_model_path, exist_ok=True)
            model.save_pretrained(fine_tuned_model_path)
            tokenizer.save_pretrained(fine_tuned_model_path)

            # Предсказание на тесте
            rubert = RuBertSentimentModel(model_path=fine_tuned_model_path)
            results = rubert.predict_batch(test_texts)
            preds = [0 if r[0] == 'negative' else 1 if r[0] == 'neutral' else 2 for r in results]

        # Метрики
        accuracy = accuracy_score(test_labels, preds)
        precision, recall, f1, _ = precision_recall_fscore_support(
            test_labels, preds, average=None, labels=[0, 1, 2]
        )
        macro_f1 = np.mean(f1)
        weighted_f1 = precision_recall_fscore_support(test_labels, preds, average='weighted')[2]

        logger.info(f"RuBERT: accuracy={accuracy:.3f}, macro_f1={macro_f1:.3f}")

        return {
            'accuracy': accuracy,
            'macro_f1': macro_f1,
            'weighted_f1': weighted_f1,
            'per_class': {'neg': f1[0], 'neutral': f1[1], 'pos': f1[2]},
            'classification_report': classification_report(test_labels, preds, output_dict=True, labels=[0, 1, 2]),
            'predictions': preds
        }
    except Exception as e:
        logger.error(f"Ошибка в evaluate_rubert: {e}", exc_info=True)
        raise ModelError(f"Не удалось выполнить оценку RuBERT: {e}")


# ----------------------------------------------------------------------
# 4. Генерация HTML-отчёта
# ----------------------------------------------------------------------
def generate_validation_report(
    baseline_metrics: Dict[str, Any],
    rubert_metrics: Dict[str, Any],
    test_labels: List[int],
    output_file: str = "validation_report.html"
) -> None:
    """
    Создаёт HTML-отчёт с результатами валидации, включая матрицы ошибок.
    """
    try:
        def fig_to_base64(fig):
            buf = BytesIO()
            fig.savefig(buf, format='png', bbox_inches='tight')
            buf.seek(0)
            img_base64 = base64.b64encode(buf.read()).decode('utf-8')
            buf.close()
            return img_base64

        # Матрица ошибок для baseline
        cm_baseline = confusion_matrix(test_labels, baseline_metrics['predictions'], labels=[0, 1, 2])
        fig_baseline, ax_baseline = plt.subplots(figsize=(6, 5))
        disp_baseline = ConfusionMatrixDisplay(confusion_matrix=cm_baseline, display_labels=['Негатив', 'Нейтраль', 'Позитив'])
        disp_baseline.plot(ax=ax_baseline, cmap='Blues')
        ax_baseline.set_title('Baseline (TF‑IDF + LR)')
        img_baseline = fig_to_base64(fig_baseline)
        plt.close(fig_baseline)

        # Матрица ошибок для RuBERT
        cm_rubert = confusion_matrix(test_labels, rubert_metrics['predictions'], labels=[0, 1, 2])
        fig_rubert, ax_rubert = plt.subplots(figsize=(6, 5))
        disp_rubert = ConfusionMatrixDisplay(confusion_matrix=cm_rubert, display_labels=['Негатив', 'Нейтраль', 'Позитив'])
        disp_rubert.plot(ax=ax_rubert, cmap='Greens')
        ax_rubert.set_title('RuBERT (fine-tuned)')
        img_rubert = fig_to_base64(fig_rubert)
        plt.close(fig_rubert)

        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="UTF-8">
            <title>Валидация моделей анализа тональности</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; background: #f5f5f5; }}
                h1, h2 {{ color: #2c3e50; }}
                .container {{ display: flex; flex-wrap: wrap; gap: 20px; }}
                .card {{ background: white; border-radius: 8px; padding: 15px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); flex: 1; min-width: 300px; }}
                .metric {{ font-size: 1.2em; margin: 10px 0; }}
                .good {{ color: #27ae60; }}
                .bad {{ color: #e74c3c; }}
                table {{ border-collapse: collapse; width: 100%; margin-top: 10px; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #2c3e50; color: white; }}
                img {{ max-width: 100%; height: auto; margin-top: 10px; }}
            </style>
        </head>
        <body>
            <h1>📊 Отчёт о валидации моделей анализа тональности</h1>
            <p>Оценка выполнена на отложенной тестовой выборке, не участвовавшей в обучении.</p>

            <div class="container">
                <div class="card">
                    <h2>Baseline (TF‑IDF + Logistic Regression)</h2>
                    <div class="metric">Accuracy: <b>{baseline_metrics['accuracy']:.3f}</b></div>
                    <div class="metric">Macro F1: <b>{baseline_metrics['macro_f1']:.3f}</b></div>
                    <div class="metric">Weighted F1: <b>{baseline_metrics['weighted_f1']:.3f}</b></div>
                    <div>F1 по классам:</div>
                    <ul>
                        <li>Негатив: {baseline_metrics['per_class']['neg']:.3f}</li>
                        <li>Нейтраль: {baseline_metrics['per_class']['neutral']:.3f}</li>
                        <li>Позитив: {baseline_metrics['per_class']['pos']:.3f}</li>
                    </ul>
                    <img src="data:image/png;base64,{img_baseline}" alt="Confusion Matrix Baseline">
                </div>
                <div class="card">
                    <h2>RuBERT (fine-tuned)</h2>
                    <div class="metric">Accuracy: <b>{rubert_metrics['accuracy']:.3f}</b></div>
                    <div class="metric">Macro F1: <b>{rubert_metrics['macro_f1']:.3f}</b></div>
                    <div class="metric">Weighted F1: <b>{rubert_metrics['weighted_f1']:.3f}</b></div>
                    <div>F1 по классам:</div>
                    <ul>
                        <li>Негатив: {rubert_metrics['per_class']['neg']:.3f}</li>
                        <li>Нейтраль: {rubert_metrics['per_class']['neutral']:.3f}</li>
                        <li>Позитив: {rubert_metrics['per_class']['pos']:.3f}</li>
                    </ul>
                    <img src="data:image/png;base64,{img_rubert}" alt="Confusion Matrix RuBERT">
                </div>
            </div>
            <p style="margin-top: 20px; font-size: 0.8em; color: gray;">Отчёт сгенерирован автоматически модулем validation.py</p>
        </body>
        </html>
        """
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(html)
        logger.info(f"Валидационный отчёт сохранён в {output_file}")
    except Exception as e:
        logger.error(f"Ошибка генерации HTML-отчёта: {e}")
        print(f"Не удалось сгенерировать HTML-отчёт: {e}")


# ----------------------------------------------------------------------
# 5. Полный запуск валидации
# ----------------------------------------------------------------------
def run_full_validation(csv_path: str = "labeled_data.csv") -> None:
    """
    Запускает полную процедуру валидации обеих моделей.
    """
    logger.info("=== ЗАПУСК ВАЛИДАЦИИ ===")
    try:
        train_texts, train_labels, val_texts, val_labels, test_texts, test_labels = split_labeled_data(csv_path)

        logger.info("--- Оценка Baseline модели ---")
        baseline_metrics = evaluate_baseline(train_texts, train_labels, test_texts, test_labels)

        logger.info("--- Оценка RuBERT модели ---")
        rubert_metrics = evaluate_rubert(train_texts, train_labels, val_texts, val_labels, test_texts, test_labels)

        logger.info("--- Генерация отчёта ---")
        generate_validation_report(baseline_metrics, rubert_metrics, test_labels)

        logger.info("=== ВАЛИДАЦИЯ ЗАВЕРШЕНА УСПЕШНО ===")
    except Exception as e:
        logger.error(f"Ошибка в процессе валидации: {e}", exc_info=True)
        raise


# ----------------------------------------------------------------------
# 6. Точка входа
# ----------------------------------------------------------------------
if __name__ == "__main__":
    run_full_validation()