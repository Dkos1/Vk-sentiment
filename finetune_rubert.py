# finetune_rubert.py
# -*- coding: utf-8 -*-
"""
Скрипт для дообучения (fine-tuning) модели RuBERT на размеченных данных.

Запуск:
    python finetune_rubert.py

Вход:
    - labeled_data.csv (колонки: text, label)
    - (опционально) модель-основа sunny3/rubert-conversational-sentiment-balanced

Выход:
    - fine_tuned_model/ (папка с моделью и токенизатором)

Зависимости:
    pip install transformers datasets scikit-learn pandas torch
"""

import os
import sys
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments
)
from datasets import Dataset
import torch

# Добавляем путь к проекту для импорта config и utils
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import MODELS_DIR, RANDOM_SEED, ensure_directories
from utils import logger, ensure_directory_exists

# ----------------------------------------------------------------------
# Константы
# ----------------------------------------------------------------------
DEFAULT_MODEL_NAME = "sunny3/rubert-conversational-sentiment-balanced"
OUTPUT_MODEL_DIR = os.path.join(MODELS_DIR, "fine_tuned_model")
LABELED_CSV = "vk-sentiment-web v2.0/data/labeled_data.csv"
TRAIN_RATIO = 0.8
VAL_RATIO = 0.2
MAX_LENGTH = 128          # длина последовательности (для коротких текстов комментариев)
BATCH_SIZE = 8
EPOCHS = 5
LEARNING_RATE = 2e-5
WEIGHT_DECAY = 0.01


# ----------------------------------------------------------------------
# Загрузка и подготовка данных
# ----------------------------------------------------------------------
def load_and_prepare_data(csv_path: str = LABELED_CSV):
    """
    Загружает размеченные данные и разбивает на train/val.

    Returns:
        train_dataset, val_dataset (объекты Dataset от Hugging Face)
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Файл {csv_path} не найден. Сначала выполните разметку (labeling_tool.py).")

    df = pd.read_csv(csv_path)
    df = df.dropna(subset=['text', 'label'])
    df = df[df['text'].str.strip() != ""]
    df['label'] = df['label'].astype(int)

    logger.info(f"Загружено {len(df)} размеченных текстов")
    logger.info(f"Распределение классов:\n{df['label'].value_counts().to_string()}")

    if len(df) < 10:
        raise ValueError(f"Слишком мало данных для дообучения: {len(df)}. Нужно минимум 10 примеров.")

    # Разделение на train/val
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        df['text'].tolist(),
        df['label'].tolist(),
        test_size=VAL_RATIO,
        random_state=RANDOM_SEED,
        stratify=df['label'].tolist()
    )

    logger.info(f"Train: {len(train_texts)}, Val: {len(val_texts)}")
    return train_texts, train_labels, val_texts, val_labels


# ----------------------------------------------------------------------
# Токенизация
# ----------------------------------------------------------------------
def tokenize_dataset(tokenizer, texts, labels, max_length=MAX_LENGTH):
    """Токенизирует тексты и возвращает Dataset."""
    encodings = tokenizer(
        texts,
        truncation=True,
        padding=True,
        max_length=max_length
    )
    return Dataset.from_dict({
        'input_ids': encodings['input_ids'],
        'attention_mask': encodings['attention_mask'],
        'labels': labels
    })


# ----------------------------------------------------------------------
# Метрики для Trainer
# ----------------------------------------------------------------------
def compute_metrics(eval_pred):
    """Вычисляет accuracy и weighted F1 для валидации."""
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    acc = accuracy_score(labels, predictions)
    f1 = f1_score(labels, predictions, average='weighted')
    return {'accuracy': acc, 'f1': f1}


# ----------------------------------------------------------------------
# Основная функция дообучения
# ----------------------------------------------------------------------
def fine_tune():
    """Запускает процесс дообучения."""
    logger.info("=== НАЧАЛО ДООБУЧЕНИЯ RUBERT ===")

    # Создаём директории
    ensure_directories()
    ensure_directory_exists(OUTPUT_MODEL_DIR)

    # 1. Загрузка данных
    train_texts, train_labels, val_texts, val_labels = load_and_prepare_data()

    # 2. Загрузка модели и токенизатора
    logger.info(f"Загрузка модели {DEFAULT_MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(DEFAULT_MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(
        DEFAULT_MODEL_NAME,
        num_labels=3  # негатив, нейтраль, позитив
    )

    # 3. Токенизация
    logger.info("Токенизация данных...")
    train_dataset = tokenize_dataset(tokenizer, train_texts, train_labels)
    val_dataset = tokenize_dataset(tokenizer, val_texts, val_labels)

    # 4. Настройка аргументов обучения
    training_args = TrainingArguments(
        output_dir=os.path.join(MODELS_DIR, "rubert_checkpoints"),
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE * 2,
        warmup_steps=100,
        weight_decay=WEIGHT_DECAY,
        logging_dir=os.path.join(MODELS_DIR, "logs"),
        logging_steps=50,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        learning_rate=LEARNING_RATE,
        save_total_limit=2,
        report_to="none"  # отключаем wandb/tensorboard для чистоты
    )

    # 5. Trainer и обучение
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
    )

    logger.info("Начинаем обучение...")
    trainer.train()

    # 6. Сохранение лучшей модели
    logger.info(f"Сохранение дообученной модели в {OUTPUT_MODEL_DIR}")
    model.save_pretrained(OUTPUT_MODEL_DIR)
    tokenizer.save_pretrained(OUTPUT_MODEL_DIR)

    # 7. Финальная оценка на валидации
    eval_results = trainer.evaluate()
    logger.info(f"Результаты валидации: accuracy={eval_results.get('eval_accuracy', 0):.3f}, "
                f"f1={eval_results.get('eval_f1', 0):.3f}")

    logger.info("=== ДООБУЧЕНИЕ ЗАВЕРШЕНО ===")


# ----------------------------------------------------------------------
# Точка входа
# ----------------------------------------------------------------------
if __name__ == "__main__":
    try:
        fine_tune()
    except Exception as e:
        logger.error(f"Ошибка при дообучении: {e}", exc_info=True)
        sys.exit(1)