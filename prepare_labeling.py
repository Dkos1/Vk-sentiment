# prepare_labeling.py
# -*- coding: utf-8 -*-
"""
Скрипт для подготовки данных к ручной разметке тональности.

Из результатов анализа (vk_sentiment_results.csv) выбирает случайные тексты
(посты и комментарии) и сохраняет их в CSV-файл to_label.csv.
Этот файл затем используется labeling_tool.py.

Запуск:
    python prepare_labeling.py

Вход:
    - output/vk_sentiment_results.csv (результаты работы main.py)

Выход:
    - to_label.csv (колонки: original_text, group, type, url)
"""

import os
import sys
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import OUTPUT_DIR, RANDOM_SEED
from utils import logger, ensure_directory_exists

# ----------------------------------------------------------------------
# Константы
# ----------------------------------------------------------------------
INPUT_CSV = os.path.join(OUTPUT_DIR, "vk_sentiment_results.csv")
OUTPUT_CSV = "to_label.csv"
SAMPLE_SIZE = 1000  # максимальное количество текстов для разметки


def prepare_labeling_data():
    """Основная функция подготовки."""
    if not os.path.exists(INPUT_CSV):
        logger.error(f"Файл {INPUT_CSV} не найден. Сначала запустите main.py для сбора данных.")
        return

    logger.info(f"Загрузка данных из {INPUT_CSV}...")
    df = pd.read_csv(INPUT_CSV)

    # Удаляем дубликаты текстов (если есть)
    df = df.drop_duplicates(subset=["original_text"])
    logger.info(f"Всего уникальных текстов: {len(df)}")

    # Выбираем случайную выборку (но не больше SAMPLE_SIZE)
    sample = df.sample(n=min(SAMPLE_SIZE, len(df)), random_state=RANDOM_SEED)

    # Оставляем нужные колонки для разметки
    output_df = sample[["original_text", "group", "type", "url"]].copy()
    output_df.columns = ["text", "group", "type", "url"]  # переименовываем для удобства

    ensure_directory_exists(os.path.dirname(OUTPUT_CSV) if os.path.dirname(OUTPUT_CSV) else ".")
    output_df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")
    logger.info(f"Подготовлено {len(output_df)} текстов для разметки. Файл сохранён: {OUTPUT_CSV}")
    logger.info("Теперь запустите labeling_tool.py для ручной разметки.")


if __name__ == "__main__":
    prepare_labeling_data()