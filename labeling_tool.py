# labeling_tool.py
# -*- coding: utf-8 -*-
"""
Интерактивный инструмент для ручной разметки тональности текстов.

Запуск:
    python labeling_tool.py

Вход:
    - to_label.csv (подготовленный prepare_labeling.py)
    - (опционально) уже существующий labeled_data.csv (добавление новых разметок)

Выход:
    - labeled_data.csv (накапливает размеченные тексты)

Правила разметки:
    0 - негативная тональность
    1 - нейтральная тональность
    2 - позитивная тональность
    q - выход и сохранение
"""

import os
import sys
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils import logger, ensure_directory_exists

# ----------------------------------------------------------------------
# Константы
# ----------------------------------------------------------------------
TO_LABEL_CSV = "to_label.csv"
LABELED_CSV = "labeled_data.csv"


def load_unlabeled_data():
    """
    Загружает тексты из to_label.csv, которые ещё не размечены.
    Если labeled_data.csv существует, исключает уже размеченные тексты.
    """
    if not os.path.exists(TO_LABEL_CSV):
        logger.error(f"Файл {TO_LABEL_CSV} не найден. Сначала запустите prepare_labeling.py.")
        return []

    df = pd.read_csv(TO_LABEL_CSV)
    texts = df["text"].tolist()

    # Если уже есть размеченные, исключаем их
    if os.path.exists(LABELED_CSV):
        labeled_df = pd.read_csv(LABELED_CSV)
        labeled_texts = set(labeled_df["text"].tolist())
        texts = [t for t in texts if t not in labeled_texts]
        logger.info(f"Уже размечено {len(labeled_df)} текстов. Осталось разметить: {len(texts)}")
    else:
        logger.info(f"Нет существующей разметки. Будет размечено {len(texts)} текстов.")

    return texts


def save_labeled_data(text, label, output_file=LABELED_CSV):
    """Добавляет одну размеченную запись в CSV."""
    new_row = pd.DataFrame({"text": [text], "label": [label]})
    if os.path.exists(output_file):
        existing = pd.read_csv(output_file)
        updated = pd.concat([existing, new_row], ignore_index=True)
        updated.to_csv(output_file, index=False, encoding="utf-8-sig")
    else:
        new_row.to_csv(output_file, index=False, encoding="utf-8-sig")
    logger.debug(f"Сохранено: {text[:50]}... -> {label}")


def main():
    """Интерактивная разметка."""
    print("=" * 50)
    print("ИНСТРУМЕНТ РУЧНОЙ РАЗМЕТКИ ТОНАЛЬНОСТИ")
    print("=" * 50)
    print("\nПравила:")
    print("  0 - негативная тональность (критика, жалоба, раздражение, гнев)")
    print("  1 - нейтральная тональность (констатация фактов, без эмоций)")
    print("  2 - позитивная тональность (одобрение, благодарность, радость)")
    print("  q - выход и сохранение")
    print()

    texts = load_unlabeled_data()
    if not texts:
        print("Нет текстов для разметки. Все уже размечены или файл to_label.csv пуст.")
        return

    total = len(texts)
    for i, text in enumerate(texts):
        print(f"\n--- Текст {i+1}/{total} ---")
        # Показываем первые 500 символов
        print(text[:500] + ("..." if len(text) > 500 else ""))
        print()

        while True:
            choice = input("Тональность (0/1/2) или q: ").strip().lower()
            if choice == 'q':
                print("Выход. Прогресс сохранён.")
                return
            if choice in ('0', '1', '2'):
                save_labeled_data(text, int(choice))
                print("✓ Сохранено.")
                break
            else:
                print("Неверный ввод. Введите 0, 1, 2 или q.")

    print("\n✅ Все тексты размечены! Файл", LABELED_CSV, "обновлён.")
    print("Теперь можно запустить finetune_rubert.py для дообучения модели.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nПрерывание пользователя. Прогресс сохранён.")
    except Exception as e:
        logger.error(f"Ошибка: {e}", exc_info=True)
        sys.exit(1)