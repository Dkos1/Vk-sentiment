# preprocessing.py
# -*- coding: utf-8 -*-
"""
Модуль предобработки текстов для анализа тональности сообщений ВКонтакте.

Основные функции:
    - Очистка текста от шума (URL, упоминания, спецсимволы)
    - Нормализация эмодзи (замена на текстовые дескрипторы, например 😊 → :smiling_face:)
    - Нормализация текстовых смайлов (:) → :smiling_face:)
    - Лемматизация для классических моделей (с помощью pymystem3)

Зависимости:
    pip install emoji pymystem3

Пример использования:
    >>> from preprocessing import clean_text, lemmatize
    >>> text = "Привет! 😊 Как дела? :( Ужасно..."
    >>> cleaned = clean_text(text)
    >>> print(cleaned)
    'привет :smiling_face: как дела :crying_face: ужасно'
"""

import re
from typing import Optional

# ----------------------------------------------------------------------
# Импорт зависимостей с обработкой отсутствия
# ----------------------------------------------------------------------
try:
    import emoji
    HAS_EMOJI = True
except ImportError:
    HAS_EMOJI = False
    print("Библиотека 'emoji' не установлена. Установите: pip install emoji")

try:
    from pymystem3 import Mystem
    mystem = Mystem()
    HAS_MYSTEM = True
except ImportError:
    HAS_MYSTEM = False
    print("pymystem3 не установлена, лемматизация отключена. Установите: pip install pymystem3")

# ----------------------------------------------------------------------
# Конфигурация
# ----------------------------------------------------------------------
# Регулярное выражение для удаления спецсимволов.
# Разрешены: буквы (русские/английские), цифры, пробелы, двоеточие, подчёркивание.
# Двоеточие и подчёркивание нужны для сохранения меток эмодзи :smiling_face:
CLEAN_REGEX = r"[^a-zA-Zа-яёА-ЯЁ0-9\s:_]"

# Словарь для замены текстовых смайлов (эмотиконов) на унифицированные метки
TEXT_SMILEYS = {
    # Позитивные
    r":-\)": " :smiling_face:",
    r"=\)": " :smiling_face:",
    r":\)": " :smiling_face:",
    r"\(:": " :smiling_face:",
    r":-D": " :laughing_face:",
    r":D": " :laughing_face:",
    r"=D": " :laughing_face:",
    r";-\)": " :winking_face:",
    r";\)": " :winking_face:",
    r":-P": " :playful_face:",
    r":P": " :playful_face:",
    r"=P": " :playful_face:",
    # Негативные
    r":-\(": " :crying_face:",
    r":\(": " :crying_face:",
    r"=\(": " :crying_face:",
    r":-S": " :worried_face:",
    r":S": " :worried_face:",
    # Нейтральные / прочие
    r":-\|": " :neutral_face:",
    r":\|": " :neutral_face:",
    r":-o": " :surprised_face:",
    r":o": " :surprised_face:",
    r":-\*": " :kiss_face:",
    r":\*": " :kiss_face:",
}

# ----------------------------------------------------------------------
# Основные функции
# ----------------------------------------------------------------------

def normalize_emojis(text: str) -> str:
    """
    Заменяет все emoji (пиктограммы) на текстовые дескрипторы.
    Например, '😊' -> ':smiling_face:', '❤️' -> ':red_heart:'.
    
    Args:
        text (str): Исходный текст, содержащий эмодзи.
    
    Returns:
        str: Текст с заменёнными эмодзи на текстовые метки.
    
    Note:
        Используется библиотека emoji (функция demojize).
        Если библиотека не установлена, возвращается исходный текст.
    """
    if not HAS_EMOJI:
        return text
    try:
        # emoji.demojize заменяет эмодзи на :имя: (например 😊 → :smiling_face:)
        return emoji.demojize(text, delimiters=(":", ":"))
    except Exception:
        # В случае любой ошибки (например, неподдерживаемый символ) возвращаем исходный текст
        return text

def normalize_text_smileys(text: str) -> str:
    """
    Заменяет текстовые смайлы (эмотиконы) на унифицированные метки.
    
    Args:
        text (str): Исходный текст.
    
    Returns:
        str: Текст с заменёнными смайлами.
    
    Examples:
        >>> normalize_text_smileys("Привет :) Как дела :(")
        'Привет :smiling_face: Как дела :crying_face:'
    """
    for pattern, replacement in TEXT_SMILEYS.items():
        text = re.sub(pattern, replacement, text)
    return text

def clean_text(text: str) -> str:
    """
    Основная функция предобработки текста.
    
    Выполняет последовательно:
        1. Приведение к нижнему регистру.
        2. Замену эмодзи на текстовые дескрипторы (через normalize_emojis).
        3. Замену текстовых смайлов (через normalize_text_smileys).
        4. Удаление URL и упоминаний пользователей.
        5. Удаление всех символов, кроме букв (русских/английских), цифр, пробелов,
           двоеточий и подчёркиваний (последние нужны для сохранения меток эмодзи).
        6. Схлопывание множественных пробелов и обрезку по краям.
    
    Args:
        text (str): Исходный текст сообщения (пост или комментарий).
    
    Returns:
        str: Очищенный текст, готовый к дальнейшей обработке (лемматизации или токенизации).
    
    Examples:
        >>> clean_text("Привеееет! Это ужас(( 😡 #жалоба https://vk.com")
        'привеееет это ужас :pouting_face: жалоба'
    """
    if not text or not text.strip():
        return ""

    # 1. Нижний регистр
    text = text.lower()

    # 2. Нормализация эмодзи (пиктограмм)
    text = normalize_emojis(text)

    # 3. Нормализация текстовых смайлов
    text = normalize_text_smileys(text)

    # 4. Удаление URL и упоминаний
    text = re.sub(r"https?://\S+|www\.\S+", "", text)
    text = re.sub(r"@\w+", "", text)

    # 5. Удаление спецсимволов (разрешены буквы, цифры, пробелы, двоеточие, подчёркивание)
    text = re.sub(CLEAN_REGEX, " ", text)

    # 6. Схлопывание множественных пробелов
    text = re.sub(r"\s+", " ", text).strip()

    return text

def lemmatize(text: str) -> str:
    """
    Лемматизация текста с помощью Mystem.
    
    Применяется только для классических моделей (TF‑IDF + логистическая регрессия).
    Для трансформеров (RuBERT) лемматизация не используется, так как модели обучаются на исходных словоформах.
    
    Args:
        text (str): Текст после clean_text().
    
    Returns:
        str: Лемматизированный текст.
    
    Note:
        Если библиотека pymystem3 не установлена, возвращается исходный текст.
    """
    if not HAS_MYSTEM or not text:
        return text
    lemmas = mystem.lemmatize(text)
    return "".join(lemmas).strip()