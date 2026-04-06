# extract_reasons.py
# -*- coding: utf-8 -*-
"""
Модуль для извлечения ключевых причин недовольства и положительных моментов
на основе анализа n-грамм в комментариях.

Функции:
    - extract_ngrams: выделяет наиболее частотные осмысленные n-граммы (2-3 слова)
    - generate_reasons_report: создаёт HTML-отчёт с причинами и примерами

Зависимости:
    pip install scikit-learn pandas
"""

import os
import re
import pandas as pd
from collections import Counter, defaultdict
from sklearn.feature_extraction.text import CountVectorizer
from typing import List, Tuple

from config import OUTPUT_DIR
from preprocessing import clean_text, lemmatize
from utils import logger, ensure_directory_exists

# ----------------------------------------------------------------------
# Стоп-слова и фильтры
# ----------------------------------------------------------------------
RUSSIAN_STOP_WORDS = [
    'и', 'в', 'во', 'не', 'что', 'на', 'я', 'с', 'со', 'как', 'а', 'то', 'все', 'она', 'так', 'его', 'но',
    'да', 'ты', 'к', 'у', 'же', 'вы', 'за', 'бы', 'по', 'только', 'ее', 'мне', 'было', 'вот', 'от', 'меня',
    'еще', 'нет', 'о', 'из', 'ему', 'теперь', 'когда', 'даже', 'ну', 'вдруг', 'ли', 'если', 'уже', 'или',
    'ни', 'быть', 'был', 'него', 'до', 'вас', 'нибудь', 'опять', 'уж', 'вам', 'ведь', 'там', 'потом', 'себя',
    'ничего', 'ей', 'может', 'они', 'тут', 'где', 'есть', 'надо', 'ней', 'для', 'мы', 'тебя', 'их', 'чем',
    'была', 'сам', 'чтоб', 'без', 'будто', 'чего', 'раз', 'тоже', 'себе', 'под', 'будет', 'ж', 'тогда', 'кто',
    'этот', 'того', 'потому', 'этого', 'какой', 'совсем', 'ним', 'здесь', 'этом', 'один', 'почти', 'мой',
    'тем', 'чтобы', 'нее', 'сейчас', 'были', 'куда', 'зачем', 'всех', 'никогда', 'можно', 'при', 'наконец',
    'два', 'об', 'другой', 'хоть', 'после', 'над', 'больше', 'тот', 'через', 'эти', 'нас', 'про', 'всего',
    'них', 'какая', 'много', 'разве', 'три', 'эту', 'моя', 'впрочем', 'хорошо', 'свою', 'этой', 'перед',
    'иногда', 'лучше', 'чуть', 'том', 'нельзя', 'такой', 'им', 'более', 'всегда', 'конечно', 'всю', 'между'
]

MEANINGLESS_NGRAMS = {
    'не не', 'ни ни', 'не очень', 'очень не', 'не так', 'так не', 'не надо', 'не буду', 'не хочу',
    'все таки', 'все же', 'все равно', 'потому что', 'так как', 'на самом', 'самом деле', 'деле не',
    'том что', 'это не', 'этот не', 'какой то', 'то есть', 'так себе', 'более менее', 'менее более',
    'чуть чуть', 'чуть ли', 'в конце', 'конце концов', 'вообще не', 'в принципе', 'по сути',
    'наверное', 'конечно же', 'может быть', 'наверняка', 'в общем', 'скорее всего', 'тем более'
}


def is_meaningful_ngram(ngram: str) -> bool:
    """Проверяет, является ли n-грамма осмысленной (не шумом)."""
    words = ngram.split()
    if len(words) < 2:
        return False
    if any(len(w) < 3 for w in words):
        return False
    if ngram in MEANINGLESS_NGRAMS:
        return False
    stop_count = sum(1 for w in words if w in RUSSIAN_STOP_WORDS)
    if stop_count > 1:
        return False
    return True


def extract_ngrams(texts: List[str], ngram_range: Tuple[int, int] = (2, 3), top_n: int = 30) -> Counter:
    """
    Извлекает наиболее частотные осмысленные n-граммы из списка текстов.

    Args:
        texts: список исходных текстов
        ngram_range: диапазон длины n-грамм (min_n, max_n)
        top_n: количество возвращаемых n-грамм

    Returns:
        Counter с n-граммами и их частотами
    """
    if not texts:
        return Counter()

    # Предобработка
    processed = [lemmatize(clean_text(t)) for t in texts]
    processed = [t for t in processed if len(t) > 10]

    if not processed:
        return Counter()

    vectorizer = CountVectorizer(
        ngram_range=ngram_range,
        stop_words=RUSSIAN_STOP_WORDS,
        token_pattern=r'(?u)\b\w+\b',
        min_df=1
    )
    X = vectorizer.fit_transform(processed)
    sums = X.sum(axis=0).A1
    features = vectorizer.get_feature_names_out()
    freq = {features[i]: sums[i] for i in range(len(features))}
    filtered = {ng: f for ng, f in freq.items() if is_meaningful_ngram(ng)}
    sorted_items = sorted(filtered.items(), key=lambda x: x[1], reverse=True)
    return Counter(dict(sorted_items[:top_n]))


def remove_substring_duplicates(ngram_counter: Counter) -> Counter:
    """Удаляет n-граммы, которые являются подстроками более длинных фраз."""
    items = list(ngram_counter.items())
    items.sort(key=lambda x: len(x[0].split()), reverse=True)
    keep = {}
    for phrase, freq in items:
        is_substring = False
        for kept in keep:
            if phrase in kept:
                is_substring = True
                break
        if not is_substring:
            keep[phrase] = freq
    return Counter(keep)


def get_examples(texts: List[str], phrase: str, n_examples: int = 3) -> List[str]:
    """Возвращает до n_examples примеров текстов, содержащих указанную фразу."""
    examples = []
    phrase_lower = phrase.lower()
    for t in texts:
        if phrase_lower in lemmatize(clean_text(t)).lower():
            short = t[:150].replace('\n', ' ') + ('...' if len(t) > 150 else '')
            examples.append(short)
        if len(examples) >= n_examples:
            break
    return examples


def generate_reasons_report(df: pd.DataFrame, output_file: str = "reasons_report.html") -> None:
    """
    Генерирует HTML-отчёт с ключевыми причинами недовольства и положительными моментами.

    Args:
        df: DataFrame с колонками 'type', 'rubert_sentiment', 'original_text'
        output_file: путь к выходному HTML-файлу
    """
    comments_df = df[df['type'] == 'comment'].copy()
    if comments_df.empty:
        logger.warning("Нет комментариев для анализа причин")
        return

    neg_texts = comments_df[comments_df['rubert_sentiment'] == 'negative']['original_text'].tolist()
    pos_texts = comments_df[comments_df['rubert_sentiment'] == 'positive']['original_text'].tolist()

    logger.info(f"Негативных комментариев: {len(neg_texts)}, позитивных: {len(pos_texts)}")

    # Извлечение n-грамм
    neg_ngrams = extract_ngrams(neg_texts, top_n=30)
    pos_ngrams = extract_ngrams(pos_texts, top_n=30)

    neg_ngrams = remove_substring_duplicates(neg_ngrams)
    pos_ngrams = remove_substring_duplicates(pos_ngrams)

    # Сбор примеров
    neg_items = [(phrase, freq, get_examples(neg_texts, phrase)) for phrase, freq in neg_ngrams.most_common(20)]
    pos_items = [(phrase, freq, get_examples(pos_texts, phrase)) for phrase, freq in pos_ngrams.most_common(20)]

    # Генерация HTML
    html = """
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8">
        <title>Анализ причин недовольства и положительных моментов</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 20px; background: #f5f5f5; }
            h1 { color: #333; }
            h2 { color: #2c3e50; margin-top: 30px; }
            table { border-collapse: collapse; width: 100%; background: white; box-shadow: 0 0 10px rgba(0,0,0,0.1); }
            th, td { border: 1px solid #ddd; padding: 12px; text-align: left; vertical-align: top; }
            th { background-color: #2c3e50; color: white; }
            tr:nth-child(even) { background-color: #f9f9f9; }
            .bad { border-left: 5px solid #e74c3c; padding-left: 15px; }
            .good { border-left: 5px solid #2ecc71; padding-left: 15px; }
            .example { font-size: 0.9em; color: #555; }
            .footer { margin-top: 30px; font-size: 0.8em; color: #777; text-align: center; }
        </style>
    </head>
    <body>
        <h1>📊 Анализ обратной связи</h1>
        <p>Выделены ключевые причины недовольства и положительные моменты на основе комментариев.</p>
    """

    # Негативные причины
    html += '<div class="bad">'
    html += '<h2>🔴 Основные причины недовольства</h2>'
    if neg_items:
        html += '<table><th>Причина / проблема</th><th>Частота упоминаний</th><th>Примеры комментариев</th></tr>'
        for phrase, freq, examples in neg_items:
            examples_html = '<br>'.join([f'“{ex}”' for ex in examples]) if examples else '—'
            html += f'<tr><td>{phrase}</td><td>{freq}</td><td class="example">{examples_html}</td></tr>'
        html += '</table>'
    else:
        html += '<p>Недостаточно негативных комментариев для анализа.</p>'
    html += '</div>'

    # Позитивные моменты
    html += '<div class="good">'
    html += '<h2>🟢 Положительные моменты</h2>'
    if pos_items:
        html += '<table><th>Положительный момент</th><th>Частота упоминаний</th><th>Примеры комментариев</th></tr>'
        for phrase, freq, examples in pos_items:
            examples_html = '<br>'.join([f'“{ex}”' for ex in examples]) if examples else '—'
            html += f'<tr><td>{phrase}</td><td>{freq}</td><td class="example">{examples_html}</td></tr>'
        html += '</table>'
    else:
        html += '<p>Недостаточно позитивных комментариев для анализа.</p>'
    html += '</div>'

    html += """
        <div class="footer">
            Отчёт сгенерирован автоматически на основе анализа тональности (RuBERT).<br>
            Проблемы сгруппированы по смыслу, отфильтрован шум, удалены дубликаты-подстроки.
        </div>
    </body>
    </html>
    """

    full_path = output_file
    if not os.path.isabs(output_file):
        full_path = os.path.join(OUTPUT_DIR, output_file)
    ensure_directory_exists(os.path.dirname(full_path))
    with open(full_path, 'w', encoding='utf-8') as f:
        f.write(html)
    logger.info(f"Отчёт о причинах сохранён в {full_path}")