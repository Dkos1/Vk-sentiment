# main.py
# -*- coding: utf-8 -*-
"""
Главный скрипт информационной системы мониторинга тональности сообщений ВКонтакте.

Выполняет полный пайплайн:
    1. Сбор данных через VK API (или загрузка из кэша)
    2. Предобработку текстов (очистка, нормализация эмодзи, лемматизация)
    3. Анализ тональности двумя моделями (Baseline и RuBERT)
    4. Построение графов сходства и иерархического графа
    5. Извлечение ключевых причин (n-граммы) и (опционально) LLM-анализ с контекстом поста
    6. Сохранение результатов и визуализация

Использование:
    python main.py                     # обычный запуск
    python main.py --validate          # запуск валидации моделей (без сбора данных)
    python main.py --no-graphs         # без построения графов (экономия ресурсов)
"""

import os
import sys
import argparse
import pickle
import glob
import pandas as pd
from typing import List, Tuple

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config
from config import (
    GROUPS, POSTS_PER_GROUP, COMMENTS_PER_POST, POSTS_OFFSET,
    USE_LLM, LLM_MAX_TEXTS, OUTPUT_DIR, LOGS_DIR, MODELS_DIR,
    ensure_directories
)

from vk_api import collect_all_posts
from preprocessing import clean_text, lemmatize
from sentiment_models import BaselineSentimentModel, RuBertSentimentModel
from graph_analysis import (
    build_similarity_graph, compute_graph_metrics,
    interactive_graph_improved
)
from hierarchy_graph import build_hierarchy_graph, save_hierarchy_graph
from extract_reasons import generate_reasons_report
from utils import logger, timeit, save_results, ensure_directory_exists

if USE_LLM:
    from saiga_analyzer import SaigaTopicExtractor, generate_llm_report


def run_validation():
    from validation import run_full_validation
    run_full_validation()


def load_or_collect_data(offset: int = 0) -> List[Tuple]:
    """Загружает данные из кэша или собирает через VK API."""
    cache_file = f"raw_data_offset_{offset}.pkl" if offset != 0 else "raw_data.pkl"
    cache_path = os.path.join(config.BASE_DIR, cache_file)
    if os.path.exists(cache_path):
        logger.info(f"Загрузка данных из кэша: {cache_path}")
        with open(cache_path, "rb") as f:
            return pickle.load(f)
    else:
        logger.info(f"Сбор данных из VK API (offset={offset})...")
        all_items = collect_all_posts(GROUPS, POSTS_PER_GROUP, COMMENTS_PER_POST, offset=offset)
        with open(cache_path, "wb") as f:
            pickle.dump(all_items, f)
        logger.info(f"Собрано {len(all_items)} элементов, кэш сохранён в {cache_path}")
        return all_items


@timeit
def main() -> None:
    parser = argparse.ArgumentParser(description="Анализ тональности сообщений ВКонтакте")
    parser.add_argument("--validate", action="store_true", help="Запустить валидацию моделей")
    parser.add_argument("--no-graphs", action="store_true", help="Не строить графы")
    args = parser.parse_args()

    if args.validate:
        logger.info("Переключение в режим валидации")
        run_validation()
        return

    ensure_directories()
    ensure_directory_exists(OUTPUT_DIR)
    ensure_directory_exists(LOGS_DIR)
    ensure_directory_exists(MODELS_DIR)

    # 1. Сбор данных
    logger.info("=== НАЧАЛО СБОРА ДАННЫХ ===")
    all_items = load_or_collect_data(offset=POSTS_OFFSET)
    if not all_items:
        logger.error("Не удалось собрать ни одного текста. Завершение.")
        return
    logger.info(f"Всего собрано элементов: {len(all_items)}")

    groups = [item[0] for item in all_items]
    texts = [item[1] for item in all_items]
    item_ids = [item[2] for item in all_items]
    owners = [item[3] for item in all_items]
    types = [item[4] for item in all_items]
    parent_ids = [item[5] for item in all_items]
    author_ids = [item[6] for item in all_items]
    hashtags = [item[7] for item in all_items]

    # 2. Предобработка
    logger.info("=== ПРЕДОБРАБОТКА ТЕКСТОВ ===")
    cleaned_texts = [clean_text(t) for t in texts]
    lemmatized_texts = [lemmatize(t) for t in cleaned_texts]

    # 3. Анализ тональности: Baseline
    logger.info("=== АНАЛИЗ ТОНАЛЬНОСТИ (BASELINE) ===")
    baseline = BaselineSentimentModel()
    baseline_preds = [0] * len(texts)
    baseline_probas = [0.5] * len(texts)
    try:
        baseline.load()
        baseline_preds, baseline_probas = baseline.predict(texts)
        logger.info("Baseline модель загружена и выполнила предсказания")
    except FileNotFoundError:
        logger.warning("Baseline модель не найдена, пробую обучить из labeled_data.csv...")
        try:
            baseline.train_from_labeled_csv()
            baseline_preds, baseline_probas = baseline.predict(texts)
            baseline.save()
            logger.info("Baseline модель обучена и сохранена")
        except (FileNotFoundError, ValueError) as e:
            logger.error(f"Не удалось обучить baseline: {e}. Использую заглушки.")
    except Exception as e:
        logger.error(f"Ошибка baseline: {e}", exc_info=True)

    # 4. Анализ тональности: RuBERT
    logger.info("=== АНАЛИЗ ТОНАЛЬНОСТИ (RuBERT) ===")
    rubert_results = []
    try:
        rubert = RuBertSentimentModel(model_path=config.FINE_TUNED_RUBERT_PATH)
        rubert_results = rubert.predict_batch(texts)
        logger.info("RuBERT модель загружена и выполнила предсказания")
    except Exception as e:
        logger.error(f"Ошибка RuBERT: {e}", exc_info=True)
        rubert_results = [("neutral", 0.5) for _ in texts]

    # 5. Формирование DataFrame
    df = pd.DataFrame({
        "group": groups,
        "type": types,
        "post_id": item_ids,
        "parent_post_id": parent_ids,
        "author_id": author_ids,
        "hashtags": [",".join(h) for h in hashtags],
        "original_text": texts,
        "clean_text": cleaned_texts,
        "lemmatized": lemmatized_texts,
        "baseline_sentiment": ["positive" if p == 2 else "negative" if p == 0 else "neutral" for p in baseline_preds],
        "baseline_proba": baseline_probas,
        "rubert_sentiment": [r[0] for r in rubert_results],
        "rubert_confidence": [r[1] for r in rubert_results]
    })
    df["owner"] = owners
    def make_url(row):
        if row["type"] == "post":
            return f"https://vk.com/wall{row['owner']}_{row['post_id']}"
        else:
            return f"https://vk.com/wall{row['owner']}_{row['parent_post_id']}?reply={row['post_id']}"
    df["url"] = df.apply(make_url, axis=1)
    df.drop(columns=["owner"], inplace=True)

    # --- Добавляем текст родительского поста для комментариев (контекст для LLM) ---
    post_text_map = {}
    for _, row in df[df['type'] == 'post'].iterrows():
        key = (row['group'], row['post_id'])
        post_text_map[key] = row['original_text']

    def get_parent_text(row):
        if row['type'] == 'comment':
            key = (row['group'], row['parent_post_id'])
            return post_text_map.get(key, '')
        return ''

    df['parent_post_text'] = df.apply(get_parent_text, axis=1)

    csv_path = os.path.join(OUTPUT_DIR, "vk_sentiment_results.csv")
    save_results(df, csv_path)
    logger.info(f"Результаты сохранены в {csv_path}")

    # 6. Граф сходства (если не отключён)
    if not args.no_graphs:
        logger.info("=== ПОСТРОЕНИЕ ГРАФА СХОДСТВА ===")
        metadata = []
        for i in range(len(all_items)):
            short_text = texts[i][:50].replace('\n', ' ').strip()
            metadata.append({
                "group": groups[i],
                "sentiment": df.iloc[i]["rubert_sentiment"],
                "baseline_sentiment": df.iloc[i]["baseline_sentiment"],
                "short_text": short_text,
                "post_id": item_ids[i],
                "type": types[i]
            })
        G = build_similarity_graph(texts, metadata=metadata)
        logger.info(f"Граф сходства: {G.number_of_nodes()} вершин, {G.number_of_edges()} рёбер")
        metrics = compute_graph_metrics(G)
        logger.info("Метрики графа сходства:")
        for k, v in metrics.items():
            if isinstance(v, dict):
                top5 = dict(sorted(v.items(), key=lambda x: x[1], reverse=True)[:5])
                logger.info(f"  {k}: (топ-5) {top5}")
            else:
                logger.info(f"  {k}: {v}")
        suffix = f"_offset{POSTS_OFFSET}" if POSTS_OFFSET != 0 else ""
        interactive_graph_improved(G, node_color_by="group", output_file=os.path.join(OUTPUT_DIR, f"graph_groups{suffix}.html"))
        interactive_graph_improved(G, node_color_by="sentiment", output_file=os.path.join(OUTPUT_DIR, f"graph_sentiment_rubert{suffix}.html"))
        interactive_graph_improved(G, node_color_by="baseline_sentiment", output_file=os.path.join(OUTPUT_DIR, f"graph_sentiment_baseline{suffix}.html"))

        # 7. Иерархический граф
        logger.info("=== ПОСТРОЕНИЕ ИЕРАРХИЧЕСКОГО ГРАФА ===")
        H = build_hierarchy_graph(df, groups_set=GROUPS)
        save_hierarchy_graph(H, output_file=os.path.join(OUTPUT_DIR, f"hierarchy_graph{suffix}.html"))

    # 8. Извлечение ключевых причин (n-граммы)
    logger.info("=== ИЗВЛЕЧЕНИЕ ПРИЧИН НЕДОВОЛЬСТВА (N-GRAMM) ===")
    reasons_report_path = os.path.join(OUTPUT_DIR, "reasons_report.html")
    generate_reasons_report(df, output_file=reasons_report_path)

    # 9. LLM-анализ с контекстом поста
    if USE_LLM:
        logger.info("=== LLM-АНАЛИЗ КОММЕНТАРИЕВ С КОНТЕКСТОМ ПОСТА (SAIGA) ===")
        comments_df = df[df['type'] == 'comment'].copy()
        comments_with_context = comments_df[
            comments_df['parent_post_text'].notna() & (comments_df['parent_post_text'] != '')
        ]
        if comments_with_context.empty:
            logger.warning("Нет комментариев с контекстом поста для LLM-анализа.")
        else:
            # Формируем расширенный список: (post_text, comment_text, post_url, post_short_text)
            context_pairs = []
            for _, row in comments_with_context.iterrows():
                post_text = row['parent_post_text']
                comment_text = row['original_text']
                # Формируем URL поста
                post_url = f"https://vk.com/wall{row['owner']}_{row['parent_post_id']}" if 'owner' in row else "#"
                short_post = (post_text[:100] + "...") if len(post_text) > 100 else post_text
                context_pairs.append((post_text, comment_text, post_url, short_post))
            if len(context_pairs) > LLM_MAX_TEXTS:
                context_pairs = context_pairs[:LLM_MAX_TEXTS]
                logger.info(f"Ограничиваемся первыми {LLM_MAX_TEXTS} комментариями с контекстом.")
            try:
                extractor = SaigaTopicExtractor(
                    model_name=config.LLM_MODEL_NAME,
                    timeout=config.LLM_TIMEOUT
                )
                themes, example_metadata = extractor.extract_themes_with_metadata(context_pairs)
                summary = extractor.generate_summary_and_recommendations(themes)
                llm_report_path = os.path.join(OUTPUT_DIR, "saiga_reasons_report.html")
                generate_llm_report(themes, summary, example_metadata, output_file=llm_report_path)
            except Exception as e:
                logger.error(f"Ошибка при LLM-анализе: {e}", exc_info=True)


if __name__ == "__main__":
    main()