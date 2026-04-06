# graph_analysis.py
# -*- coding: utf-8 -*-
"""
Модуль для построения и анализа графа сходства текстов.

Функции:
    - build_similarity_graph: строит взвешенный граф на основе косинусной близости TF‑IDF векторов.
    - compute_graph_metrics: вычисляет метрики графа (плотность, центральности, модулярность).
    - interactive_graph_improved: создаёт интерактивный HTML-граф с помощью Plotly.

Зависимости:
    pip install networkx plotly scikit-learn
"""

import os
import networkx as nx
import pandas as pd
import plotly.graph_objects as go
from collections import defaultdict
from typing import List, Dict, Any, Optional
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from config import SIMILARITY_THRESHOLD, TFIDF_MAX_FEATURES, OUTPUT_DIR
from preprocessing import clean_text, lemmatize
from utils import logger, ensure_directory_exists, handle_exceptions

# ----------------------------------------------------------------------
# 1. Построение графа сходства
# ----------------------------------------------------------------------
@handle_exceptions(default_return=nx.Graph(), reraise=False)
def build_similarity_graph(
    texts: List[str],
    metadata: Optional[List[Dict[str, Any]]] = None,
    threshold: float = SIMILARITY_THRESHOLD,
    max_features: int = TFIDF_MAX_FEATURES
) -> nx.Graph:
    """
    Строит неориентированный взвешенный граф сходства текстов.
    Узлы – индексы текстов, рёбра – если косинусная близость > threshold.

    Args:
        texts: список исходных текстов
        metadata: список словарей с атрибутами для каждого узла (group, sentiment, short_text, ...)
        threshold: порог косинусной близости для добавления ребра
        max_features: максимальное число признаков TF-IDF

    Returns:
        networkx.Graph с узлами (индексы) и атрибутами узлов
    """
    if not texts:
        logger.warning("Пустой список текстов, возвращаем пустой граф")
        return nx.Graph()

    # Очистка и лемматизация для TF-IDF
    cleaned = [clean_text(t) for t in texts]
    lemmatized = [lemmatize(t) for t in cleaned]

    # Отфильтровываем пустые тексты
    non_empty_idx = [i for i, t in enumerate(lemmatized) if t.strip()]
    if not non_empty_idx:
        logger.warning("Нет непустых текстов после предобработки")
        return nx.Graph()

    texts_filtered = [lemmatized[i] for i in non_empty_idx]

    # TF-IDF и косинусная близость
    vectorizer = TfidfVectorizer(max_features=max_features, stop_words=None)
    tfidf = vectorizer.fit_transform(texts_filtered)
    sim = cosine_similarity(tfidf)

    G = nx.Graph()
    # Добавляем узлы
    for i, idx in enumerate(non_empty_idx):
        attrs = {
            "original_text": texts[idx],
            "clean_text": cleaned[idx],
            "short_text": (texts[idx][:100] + "...") if len(texts[idx]) > 100 else texts[idx]
        }
        if metadata and idx < len(metadata):
            attrs.update(metadata[idx])
        G.add_node(idx, **attrs)

    # Добавляем рёбра
    for i in range(len(non_empty_idx)):
        for j in range(i + 1, len(non_empty_idx)):
            if sim[i][j] > threshold:
                G.add_edge(non_empty_idx[i], non_empty_idx[j], weight=sim[i][j])

    logger.info(f"Построен граф с {G.number_of_nodes()} узлами и {G.number_of_edges()} рёбрами")
    return G


# ----------------------------------------------------------------------
# 2. Вычисление метрик графа
# ----------------------------------------------------------------------
def compute_graph_metrics(G: nx.Graph) -> Dict[str, Any]:
    """
    Вычисляет основные метрики графа.

    Returns:
        Словарь с метриками: количество узлов, рёбер, плотность, средняя степень,
        центральности (топ-5), межгрупповые рёбра (если есть атрибут 'group'),
        модулярность (если заданы сообщества по группам).
    """
    if G.number_of_nodes() == 0:
        logger.warning("Граф пуст, возвращаем пустые метрики")
        return {}

    metrics = {
        "nodes": G.number_of_nodes(),
        "edges": G.number_of_edges(),
        "density": nx.density(G),
        "avg_degree": sum(dict(G.degree()).values()) / G.number_of_nodes()
    }

    # Центральности (топ-5 для отображения)
    try:
        deg_cent = nx.degree_centrality(G)
        top_deg = dict(sorted(deg_cent.items(), key=lambda x: x[1], reverse=True)[:5])
        metrics["degree_centrality_top5"] = top_deg
    except Exception as e:
        logger.warning(f"Не удалось вычислить degree_centrality: {e}")

    try:
        bet_cent = nx.betweenness_centrality(G, k=min(100, G.number_of_nodes()))
        top_bet = dict(sorted(bet_cent.items(), key=lambda x: x[1], reverse=True)[:5])
        metrics["betweenness_centrality_top5"] = top_bet
    except Exception as e:
        logger.warning(f"Не удалось вычислить betweenness_centrality: {e}")

    # Межгрупповые рёбра (если есть атрибут group)
    if all('group' in G.nodes[n] for n in G.nodes()):
        groups = {n: G.nodes[n]['group'] for n in G.nodes()}
        inter_edges = 0
        for u, v in G.edges():
            if groups[u] != groups[v]:
                inter_edges += 1
        metrics["inter_group_edges_ratio"] = inter_edges / G.number_of_edges() if G.number_of_edges() > 0 else 0

        # Модулярность (сообщества на основе групп)
        groups_by_val = defaultdict(list)
        for n, attrs in G.nodes(data=True):
            grp = attrs.get('group')
            if grp is not None:
                groups_by_val[grp].append(n)
        communities = list(groups_by_val.values())
        if len(communities) > 1:
            try:
                metrics["modularity"] = nx.community.modularity(G, communities)
            except Exception as e:
                logger.warning(f"Не удалось вычислить модулярность: {e}")

    return metrics


# ----------------------------------------------------------------------
# 3. Интерактивная визуализация графа с помощью Plotly
# ----------------------------------------------------------------------
def interactive_graph_improved(
    G: nx.Graph,
    node_color_by: str = 'group',
    output_file: str = 'graph_interactive.html',
    width: int = 1200,
    height: int = 800
) -> None:
    """
    Создаёт интерактивный HTML-граф с раскраской узлов по заданному атрибуту.

    Args:
        G: граф networkx
        node_color_by: имя атрибута узла, по которому раскрашивать (например, 'group', 'sentiment')
        output_file: путь к выходному HTML-файлу
        width, height: размеры графика
    """
    if G.number_of_nodes() == 0:
        logger.warning("Граф пуст, визуализация не выполняется")
        return

    try:
        # Позиционирование узлов (spring layout)
        pos = nx.spring_layout(G, seed=42, k=2, iterations=100)

        # Рёбра
        edge_x = []
        edge_y = []
        for u, v in G.edges():
            x0, y0 = pos[u]
            x1, y1 = pos[v]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])

        edge_trace = go.Scatter(
            x=edge_x, y=edge_y,
            line=dict(width=0.8, color='#888'),
            hoverinfo='none',
            mode='lines'
        )

        # Узлы
        node_x = []
        node_y = []
        node_text = []
        node_color_vals = []
        node_sizes = []
        degrees = dict(G.degree())
        max_deg = max(degrees.values()) if degrees else 1

        # Определяем цветовую шкалу
        color_values = [G.nodes[n].get(node_color_by, '') for n in G.nodes()]
        if color_values and isinstance(color_values[0], str):
            # Кодируем категории числами
            codes, uniques = pd.factorize(pd.Series(color_values))
            node_color_vals = codes.tolist()
            colorbar_title = node_color_by
        else:
            node_color_vals = color_values
            colorbar_title = node_color_by

        for node in G.nodes():
            x, y = pos[node]
            node_x.append(x)
            node_y.append(y)

            # Текст всплывающей подсказки
            attrs = G.nodes[node]
            text = f"ID: {node}<br>"
            for k, v in attrs.items():
                if k in ['original_text', 'clean_text', 'short_text']:
                    continue
                text += f"{k}: {v}<br>"
            if 'short_text' in attrs:
                text += f"Текст: {attrs['short_text']}<br>"
            node_text.append(text)

            size = 8 + 20 * (degrees[node] / max_deg)
            node_sizes.append(size)

        node_trace = go.Scatter(
            x=node_x, y=node_y,
            mode='markers',
            hoverinfo='text',
            text=node_text,
            marker=dict(
                size=node_sizes,
                color=node_color_vals,
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(title=colorbar_title),
                line=dict(width=1, color='DarkSlateGrey')
            )
        )

        fig = go.Figure(data=[edge_trace, node_trace],
                        layout=go.Layout(
                            title=dict(text=f"Граф сходства постов (цвет = {node_color_by})"),
                            showlegend=False,
                            hovermode='closest',
                            width=width,
                            height=height,
                            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                            plot_bgcolor='white'
                        ))

        # Сохраняем в выходную директорию, если передан относительный путь
        full_path = output_file
        if not os.path.isabs(output_file):
            full_path = os.path.join(OUTPUT_DIR, output_file)
        ensure_directory_exists(os.path.dirname(full_path))
        fig.write_html(full_path)
        logger.info(f"Интерактивный граф сохранён в {full_path}")
    except Exception as e:
        logger.error(f"Ошибка при создании интерактивного графа: {e}", exc_info=True)