# hierarchy_graph.py
# -*- coding: utf-8 -*-
"""
Модуль для построения иерархического графа "сообщество → теги → посты → комментарии".
Цвет узлов зависит от тональности (для постов и комментариев).
"""

import os
import networkx as nx
import plotly.graph_objects as go
from collections import defaultdict
from typing import List, Set
import pandas as pd

from config import OUTPUT_DIR
from utils import logger, ensure_directory_exists


def build_hierarchy_graph(df: pd.DataFrame, groups_set: List[str]) -> nx.DiGraph:
    """
    Строит иерархический граф:
        - Узлы: сообщества, теги, посты, комментарии
        - Рёбра: сообщество -> пост, тег -> пост, пост -> комментарий

    Args:
        df: DataFrame с колонками (group, post_id, original_text, rubert_sentiment,
            hashtags, type, parent_post_id, author_id)
        groups_set: список имён групп (для создания узлов сообществ)

    Returns:
        networkx.DiGraph (направленный граф)
    """
    required_cols = ["group", "post_id", "original_text", "rubert_sentiment",
                     "hashtags", "type", "parent_post_id", "author_id"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        logger.error(f"В DataFrame отсутствуют обязательные столбцы: {missing}")
        raise ValueError(f"Отсутствуют столбцы: {missing}")

    G = nx.DiGraph()

    # Добавляем узлы сообществ
    for group in groups_set:
        G.add_node(f"group_{group}", type="group", name=group)

    post_nodes = {}

    # Добавляем все посты
    posts_df = df[df['type'] == 'post']
    for _, row in posts_df.iterrows():
        group = row["group"]
        post_id = row["post_id"]
        post_node = f"post_{group}_{post_id}"

        full_text = row["original_text"]
        short_text = (full_text[:100] + "...") if len(full_text) > 100 else full_text

        G.add_node(post_node,
                   type="post",
                   group=group,
                   post_id=post_id,
                   full_text=full_text,
                   short_text=short_text,
                   sentiment=row["rubert_sentiment"])
        post_nodes[(group, post_id)] = post_node
        G.add_edge(f"group_{group}", post_node, relation="publishes")

        # Теги
        hashtags_str = row["hashtags"]
        if isinstance(hashtags_str, str) and hashtags_str:
            for tag in hashtags_str.split(","):
                tag = tag.strip()
                if tag:
                    tag_node = f"tag_{tag}"
                    G.add_node(tag_node, type="tag", name=tag)
                    G.add_edge(tag_node, post_node, relation="tagged")

    # Добавляем комментарии
    comments_df = df[df['type'] == 'comment']
    added = 0
    skipped = 0
    for _, row in comments_df.iterrows():
        group = row["group"]
        parent_id = row["parent_post_id"]
        comment_id = row["post_id"]
        author_id = row["author_id"]
        parent_key = (group, parent_id)
        if parent_key in post_nodes:
            parent_node = post_nodes[parent_key]
            comment_node = f"comment_{group}_{comment_id}"

            full_text = row["original_text"]
            short_text = (full_text[:100] + "...") if len(full_text) > 100 else full_text

            G.add_node(comment_node,
                       type="comment",
                       author=author_id,
                       comment_id=comment_id,
                       full_text=full_text,
                       short_text=short_text,
                       sentiment=row["rubert_sentiment"])
            G.add_edge(parent_node, comment_node, relation="has_comment")
            added += 1
        else:
            skipped += 1

    logger.info(f"Иерархический граф: {G.number_of_nodes()} узлов, {G.number_of_edges()} рёбер")
    logger.info(f"Добавлено комментариев: {added}, пропущено (нет поста): {skipped}")
    return G


def save_hierarchy_graph(
    G: nx.DiGraph,
    output_file: str = "hierarchy_graph.html",
    width: int = 1400,
    height: int = 900
) -> None:
    """
    Сохраняет иерархический граф в интерактивный HTML-файл.
    Цвет постов и комментариев зависит от тональности:
        - пост: positive → зелёный, negative → красный, neutral → серый
        - комментарий: positive → салатовый, negative → розовый, neutral → светло-серый
    """
    if G.number_of_nodes() == 0:
        logger.warning("Иерархический граф пуст, сохранение отменено")
        return

    try:
        level_y = {
            "group": 0.85,
            "tag": 0.85,
            "post": 0.50,
            "comment": 0.15
        }

        nodes_by_type = defaultdict(list)
        for node, attrs in G.nodes(data=True):
            node_type = attrs.get('type', 'post')
            nodes_by_type[node_type].append(node)

        pos = {}
        for node_type, nodes in nodes_by_type.items():
            base_y = level_y.get(node_type, 0.5)
            n = len(nodes)
            if n == 0:
                continue
            for i, node in enumerate(nodes):
                x = (i + 0.5) / n
                y_offset = (hash(node) % 100) / 1000.0 - 0.05
                y = base_y + y_offset
                y = max(0.05, min(0.95, y))
                pos[node] = (x * width, y * height)

        # Рёбра
        edge_x, edge_y = [], []
        for u, v in G.edges():
            if u not in pos or v not in pos:
                continue
            x0, y0 = pos[u]
            x1, y1 = pos[v]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])

        edge_trace = go.Scatter(
            x=edge_x, y=edge_y,
            line=dict(width=0.8, color='#aaa'),
            hoverinfo='none',
            mode='lines'
        )

        # Узлы с динамическим цветом в зависимости от тональности
        node_x, node_y, node_text, node_sizes, node_colors = [], [], [], [], []

        # Функция определения цвета для узла
        def get_node_color(attrs):
            node_type = attrs.get('type')
            sentiment = attrs.get('sentiment', 'neutral')
            if node_type == 'post':
                if sentiment == 'positive':
                    return '#2ecc71'      # зелёный
                elif sentiment == 'negative':
                    return '#e74c3c'      # красный
                else:
                    return '#95a5a6'      # серый
            elif node_type == 'comment':
                if sentiment == 'positive':
                    return '#a3e4a3'      # салатовый
                elif sentiment == 'negative':
                    return '#f5b7b1'      # розовый
                else:
                    return '#d5dbdb'      # светло-серый
            elif node_type == 'group':
                return '#5dade2'           # голубой
            elif node_type == 'tag':
                return '#58d68d'           # зелёный (теги)
            else:
                return '#bdc3c7'           # нейтральный

        for node in G.nodes():
            if node not in pos:
                continue
            x, y = pos[node]
            node_x.append(x)
            node_y.append(y)
            attrs = G.nodes[node]

            # Текст подсказки
            text = f"<b>Тип: {attrs.get('type')}</b><br>"
            if attrs.get('type') == 'group':
                text += f"Группа: {attrs.get('name')}"
            elif attrs.get('type') == 'tag':
                text += f"Тег: {attrs.get('name')}"
            elif attrs.get('type') == 'post':
                text += (f"Пост {attrs.get('post_id')}<br>"
                         f"Группа: {attrs.get('group')}<br>"
                         f"Тональность: {attrs.get('sentiment')}<br>"
                         f"Текст: {attrs.get('short_text')}")
            elif attrs.get('type') == 'comment':
                text += (f"Комментарий {attrs.get('comment_id')}<br>"
                         f"Автор: {attrs.get('author')}<br>"
                         f"Тональность: {attrs.get('sentiment')}<br>"
                         f"Текст: {attrs.get('short_text')}")
            node_text.append(text)

            size = 15 if attrs.get('type') == 'post' else (12 if attrs.get('type') == 'comment' else 8)
            node_sizes.append(size)
            node_colors.append(get_node_color(attrs))

        node_trace = go.Scatter(
            x=node_x, y=node_y,
            mode='markers',
            hoverinfo='text',
            text=node_text,
            marker=dict(
                size=node_sizes,
                color=node_colors,
                line=dict(width=1, color='black')
            )
        )

        # Легенда с новыми цветами
        legend_annotations = [
            dict(x=0.02, y=0.98, xref="paper", yref="paper",
                 text="<b>Легенда</b>", showarrow=False, font=dict(size=14)),
            dict(x=0.02, y=0.94, xref="paper", yref="paper",
                 text="🔵 Группа", showarrow=False, font=dict(size=12)),
            dict(x=0.02, y=0.90, xref="paper", yref="paper",
                 text="🟢 Тег", showarrow=False, font=dict(size=12)),
            dict(x=0.02, y=0.86, xref="paper", yref="paper",
                 text="🟢 Пост (позитивный)", showarrow=False, font=dict(size=12), font_color="#2ecc71"),
            dict(x=0.02, y=0.82, xref="paper", yref="paper",
                 text="🔴 Пост (негативный)", showarrow=False, font=dict(size=12), font_color="#e74c3c"),
            dict(x=0.02, y=0.78, xref="paper", yref="paper",
                 text="⚪ Пост (нейтральный)", showarrow=False, font=dict(size=12), font_color="#95a5a6"),
            dict(x=0.02, y=0.74, xref="paper", yref="paper",
                 text="🟢 Комментарий (позитивный)", showarrow=False, font=dict(size=12), font_color="#a3e4a3"),
            dict(x=0.02, y=0.70, xref="paper", yref="paper",
                 text="🔴 Комментарий (негативный)", showarrow=False, font=dict(size=12), font_color="#f5b7b1"),
            dict(x=0.02, y=0.66, xref="paper", yref="paper",
                 text="⚪ Комментарий (нейтральный)", showarrow=False, font=dict(size=12), font_color="#d5dbdb"),
            dict(x=0.02, y=0.62, xref="paper", yref="paper",
                 text="(Наведите на узел для деталей)", showarrow=False,
                 font=dict(size=10, color="gray"))
        ]

        fig = go.Figure(data=[edge_trace, node_trace],
                        layout=go.Layout(
                            title=dict(text="Иерархический граф: теги → посты → комментарии (цвет по тональности)"),
                            showlegend=False,
                            hovermode='closest',
                            width=width,
                            height=height,
                            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                            plot_bgcolor='white',
                            annotations=legend_annotations
                        ))

        full_path = output_file
        if not os.path.isabs(output_file):
            full_path = os.path.join(OUTPUT_DIR, output_file)
        ensure_directory_exists(os.path.dirname(full_path))
        fig.write_html(full_path)
        logger.info(f"Иерархический граф сохранён в {full_path}")
    except Exception as e:
        logger.error(f"Ошибка при сохранении иерархического графа: {e}", exc_info=True)