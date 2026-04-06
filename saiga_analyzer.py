# saiga_analyzer.py
# -*- coding: utf-8 -*-
"""
Модуль для анализа тем и причин в комментариях с помощью локальной LLM (Saiga через Ollama).
Учитывает контекст исходного поста. Возвращает темы с метаданными (URL поста, текст поста).
"""

import os
import json
import re
import time
from collections import defaultdict
from typing import List, Dict, Tuple, Optional, Any

import ollama

from config import LLM_MODEL_NAME, LLM_TIMEOUT, OUTPUT_DIR
from utils import logger, ensure_directory_exists


class SaigaTopicExtractor:
    """
    Класс для выделения тем/причин из комментариев с учётом контекста поста.
    Результаты кэшируются в JSON-файл для ускорения повторных запусков.
    """

    def __init__(
        self,
        model_name: str = LLM_MODEL_NAME,
        timeout: int = LLM_TIMEOUT,
        cache_file: Optional[str] = None
    ):
        self.model_name = model_name
        self.timeout = timeout
        if cache_file is None:
            cache_file = os.path.join(OUTPUT_DIR, "saiga_cache.json")
        self.cache_file = cache_file
        self.cache = self._load_cache()
        if "summary" not in self.cache:
            self.cache["summary"] = None
        self._save_cache()

    def _load_cache(self) -> dict:
        if self.cache_file and os.path.exists(self.cache_file):
            try:
                with open(self.cache_file, "r", encoding="utf-8") as f:
                    return json.load(f)
            except (json.JSONDecodeError, IOError) as e:
                logger.warning(f"Не удалось загрузить кэш: {e}. Создаю новый.")
        return {}

    def _save_cache(self):
        if self.cache_file:
            try:
                cache_dir = os.path.dirname(self.cache_file)
                if cache_dir:
                    ensure_directory_exists(cache_dir)
                with open(self.cache_file, "w", encoding="utf-8") as f:
                    json.dump(self.cache, f, ensure_ascii=False, indent=2)
            except Exception as e:
                logger.error(f"Ошибка сохранения кэша: {e}")

    def _make_cache_key(self, post_text: str, comment_text: str) -> str:
        post_preview = post_text[:200] if len(post_text) > 200 else post_text
        comment_preview = comment_text[:200] if len(comment_text) > 200 else comment_text
        return f"POST: {post_preview}\nCOMMENT: {comment_preview}"

    def _is_empty_or_emoji(self, text: str) -> bool:
        cleaned = re.sub(r'[\s\.,!?;:0-9]', '', text)
        if not cleaned:
            return True
        if all(ord(ch) > 127 for ch in cleaned) and len(cleaned) < 5:
            return True
        return False

    def _extract_json_from_response(self, response_text: str) -> dict:
        json_match = re.search(r"```json\s*\n(.*?)\n```", response_text, re.DOTALL)
        if not json_match:
            json_match = re.search(r"(\{.*\})", response_text, re.DOTALL)
        if json_match:
            json_str = json_match.group(1) if json_match.group(1) else json_match.group(0)
            try:
                return json.loads(json_str)
            except json.JSONDecodeError as e:
                logger.warning(f"Ошибка парсинга JSON: {e}")
        return {"negative_themes": [], "positive_themes": []}

    def _call_model(self, prompt: str, max_retries: int = 2, temperature: float = 0.2) -> dict:
        for attempt in range(max_retries):
            try:
                response = ollama.chat(
                    model=self.model_name,
                    messages=[{"role": "user", "content": prompt}],
                    options={"temperature": temperature, "num_predict": 2048}
                )
                return {"text": response["message"]["content"]}
            except Exception as e:
                logger.warning(f"Ошибка LLM (попытка {attempt+1}): {e}")
                if attempt == max_retries - 1:
                    return {"text": "Не удалось сгенерировать ответ."}
                time.sleep(2 ** attempt)
        return {"text": "Ошибка"}

    def _call_model_for_json(self, prompt: str, max_retries: int = 2) -> dict:
        for attempt in range(max_retries):
            try:
                response = ollama.chat(
                    model=self.model_name,
                    messages=[{"role": "user", "content": prompt}],
                    options={"temperature": 0.2, "num_predict": 512}
                )
                return self._extract_json_from_response(response["message"]["content"])
            except Exception as e:
                logger.warning(f"Ошибка JSON (попытка {attempt+1}): {e}")
                if attempt == max_retries - 1:
                    return {"negative_themes": [], "positive_themes": []}
                time.sleep(2 ** attempt)
        return {"negative_themes": [], "positive_themes": []}

    def extract_themes_with_metadata(
        self,
        context_pairs: List[Tuple[str, str, str, str]]
    ) -> Tuple[Dict[str, List[Tuple[str, int, List[str]]]], Dict[str, List[List[Tuple[str, str, str]]]]]:
        """
        Анализирует комментарии с учётом контекста поста.
        context_pairs: список кортежей (post_text, comment_text, post_url, post_short_text)
        Returns:
            themes: dict с ключами 'negative_themes' и 'positive_themes', где каждый элемент -
                (тема, частота, список_примеров_комментариев)
            example_metadata: dict с ключами 'negative' и 'positive', где каждый элемент -
                список списков метаданных для каждой темы (в том же порядке, что и themes)
        """
        # Фильтруем новые пары (не в кэше)
        new_pairs = []
        for post, comment, url, short_post in context_pairs:
            key = self._make_cache_key(post, comment)
            if key not in self.cache:
                new_pairs.append((post, comment, url, short_post))
        logger.info(f"Всего пар: {len(context_pairs)}, новых: {len(new_pairs)}")
        for idx, (post_text, comment_text, _, _) in enumerate(new_pairs):
            logger.debug(f"Обработка {idx+1}/{len(new_pairs)}...")
            if self._is_empty_or_emoji(comment_text):
                key = self._make_cache_key(post_text, comment_text)
                self.cache[key] = {"negative_themes": [], "positive_themes": []}
                self._save_cache()
                continue
            truncated_post = post_text[:800] + "..." if len(post_text) > 800 else post_text
            truncated_comment = comment_text[:500] + "..." if len(comment_text) > 500 else comment_text
            prompt = f"""
Ты — профессиональный аналитик обратной связи. Проанализируй комментарий пользователя ВКонтакте в контексте исходного поста.

ТЕКСТ ПОСТА (контекст):
{truncated_post}

КОММЕНТАРИЙ ПОЛЬЗОВАТЕЛЯ:
{truncated_comment}

Задача: выделить основные темы и причины, о которых пишет пользователь, и классифицировать их как негативные или позитивные.

Правила:
1. НЕ включай в позитивные темы фразы о смерти, трауре, памяти, соболезнованиях.
2. Если комментарий состоит только из эмодзи или бессмысленных символов, верни пустые списки.
3. Объединяй синонимичные формулировки.
4. Каждая тема должна быть краткой фразой (2-5 слов).

Верни строго JSON:
{{"negative_themes": [...], "positive_themes": [...]}}
"""
            result = self._call_model_for_json(prompt)
            key = self._make_cache_key(post_text, comment_text)
            self.cache[key] = result
            self._save_cache()

        # Собираем все данные с метаинформацией
        all_negative = []   # (theme, comment, url, short_post)
        all_positive = []
        for post, comment, url, short_post in context_pairs:
            key = self._make_cache_key(post, comment)
            data = self.cache.get(key, {"negative_themes": [], "positive_themes": []})
            for theme in data.get("negative_themes", []):
                all_negative.append((theme, comment, url, short_post))
            for theme in data.get("positive_themes", []):
                all_positive.append((theme, comment, url, short_post))

        # Группируем с дедупликацией и сохраняем метаданные
        themes_neg, meta_neg = self._group_themes_with_metadata(all_negative, deduplicate=True)
        themes_pos, meta_pos = self._group_themes_with_metadata(all_positive, deduplicate=True)
        themes = {
            "negative_themes": themes_neg,
            "positive_themes": themes_pos,
        }
        example_metadata = {
            "negative": meta_neg,
            "positive": meta_pos,
        }
        return themes, example_metadata

    def _group_themes_with_metadata(
        self,
        items: List[Tuple[str, str, str, str]],
        deduplicate: bool = True
    ) -> Tuple[List[Tuple[str, int, List[str]]], List[List[Tuple[str, str, str]]]]:
        """
        Группирует темы, подсчитывает частоту и собирает примеры с метаданными.
        items: список (theme, comment_text, post_url, post_short_text)
        Возвращает:
            themes: список (тема, частота, список_комментариев_примеров)
            metadata: список списков метаданных для каждого примера в том же порядке
        """
        if not items:
            return [], []
        # Подсчёт частоты тем
        theme_counts = defaultdict(int)
        for theme, _, _, _ in items:
            theme_counts[theme] += 1
        # Сортируем темы по убыванию частоты
        sorted_themes = sorted(theme_counts.keys(), key=lambda x: (-theme_counts[x], x))

        if not deduplicate:
            groups = defaultdict(list)
            meta_groups = defaultdict(list)
            for theme, comment, url, short_post in items:
                groups[theme].append(comment)
                meta_groups[theme].append((comment, url, short_post))
            themes_res = []
            meta_res = []
            for theme in sorted_themes:
                examples = list(dict.fromkeys(groups.get(theme, [])))[:3]
                # Собрать метаданные для этих примеров (сохраняя порядок)
                full_meta = meta_groups.get(theme, [])
                unique_meta = []
                seen = set()
                for cmt, url_, sp in full_meta:
                    if cmt not in seen:
                        seen.add(cmt)
                        unique_meta.append((cmt, url_, sp))
                themes_res.append((theme, theme_counts[theme], examples))
                meta_res.append(unique_meta[:3])
            return themes_res, meta_res

        # Дедупликация: для каждого текста комментария оставляем тему с наивысшим приоритетом
        theme_priority = {theme: idx for idx, theme in enumerate(sorted_themes)}
        comment_to_theme = {}
        comment_to_metadata = {}
        for theme, comment, url, short_post in items:
            if comment not in comment_to_theme:
                comment_to_theme[comment] = theme
                comment_to_metadata[comment] = (url, short_post)
            else:
                current_priority = theme_priority[comment_to_theme[comment]]
                new_priority = theme_priority[theme]
                if new_priority < current_priority:
                    comment_to_theme[comment] = theme
                    comment_to_metadata[comment] = (url, short_post)

        # Теперь собираем по темам
        theme_to_comments = defaultdict(list)
        theme_to_metadata = defaultdict(list)
        for comment, theme in comment_to_theme.items():
            url, short_post = comment_to_metadata[comment]
            theme_to_comments[theme].append(comment)
            theme_to_metadata[theme].append((comment, url, short_post))

        themes_res = []
        meta_res = []
        for theme in sorted_themes:
            examples = theme_to_comments.get(theme, [])[:3]
            meta_examples = theme_to_metadata.get(theme, [])[:3]
            themes_res.append((theme, theme_counts[theme], examples))
            meta_res.append(meta_examples)
        return themes_res, meta_res

    def generate_summary_and_recommendations(self, themes: Dict[str, List[Tuple[str, int, List[str]]]]) -> str:
        if self.cache.get("summary"):
            return self.cache["summary"]
        neg_list = "\n".join([f"- {theme} (упоминаний: {freq})" for theme, freq, _ in themes.get("negative_themes", [])])
        pos_list = "\n".join([f"- {theme} (упоминаний: {freq})" for theme, freq, _ in themes.get("positive_themes", [])])
        if not neg_list:
            neg_list = "Нет значимых негативных тем."
        if not pos_list:
            pos_list = "Нет значимых позитивных тем."
        prompt = f"""
Ты — опытный бизнес-аналитик. На основе анализа комментариев пользователей выделены следующие темы:

НЕГАТИВНЫЕ ТЕМЫ:
{neg_list}

ПОЗИТИВНЫЕ ТЕМЫ:
{pos_list}

Напиши ДЕТАЛЬНОЕ РЕЗЮМЕ (2-3 абзаца): общий эмоциональный фон, главные болевые точки, сильные стороны.
Затем напиши ПРАКТИЧЕСКИЕ РЕКОМЕНДАЦИИ (5-7 пунктов): краткосрочные и долгосрочные, конкретные.

Формат:
РЕЗЮМЕ:
текст резюме

РЕКОМЕНДАЦИИ:
- рекомендация 1
- рекомендация 2
...
"""
        result = self._call_model(prompt, temperature=0.3)
        summary_text = result.get("text", "Не удалось сгенерировать резюме.")
        self.cache["summary"] = summary_text
        self._save_cache()
        return summary_text


def generate_llm_report(
    themes: Dict[str, List[Tuple[str, int, List[str]]]],
    summary_text: str,
    example_metadata: Dict[str, List[List[Tuple[str, str, str]]]],
    output_file: str = "saiga_reasons_report.html"
) -> None:
    """
    Генерирует HTML-отчёт с таблицами, включая столбцы "Текст поста" и "Ссылка на пост".
    example_metadata: dict с ключами 'negative' и 'positive', каждый содержит список списков метаданных
        для каждой темы в том же порядке, что и themes.
    """
    # Отладочный вывод (можно убрать после проверки)
    logger.debug(f"Themes keys: {themes.keys()}")
    logger.debug(f"Negative themes count: {len(themes.get('negative_themes', []))}")
    logger.debug(f"Negative metadata count: {len(example_metadata.get('negative', []))}")
    logger.debug(f"Positive themes count: {len(themes.get('positive_themes', []))}")
    logger.debug(f"Positive metadata count: {len(example_metadata.get('positive', []))}")

    html = """
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Анализ обратной связи (LLM)</title>
    <style>
        body { font-family: 'Segoe UI', Arial, sans-serif; margin: 20px; background: #f5f5f5; }
        h1 { color: #2c3e50; }
        h2 { color: #2c3e50; margin-top: 30px; border-bottom: 2px solid #ddd; padding-bottom: 5px; }
        table { border-collapse: collapse; width: 100%; background: white; box-shadow: 0 0 10px rgba(0,0,0,0.1); }
        th, td { border: 1px solid #ddd; padding: 12px; text-align: left; vertical-align: top; }
        th { background-color: #2c3e50; color: white; }
        tr:nth-child(even) { background-color: #f9f9f9; }
        .bad { border-left: 5px solid #e74c3c; padding-left: 15px; margin: 20px 0; }
        .good { border-left: 5px solid #2ecc71; padding-left: 15px; margin: 20px 0; }
        .summary { background-color: #fff3e0; border-left: 5px solid #f39c12; padding: 15px; margin: 20px 0; border-radius: 5px; }
        .recommendations { background-color: #e8f8f5; border-left: 5px solid #1abc9c; padding: 15px; margin: 20px 0; border-radius: 5px; }
        .example { font-size: 0.9em; color: #555; }
        .footer { margin-top: 30px; font-size: 0.8em; color: #777; text-align: center; border-top: 1px solid #ccc; padding-top: 15px; }
        .badge { display: inline-block; padding: 2px 8px; border-radius: 12px; font-size: 0.8em; margin-right: 8px; }
        .badge-neg { background-color: #e74c3c; color: white; }
        .badge-pos { background-color: #2ecc71; color: white; }
        .post-link { font-size: 0.85em; }
        .post-text { font-size: 0.85em; color: #2c3e50; background: #f0f0f0; padding: 4px; border-radius: 4px; }
    </style>
</head>
<body>
    <h1>📊 Анализ обратной связи (на основе LLM)</h1>
    <p>Выделены ключевые причины недовольства и положительные моменты на основе анализа комментариев нейросетью Saiga с учётом контекста постов.</p>
"""

    # ----- Негативные темы -----
    html += '<div class="bad"><h2>🔴 Основные причины недовольства</h2>'
    neg_themes = themes.get("negative_themes", [])
    neg_meta = example_metadata.get("negative", [])
    if neg_themes:
        html += '<table><th>Причина / проблема</th><th>Частота</th><th>Пример комментария</th><th>Текст поста</th><th>Ссылка на пост</th></tr>'
        for idx, (theme, freq, examples) in enumerate(neg_themes):
            comment_ex = examples[0] if examples else "—"
            # Безопасное получение метаданных
            if idx < len(neg_meta) and neg_meta[idx]:
                meta_list = neg_meta[idx]
                # Берём метаданные первого примера (можно расширить)
                _, post_url, post_short = meta_list[0]
            else:
                post_url = "#"
                post_short = "Нет данных"
            html += f'''
            <tr>
                <td><span class="badge badge-neg">негатив</span> {theme}</td>
                <td>{freq}</td>
                <td class="example">“{comment_ex[:150]}...”</td>
                <td class="post-text">{post_short}</td>
                <td class="post-link"><a href="{post_url}" target="_blank">Перейти к посту</a></td>
            </tr>
            '''
        html += '</table>'
    else:
        html += '<p>Недостаточно негативных комментариев для анализа.</p>'
    html += '</div>'

    # ----- Позитивные темы -----
    html += '<div class="good"><h2>🟢 Положительные моменты</h2>'
    pos_themes = themes.get("positive_themes", [])
    pos_meta = example_metadata.get("positive", [])
    if pos_themes:
        html += '<table>'
        html += '<tr><th>Положительный момент</th><th>Частота</th><th>Пример комментария</th><th>Текст поста</th><th>Ссылка на пост</th></tr>'
        for idx, (theme, freq, examples) in enumerate(pos_themes):
            comment_ex = examples[0] if examples else "—"
            if idx < len(pos_meta) and pos_meta[idx]:
                meta_list = pos_meta[idx]
                _, post_url, post_short = meta_list[0]
            else:
                post_url = "#"
                post_short = "Нет данных"
            html += f'''
            <tr>
                <td><span class="badge badge-pos">позитив</span> {theme}</td>
                <td>{freq}</td>
                <td class="example">“{comment_ex[:150]}...”</td>
                <td class="post-text">{post_short}</td>
                <td class="post-link"><a href="{post_url}" target="_blank">Перейти к посту</a></td>
            </tr>
            '''
        html += '</table>'
    else:
        html += '<p>Недостаточно позитивных комментариев для анализа.</p>'
    html += '</div>'
    
    # ----- Резюме и рекомендации -----
    html += '<div class="summary"><h2>📋 Резюме ситуации</h2>'
    if "РЕЗЮМЕ:" in summary_text and "РЕКОМЕНДАЦИИ:" in summary_text:
        parts = summary_text.split("РЕКОМЕНДАЦИИ:")
        summary_part = parts[0].replace("РЕЗЮМЕ:", "").strip()
        rec_part = parts[1].strip()
        for p in summary_part.split('\n'):
            if p.strip():
                html += f'<p>{p}</p>'
        html += '</div><div class="recommendations"><h2>💡 Рекомендации</h2><ul>'
        for line in rec_part.split('\n'):
            line = line.strip()
            if line.startswith('-'):
                html += f'<li>{line[1:].strip()}</li>'
            elif line:
                html += f'<li>{line}</li>'
        html += '</ul>'
    else:
        html += f'<p>{summary_text}</p></div><div class="recommendations"><h2>💡 Рекомендации</h2><p>Не удалось выделить структурированные рекомендации.</p>'
    html += '</div>'

    html += """
    <div class="footer">
        Отчёт сгенерирован автоматически на основе анализа тональности (RuBERT) и нейросетевого анализа комментариев (Saiga).<br>
        Выделены наиболее частотные темы и причины. Кэширование результатов позволяет ускорить повторные запуски.
    </div>
</body>
</html>
"""
    full_path = output_file
    if not os.path.isabs(output_file):
        full_path = os.path.join(OUTPUT_DIR, output_file)
    ensure_directory_exists(os.path.dirname(full_path))
    with open(full_path, "w", encoding="utf-8") as f:
        f.write(html)
    logger.info(f"LLM-отчёт сохранён в {full_path}")