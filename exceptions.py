# exceptions.py
# -*- coding: utf-8 -*-
"""
Пользовательские исключения для проекта анализа тональности.

Позволяют более точно обрабатывать ошибки на разных этапах пайплайна.
"""

class VKAPIError(Exception):
    """Ошибка при вызове VK API (сетевые проблемы, неверный токен, лимиты)."""
    pass

class DataCollectionError(Exception):
    """Ошибка при сборе данных (не найдена группа, отсутствуют посты)."""
    pass

class PreprocessingError(Exception):
    """Ошибка в ходе предобработки текста (например, не удалось лемматизировать)."""
    pass

class ModelError(Exception):
    """Ошибка при загрузке или использовании модели машинного обучения."""
    pass

class ValidationError(Exception):
    """Ошибка валидации данных (неверный формат, отсутствие колонок)."""
    pass

class LLMError(Exception):
    """Ошибка при вызове LLM (Ollama/Saiga)."""
    pass