# utils.py
# -*- coding: utf-8 -*-
"""
Утилиты для проекта: логирование, замер времени, сохранение результатов.
"""

import time
import logging
import functools
import pandas as pd
from typing import Any, Callable

# ----------------------------------------------------------------------
# Настройка логирования
# ----------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("app.log", encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# ----------------------------------------------------------------------
# Декораторы
# ----------------------------------------------------------------------
def timeit(func: Callable) -> Callable:
    """
    Декоратор для измерения времени выполнения функции.
    Логирует результат в info-уровень.
    
    Args:
        func: целевая функция
    
    Returns:
        обёрнутая функция
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        try:
            result = func(*args, **kwargs)
            elapsed = time.time() - start
            logger.info(f"{func.__name__} выполнена за {elapsed:.2f} сек")
            return result
        except Exception as e:
            elapsed = time.time() - start
            logger.error(f"{func.__name__} упала после {elapsed:.2f} сек с ошибкой: {e}")
            raise
    return wrapper

def handle_exceptions(default_return: Any = None, reraise: bool = False) -> Callable:
    """
    Декоратор для унифицированной обработки исключений в функциях.
    
    Args:
        default_return: значение, возвращаемое при возникновении исключения (если reraise=False)
        reraise: если True, исключение пробрасывается дальше
    
    Returns:
        декоратор
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                logger.error(f"В {func.__name__} возникла ошибка: {e}", exc_info=True)
                if reraise:
                    raise
                return default_return
        return wrapper
    return decorator

# ----------------------------------------------------------------------
# Функции работы с данными
# ----------------------------------------------------------------------
def save_results(df: pd.DataFrame, filename: str) -> None:
    """
    Сохраняет DataFrame в CSV с обработкой ошибок.
    
    Args:
        df: DataFrame для сохранения
        filename: путь к файлу
    
    Raises:
        IOError: если не удаётся записать файл
    """
    try:
        df.to_csv(filename, index=False, encoding='utf-8-sig')
        logger.info(f"Результаты сохранены в {filename}")
    except Exception as e:
        logger.error(f"Не удалось сохранить {filename}: {e}")
        raise IOError(f"Ошибка записи {filename}: {e}") from e

def log_message(msg: str, level: str = "info") -> None:
    """
    Унифицированное логирование сообщений.
    
    Args:
        msg: текст сообщения
        level: уровень ('debug', 'info', 'warning', 'error')
    """
    getattr(logger, level.lower(), logger.info)(msg)

def ensure_directory_exists(path: str) -> None:
    """
    Создаёт директорию, если её нет.
    
    Args:
        path: путь к директории
    """
    import os
    if not os.path.exists(path):
        os.makedirs(path)
        logger.debug(f"Создана директория: {path}")