# vk_api.py
# -*- coding: utf-8 -*-
"""
Модуль для сбора данных из VK API с обработкой ошибок, повторными попытками и логированием.
"""

import re
import time
import requests
from typing import List, Tuple, Any, Optional

from config import VK_TOKEN, VK_API_VERSION, VK_TIMEOUT, REQUEST_DELAY
from exceptions import VKAPIError, DataCollectionError
from utils import logger, handle_exceptions

def extract_hashtags(text: str) -> list:
    """Извлекает хэштеги из текста поста."""
    if not text:
        return []
    return re.findall(r'#([а-яёa-z0-9_]+)', text.lower())

@handle_exceptions(default_return=None, reraise=True)
def vk_api_request(method: str, params: dict, retries: int = 3) -> Any:
    """
    Универсальный вызов VK API с повторными попытками при ошибках.
    
    Args:
        method: метод API (например, 'wall.get')
        params: параметры запроса
        retries: количество повторных попыток
    
    Returns:
        ответ API (поле 'response')
    
    Raises:
        VKAPIError: при превышении лимитов или ошибках авторизации
    """
    params["access_token"] = VK_TOKEN
    params["v"] = VK_API_VERSION
    
    for attempt in range(retries):
        try:
            time.sleep(REQUEST_DELAY * (attempt + 1))  # увеличиваем задержку при повторах
            response = requests.get(
                f"https://api.vk.com/method/{method}",
                params=params,
                timeout=VK_TIMEOUT
            )
            response.raise_for_status()
            data = response.json()
        except requests.exceptions.Timeout:
            logger.warning(f"Таймаут запроса к {method}, попытка {attempt+1}/{retries}")
            if attempt == retries - 1:
                raise VKAPIError(f"Таймаут VK API после {retries} попыток") from None
            continue
        except requests.exceptions.RequestException as e:
            logger.warning(f"Сетевая ошибка: {e}, попытка {attempt+1}/{retries}")
            if attempt == retries - 1:
                raise VKAPIError(f"Сетевая ошибка VK API: {e}") from e
            continue
        except Exception as e:
            logger.error(f"Неизвестная ошибка при запросе: {e}")
            if attempt == retries - 1:
                raise VKAPIError(f"Неизвестная ошибка: {e}") from e
            continue

        # Проверка ошибок VK в ответе
        if "error" in data:
            error = data["error"]
            error_code = error.get("error_code")
            error_msg = error.get("error_msg", "Неизвестная ошибка VK")
            
            # Код 6: слишком много запросов в секунду
            if error_code == 6 and attempt < retries - 1:
                wait_time = 2 ** attempt
                logger.warning(f"Лимит запросов VK, пауза {wait_time} сек, попытка {attempt+1}/{retries}")
                time.sleep(wait_time)
                continue
            # Код 5: авторизация
            elif error_code == 5:
                raise VKAPIError("Неверный токен доступа VK")
            else:
                raise VKAPIError(f"VK API error {error_code}: {error_msg}")
        
        return data["response"]
    
    raise VKAPIError(f"Не удалось выполнить {method} после {retries} попыток")

def get_group_owner_id(screen_name: str) -> int:
    """
    Получает owner_id сообщества по его короткому имени.
    
    Args:
        screen_name: короткое имя группы (например, 'pressrkomi')
    
    Returns:
        owner_id (отрицательное число для групп)
    
    Raises:
        DataCollectionError: если группа не найдена
    """
    resp = vk_api_request("utils.resolveScreenName", {"screen_name": screen_name})
    if resp and resp.get("type") == "group":
        return -resp["object_id"]
    else:
        raise DataCollectionError(f"Группа {screen_name} не найдена или недоступна")

def get_wall_posts(owner_id: int, count: int = 10, offset: int = 0) -> List[Tuple[str, str, int]]:
    """
    Получает посты со стены сообщества.
    
    Returns:
        список кортежей (type='post', text, post_id)
    """
    try:
        posts = vk_api_request("wall.get", {
            "owner_id": owner_id,
            "count": count,
            "offset": offset
        })
    except VKAPIError as e:
        logger.error(f"Не удалось получить посты owner_id={owner_id}: {e}")
        return []
    
    items = posts.get("items", [])
    texts = []
    for item in items:
        text = item.get("text", "").strip()
        if text:
            texts.append(("post", text, item["id"]))
    return texts

def get_all_comments(owner_id: int, post_id: int) -> List[Tuple[str, str, int, int, int]]:
    """
    Загружает все комментарии к посту (пагинация).
    
    Returns:
        список кортежей ("comment", текст, id_комментария, id_поста, id_автора)
    """
    all_comments = []
    offset = 0
    count_per_request = 100
    
    while True:
        try:
            comments = vk_api_request("wall.getComments", {
                "owner_id": owner_id,
                "post_id": post_id,
                "count": count_per_request,
                "offset": offset,
                "need_likes": 0,
                "extended": 0
            })
        except VKAPIError as e:
            logger.error(f"Ошибка загрузки комментариев к посту {post_id} (offset={offset}): {e}")
            break
        
        items = comments.get("items", [])
        if not items:
            break
        
        for item in items:
            text = item.get("text", "").strip()
            if text:
                author_id = item.get("from_id", None)
                all_comments.append(("comment", text, item["id"], post_id, author_id))
        
        if len(items) < count_per_request:
            break
        offset += count_per_request
        time.sleep(REQUEST_DELAY)
    
    return all_comments

def collect_all_posts(
    groups: List[str],
    posts_per_group: int,
    comments_per_post: Optional[int] = None,
    offset: int = 0
) -> List[Tuple]:
    """
    Собирает посты и комментарии для списка групп.
    
    Returns:
        список кортежей (group, text, item_id, owner_id, type, parent_id, author_id, hashtags)
    """
    all_items = []
    group_owners = {}
    
    # Получаем owner_id для всех групп
    for group in groups:
        try:
            owner_id = get_group_owner_id(group)
            group_owners[group] = owner_id
            logger.info(f"Группа {group}: owner_id = {owner_id}")
        except DataCollectionError as e:
            logger.error(f"Пропускаем группу {group}: {e}")
            continue
    
    # Сбор данных
    for group, owner_id in group_owners.items():
        try:
            posts = get_wall_posts(owner_id, posts_per_group, offset)
            logger.info(f"{group}: получено {len(posts)} постов")
            
            for p_type, text, p_id in posts:
                hashtags = extract_hashtags(text)
                all_items.append((group, text, p_id, owner_id, "post", None, owner_id, hashtags))
                
                # Загрузка комментариев
                comments = get_all_comments(owner_id, p_id)
                if comments_per_post is not None and len(comments) > comments_per_post:
                    comments = comments[:comments_per_post]
                
                for c_type, c_text, c_id, parent_id, author_id in comments:
                    all_items.append((group, c_text, c_id, owner_id, "comment", parent_id, author_id, []))
                
                logger.info(f"  Пост {p_id}: собрано {len(comments)} комментариев")
            
            time.sleep(REQUEST_DELAY)
        except Exception as e:
            logger.error(f"Ошибка при сборе группы {group}: {e}", exc_info=True)
    
    return all_items