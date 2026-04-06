# tests/test_preprocessing.py
# -*- coding: utf-8 -*-
"""
Unit-тесты для модуля preprocessing.py.

Запуск:
    pytest tests/test_preprocessing.py -v
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pytest
from preprocessing import clean_text, normalize_emojis, normalize_text_smileys, lemmatize


class TestNormalizeEmojis:
    """Тесты для функции normalize_emojis."""
    
    def test_simple_emoji(self):
        text = "Привет 😊"
        result = normalize_emojis(text)
        # Ожидаем, что эмодзи заменится на :smiling_face: (или аналогичное)
        assert ":smiling_face:" in result or "😊" not in result
    
    def test_multiple_emojis(self):
        text = "😡❤️😂"
        result = normalize_emojis(text)
        # Проверяем, что исходные эмодзи отсутствуют
        assert "😡" not in result
        assert "❤️" not in result
        assert "😂" not in result
    
    def test_no_emoji(self):
        text = "Обычный текст без эмодзи"
        result = normalize_emojis(text)
        assert result == text
    
    def test_empty_string(self):
        assert normalize_emojis("") == ""
        assert normalize_emojis(None) is None  # поведение может быть другим, но мы проверяем
    
    # Если библиотека emoji не установлена, тесты должны проходить без ошибок
    @pytest.mark.skipif(not hasattr(normalize_emojis, '__code__'), reason="emoji not installed")
    def test_edge_cases(self):
        assert normalize_emojis("   ") == "   "


class TestNormalizeTextSmileys:
    """Тесты для функции normalize_text_smileys."""
    
    def test_smiley_happy(self):
        text = "Привет :) Как дела?"
        result = normalize_text_smileys(text)
        assert ":smiling_face:" in result
        assert ":)" not in result
    
    def test_smiley_sad(self):
        text = "Ужас :("
        result = normalize_text_smileys(text)
        assert ":crying_face:" in result
        assert ":(" not in result
    
    def test_smiley_wink(self):
        text = "Шучу ;)"
        result = normalize_text_smileys(text)
        assert ":winking_face:" in result
    
    def test_multiple_smileys(self):
        text = ":) :( :D ;)"
        result = normalize_text_smileys(text)
        assert result.count(":smiling_face:") == 1
        assert result.count(":crying_face:") == 1
        assert result.count(":laughing_face:") == 1
        assert result.count(":winking_face:") == 1
    
    def test_no_smileys(self):
        text = "Обычный текст"
        result = normalize_text_smileys(text)
        assert result == text


class TestCleanText:
    """Тесты для основной функции clean_text."""
    
    def test_basic_cleaning(self):
        text = "Привет! Это текст с https://example.com и @username."
        result = clean_text(text)
        assert "https" not in result
        assert "@username" not in result
        assert "привет" in result
    
    def test_emoji_and_smileys(self):
        text = "Отлично 😊 и плохо :("
        result = clean_text(text)
        assert ":smiling_face:" in result
        assert ":crying_face:" in result
        assert "😊" not in result
        assert ":(" not in result
    
    def test_repeated_chars(self):
        text = "Привееееет!!!"
        result = clean_text(text)
        # Наша clean_text не сжимает повторы букв (это отдельный этап, которого нет в текущей реализации)
        # Но проверяем, что спецсимволы удалены
        assert "!" not in result
        assert "привееееет" in result  # без восклицательных
    
    def test_hashtags(self):
        text = "#хорошо #плохо"
        result = clean_text(text)
        assert "хорошо" in result
        assert "плохо" in result
        assert "#" not in result  # решётка удаляется, но слово остаётся
    
    def test_empty_or_whitespace(self):
        assert clean_text("") == ""
        assert clean_text("   ") == ""
        assert clean_text(None) == ""
    
    def test_only_special_chars(self):
        text = "!@#$%^&*()"
        result = clean_text(text)
        assert result == "" or result == " "  # зависит от реализации


class TestLemmatize:
    """Тесты для функции lemmatize (если доступна pymystem3)."""
    
    @pytest.mark.skipif(not hasattr(lemmatize, '__code__'), reason="pymystem3 not installed")
    def test_basic_lemmatization(self):
        text = "книги читают студенты"
        result = lemmatize(text)
        # Ожидаем что-то вроде "книга читать студент"
        assert "книг" in result or "книга" in result
        assert "чита" in result
        assert "студент" in result
    
    @pytest.mark.skipif(not hasattr(lemmatize, '__code__'), reason="pymystem3 not installed")
    def test_empty_string(self):
        assert lemmatize("") == ""
    
    def test_without_pymystem(self):
        # Если библиотеки нет, lemmatize возвращает исходный текст
        if not hasattr(lemmatize, '__code__'):
            text = "тест"
            assert lemmatize(text) == text


# Запуск всех тестов
if __name__ == "__main__":
    pytest.main([__file__, "-v"])