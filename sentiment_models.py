# sentiment_models.py
# -*- coding: utf-8 -*-
"""
Модели анализа тональности: классическая (TF‑IDF + LR) и трансформер (RuBERT).
Содержит загрузку, обучение, предсказание с обработкой ошибок.
"""

import os
import joblib
import numpy as np
import pandas as pd
import torch
from typing import List, Tuple, Optional
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from config import BASELINE_MODEL_PATH, TFIDF_VECTORIZER_PATH
from preprocessing import clean_text, lemmatize
from exceptions import ModelError, ValidationError
from utils import logger, ensure_directory_exists

# ----------------------------------------------------------------------
# Baseline модель (TF‑IDF + логистическая регрессия)
# ----------------------------------------------------------------------
class BaselineSentimentModel:
    """
    Классическая модель анализа тональности на основе TF‑IDF и логистической регрессии.
    
    Attributes:
        vectorizer (TfidfVectorizer): векторизатор текста
        model (LogisticRegression): классификатор
    """
    
    def __init__(self):
        self.vectorizer = None
        self.model = None
    
    def _validate_texts(self, texts: List[str]) -> List[str]:
        """Проверяет, что тексты не пусты."""
        if not texts:
            raise ValidationError("Список текстов пуст")
        valid = [t for t in texts if t and t.strip()]
        if not valid:
            raise ValidationError("Все тексты пусты после очистки")
        return valid
    
    def train(self, texts: List[str], labels: List[int], test_size: float = 0.2, random_state: int = 42) -> None:
        """
        Обучает модель на размеченных данных.
        
        Args:
            texts: список исходных текстов
            labels: список меток (0,1,2)
            test_size: доля тестовой выборки
            random_state: seed для воспроизводимости
        
        Raises:
            ValidationError: если данные некорректны
            ModelError: если обучение не удалось
        """
        try:
            # Очистка и лемматизация
            cleaned = [clean_text(t) for t in texts]
            lemmatized = [lemmatize(t) for t in cleaned]
            
            data = pd.DataFrame({'text': lemmatized, 'label': labels})
            data = data[data['text'].str.strip() != ""]
            
            if len(data) < 10:
                raise ValidationError(f"Недостаточно данных для обучения: {len(data)} примеров")
            
            X = data['text']
            y = data['label']
            
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=random_state, stratify=y
            )
            
            self.vectorizer = TfidfVectorizer(max_features=5000)
            X_train_vec = self.vectorizer.fit_transform(X_train)
            X_test_vec = self.vectorizer.transform(X_test)
            
            self.model = LogisticRegression(max_iter=1000, class_weight='balanced')
            self.model.fit(X_train_vec, y_train)
            
            y_pred = self.model.predict(X_test_vec)
            acc = accuracy_score(y_test, y_pred)
            logger.info(f"Baseline модель обучена. Accuracy на тесте: {acc:.3f}")
            logger.debug(f"Classification report:\n{classification_report(y_test, y_pred)}")
            
        except Exception as e:
            logger.error(f"Ошибка при обучении baseline модели: {e}", exc_info=True)
            raise ModelError(f"Не удалось обучить baseline модель: {e}") from e
    
    def train_from_labeled_csv(self, csv_path: str = "labeled_data.csv", test_size: float = 0.2, random_state: int = 42) -> None:
        """
        Загружает размеченные данные из CSV и обучает модель.
        """
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"Файл {csv_path} не найден.")
        
        df = pd.read_csv(csv_path)
        df = df.dropna(subset=["text", "label"])
        df = df[df["text"].str.strip() != ""]
        
        if len(df) < 10:
            raise ValidationError(f"Слишком мало данных в {csv_path}: {len(df)}")
        
        texts = df["text"].tolist()
        labels = df["label"].tolist()
        self.train(texts, labels, test_size, random_state)
    
    def predict(self, texts: List[str]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Предсказывает тональность для списка текстов.
        
        Returns:
            (predictions, probabilities) – метки классов (0/1/2) и вероятности для класса 1 (нейтраль)
        
        Raises:
            ModelError: если модель не обучена или не загружена
        """
        if self.vectorizer is None or self.model is None:
            raise ModelError("Модель не обучена или не загружена")
        
        try:
            texts = self._validate_texts(texts)
            cleaned = [clean_text(t) for t in texts]
            lemmatized = [lemmatize(t) for t in cleaned]
            X = self.vectorizer.transform(lemmatized)
            preds = self.model.predict(X)
            probs = self.model.predict_proba(X)[:, 1]  # вероятность нейтрального класса
            return preds, probs
        except Exception as e:
            logger.error(f"Ошибка при предсказании baseline: {e}")
            raise ModelError(f"Ошибка инференса baseline модели: {e}") from e
    
    def save(self, vec_path: str = TFIDF_VECTORIZER_PATH, model_path: str = BASELINE_MODEL_PATH) -> None:
        """Сохраняет векторизатор и модель в файлы."""
        ensure_directory_exists(os.path.dirname(vec_path))
        ensure_directory_exists(os.path.dirname(model_path))
        try:
            joblib.dump(self.vectorizer, vec_path)
            joblib.dump(self.model, model_path)
            logger.info(f"Baseline модель сохранена в {model_path}")
        except Exception as e:
            logger.error(f"Ошибка сохранения модели: {e}")
            raise ModelError(f"Не удалось сохранить модель: {e}") from e
    
    def load(self, vec_path: str = TFIDF_VECTORIZER_PATH, model_path: str = BASELINE_MODEL_PATH) -> None:
        """Загружает векторизатор и модель из файлов."""
        if not os.path.exists(vec_path) or not os.path.exists(model_path):
            raise FileNotFoundError(f"Файлы модели не найдены: {vec_path} или {model_path}")
        try:
            self.vectorizer = joblib.load(vec_path)
            self.model = joblib.load(model_path)
            logger.info(f"Baseline модель загружена из {model_path}")
        except Exception as e:
            logger.error(f"Ошибка загрузки модели: {e}")
            raise ModelError(f"Не удалось загрузить модель: {e}") from e


# ----------------------------------------------------------------------
# RuBERT модель (трансформер)
# ----------------------------------------------------------------------
class RuBertSentimentModel:
    """
    Модель на основе предобученного RuBERT (с возможностью дообучения).
    
    Attributes:
        tokenizer: токенизатор HuggingFace
        model: трансформер для классификации
        device: 'cuda' или 'cpu'
        batch_size: размер батча для инференса
    """
    
    def __init__(self, model_path: str = "./fine_tuned_model", batch_size: int = 32):
        """
        Args:
            model_path: путь к дообученной модели (или имя стандартной модели)
            batch_size: размер батча для предсказаний
        
        Raises:
            ModelError: если не удаётся загрузить модель
        """
        self.batch_size = batch_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        try:
            if os.path.exists(model_path):
                self.tokenizer = AutoTokenizer.from_pretrained(model_path)
                self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
                logger.info(f"Загружена дообученная модель из {model_path}")
            else:
                # Попробуем стандартную модель
                model_name = "cointegrated/rubert-tiny-sentiment-balanced"
                self.tokenizer = AutoTokenizer.from_pretrained(model_name)
                self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
                logger.info(f"Загружена стандартная модель {model_name}")
            
            self.model.to(self.device)
            self.model.eval()
        except Exception as e:
            logger.error(f"Ошибка загрузки RuBERT модели: {e}", exc_info=True)
            raise ModelError(f"Не удалось загрузить RuBERT модель: {e}") from e
    
    def predict_batch(self, texts: List[str]) -> List[Tuple[str, float]]:
        """
        Предсказывает тональность для батча текстов.
        
        Returns:
            список кортежей (sentiment, confidence), где sentiment: 'negative'/'neutral'/'positive'
        """
        if not texts:
            return []
        
        results = []
        for i in range(0, len(texts), self.batch_size):
            batch_texts = texts[i:i+self.batch_size]
            try:
                inputs = self.tokenizer(
                    batch_texts,
                    return_tensors="pt",
                    truncation=True,
                    padding=True,
                    max_length=512
                )
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                
                with torch.no_grad():
                    outputs = self.model(**inputs)
                logits = outputs.logits
                probs = torch.softmax(logits, dim=-1).cpu().numpy()
                
                for prob in probs:
                    # prob: [negative, neutral, positive]
                    if prob[2] > prob[0] and prob[2] > prob[1]:
                        sentiment = "positive"
                        confidence = prob[2]
                    elif prob[0] > prob[1] and prob[0] > prob[2]:
                        sentiment = "negative"
                        confidence = prob[0]
                    else:
                        sentiment = "neutral"
                        confidence = prob[1]
                    results.append((sentiment, float(confidence)))
            except Exception as e:
                logger.error(f"Ошибка при обработке батча: {e}")
                # Для каждого текста в батче возвращаем нейтральную заглушку
                for _ in batch_texts:
                    results.append(("neutral", 0.5))
        return results
    
    def predict_single(self, text: str) -> Tuple[str, float]:
        """Предсказывает тональность для одного текста."""
        return self.predict_batch([text])[0]