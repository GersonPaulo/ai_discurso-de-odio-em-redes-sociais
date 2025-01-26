
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from transformers import BertTokenizer, BertForSequenceClassification
from sklearn.metrics import classification_report, f1_score

class HateSpeechModel:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(max_features=5000)
        self.nb_model = MultinomialNB()
        self.bert_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        self.bert_model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)

    def preprocess_data(self, data):
        """Processa e divide os dados."""
        X = data['text']
        y = data['label']
        return X, y

    def train_naive_bayes(self, X_train, y_train):
        """Treina o modelo Naive Bayes."""
        X_train_tfidf = self.vectorizer.fit_transform(X_train)
        self.nb_model.fit(X_train_tfidf, y_train)

    def evaluate_naive_bayes(self, X_test, y_test):
        """Avalia o modelo Naive Bayes."""
        X_test_tfidf = self.vectorizer.transform(X_test)
        y_pred = self.nb_model.predict(X_test_tfidf)
        print("Resultados Naive Bayes:")
        print(classification_report(y_test, y_pred))
        return f1_score(y_test, y_pred)

    def tokenize_for_bert(self, texts, labels):
        """Tokeniza os dados para o modelo BERT."""
        tokens = self.bert_tokenizer(list(texts), max_length=128, padding=True, truncation=True, return_tensors="pt")
        tokens["labels"] = labels
        return tokens
