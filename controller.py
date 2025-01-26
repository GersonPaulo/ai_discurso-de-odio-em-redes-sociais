from model import HateSpeechModel
from sklearn.model_selection import train_test_split
import pandas as pd

class HateSpeechController:
    def __init__(self, data_path):
        self.data_path = data_path
        self.model = HateSpeechModel()

    def load_data(self):
        """Carrega os dados do CSV."""
        data = pd.read_csv(self.data_path)
        print(f"Dados carregados: {data.shape[0]} linhas.")
        return data

    def execute_pipeline(self):
        """Executa o pipeline completo."""
        data = self.load_data()
        X, y = self.model.preprocess_data(data)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Naive Bayes
        self.model.train_naive_bayes(X_train, y_train)
        f1_score_nb = self.model.evaluate_naive_bayes(X_test, y_test)
        print(f"F1-Score Naive Bayes: {f1_score_nb:.2f}")

        # (BERT poderia ser chamado aqui para treinamento avançado)

