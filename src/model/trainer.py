"""
trainer.py — Treinamento do modelo Random Forest para detecção de falhas.
"""
import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


class TreinadorModelo:
    """Treina e avalia o modelo Random Forest de manutenção preditiva."""

    def __init__(self, test_size: float = 0.2, seed: int = 42):
        self.test_size = test_size
        self.seed = seed
        self.modelo = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=seed,
        )

    def treinar(self, X: np.ndarray, y: np.ndarray) -> dict:
        X_treino, X_teste, y_treino, y_teste = train_test_split(
            X, y, test_size=self.test_size, random_state=self.seed, stratify=y
        )
        self.modelo.fit(X_treino, y_treino)
        y_pred = self.modelo.predict(X_teste)

        return {
            "acuracia":  accuracy_score(y_teste, y_pred),
            "precisao":  precision_score(y_teste, y_pred, zero_division=0),
            "recall":    recall_score(y_teste, y_pred, zero_division=0),
            "f1":        f1_score(y_teste, y_pred, zero_division=0),
        }

    def salvar(self, caminho: str) -> None:
        joblib.dump(self.modelo, caminho)

    @staticmethod
    def carregar(caminho: str) -> "TreinadorModelo":
        treinador = TreinadorModelo()
        treinador.modelo = joblib.load(caminho)
        return treinador
