"""
pipeline.py — Pré-processamento e normalização dos dados de sensores.
"""
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

FEATURES = [
    "temperatura_fuso", "vibracao_x", "vibracao_y", "vibracao_z",
    "corrente_motor", "rpm_fuso", "pressao_refrigerante", "desgaste_ferramenta",
]


class PipelinePreprocessamento:
    """Normaliza os dados de sensores e separa features do label."""

    def __init__(self):
        self.scaler = StandardScaler()
        self._fitado = False

    def fit_transform(self, df: pd.DataFrame):
        X = df[FEATURES].values
        y = df["label"].values
        X_norm = self.scaler.fit_transform(X)
        self._fitado = True
        return X_norm, y

    def transformar_leitura(self, leitura: dict) -> np.ndarray:
        if not self._fitado:
            raise RuntimeError("Pipeline não foi fitado. Chame fit_transform primeiro.")
        valores = [[leitura[f] for f in FEATURES]]
        return self.scaler.transform(valores)
