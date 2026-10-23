"""
predictor.py — Interface de predição para leituras individuais de sensores.
"""
from sklearn.ensemble import RandomForestClassifier
from preprocessing.pipeline import PipelinePreprocessamento


class PreditorFalha:
    """Encapsula o modelo treinado e o pipeline para predição em tempo real."""

    def __init__(self, modelo: RandomForestClassifier,
                 pipeline: PipelinePreprocessamento):
        self.modelo = modelo
        self.pipeline = pipeline

    def prever(self, leitura: dict) -> dict:
        X = self.pipeline.transformar_leitura(leitura)
        classe = int(self.modelo.predict(X)[0])
        prob_falha = float(self.modelo.predict_proba(X)[0][1])
        return {
            "classe": classe,
            "probabilidade_falha": round(prob_falha, 4),
            "diagnostico": "FALHA IMINENTE" if classe == 1 else "NORMAL",
        }
