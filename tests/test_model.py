"""
FASE RED — test_model.py
Testes para o treinamento e predição do modelo Random Forest.
"""
import pytest
import numpy as np
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from preprocessing.pipeline import PipelinePreprocessamento
from model.trainer import TreinadorModelo
from model.predictor import PreditorFalha


def test_treinador_treina_sem_erro(dataframe_simulado):
    """Treinamento deve completar sem levantar exceções."""
    pipeline = PipelinePreprocessamento()
    X, y = pipeline.fit_transform(dataframe_simulado)
    treinador = TreinadorModelo()
    treinador.treinar(X, y)  # Não deve lançar exceção


def test_treinador_retorna_acuracia_acima_de_80(dataframe_simulado):
    """Acurácia no conjunto de teste deve ser >= 80%."""
    pipeline = PipelinePreprocessamento()
    X, y = pipeline.fit_transform(dataframe_simulado)
    treinador = TreinadorModelo(test_size=0.2, seed=42)
    metricas = treinador.treinar(X, y)
    assert metricas["acuracia"] >= 0.80, (
        f"Acurácia insuficiente: {metricas['acuracia']:.2%}"
    )


def test_treinador_retorna_metricas_completas(dataframe_simulado):
    """Métricas devem incluir acurácia, precisão, recall e f1."""
    pipeline = PipelinePreprocessamento()
    X, y = pipeline.fit_transform(dataframe_simulado)
    treinador = TreinadorModelo(seed=42)
    metricas = treinador.treinar(X, y)
    for chave in ["acuracia", "precisao", "recall", "f1"]:
        assert chave in metricas, f"Métrica ausente: {chave}"


def test_treinador_salva_modelo(dataframe_simulado, tmp_path):
    """Modelo deve ser salvo no caminho especificado."""
    pipeline = PipelinePreprocessamento()
    X, y = pipeline.fit_transform(dataframe_simulado)
    treinador = TreinadorModelo(seed=42)
    treinador.treinar(X, y)
    caminho = str(tmp_path / "modelo.joblib")
    treinador.salvar(caminho)
    assert os.path.exists(caminho)


def test_preditor_retorna_classe_binaria(dataframe_simulado, leitura_normal):
    """Preditor deve retornar 0 ou 1."""
    pipeline = PipelinePreprocessamento()
    X, y = pipeline.fit_transform(dataframe_simulado)
    treinador = TreinadorModelo(seed=42)
    treinador.treinar(X, y)

    preditor = PreditorFalha(modelo=treinador.modelo, pipeline=pipeline)
    resultado = preditor.prever(leitura_normal)
    assert resultado["classe"] in [0, 1]


def test_preditor_retorna_probabilidade(dataframe_simulado, leitura_normal):
    """Preditor deve retornar probabilidade entre 0 e 1."""
    pipeline = PipelinePreprocessamento()
    X, y = pipeline.fit_transform(dataframe_simulado)
    treinador = TreinadorModelo(seed=42)
    treinador.treinar(X, y)

    preditor = PreditorFalha(modelo=treinador.modelo, pipeline=pipeline)
    resultado = preditor.prever(leitura_normal)
    assert 0.0 <= resultado["probabilidade_falha"] <= 1.0


def test_preditor_leitura_normal_nao_e_falha(dataframe_simulado, leitura_normal):
    """Leitura dentro dos parâmetros normais não deve ser classificada como falha."""
    pipeline = PipelinePreprocessamento()
    X, y = pipeline.fit_transform(dataframe_simulado)
    treinador = TreinadorModelo(seed=42)
    treinador.treinar(X, y)

    preditor = PreditorFalha(modelo=treinador.modelo, pipeline=pipeline)
    resultado = preditor.prever(leitura_normal)
    assert resultado["classe"] == 0


def test_preditor_leitura_critica_e_falha(dataframe_simulado, leitura_critica):
    """Leitura com sensores críticos deve ser classificada como falha iminente."""
    pipeline = PipelinePreprocessamento()
    X, y = pipeline.fit_transform(dataframe_simulado)
    treinador = TreinadorModelo(seed=42)
    treinador.treinar(X, y)

    preditor = PreditorFalha(modelo=treinador.modelo, pipeline=pipeline)
    resultado = preditor.prever(leitura_critica)
    assert resultado["classe"] == 1
