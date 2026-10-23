"""
FASE RED — test_preprocessing.py
Testes para o pipeline de pré-processamento dos dados.
"""
import pytest
import pandas as pd
import numpy as np
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from preprocessing.pipeline import PipelinePreprocessamento

FEATURES = [
    "temperatura_fuso", "vibracao_x", "vibracao_y", "vibracao_z",
    "corrente_motor", "rpm_fuso", "pressao_refrigerante", "desgaste_ferramenta"
]


def test_pipeline_retorna_arrays(dataframe_simulado):
    """Pipeline deve retornar X e y como arrays numpy."""
    pipeline = PipelinePreprocessamento()
    X, y = pipeline.fit_transform(dataframe_simulado)
    assert hasattr(X, 'shape')
    assert hasattr(y, 'shape')


def test_pipeline_numero_correto_de_features(dataframe_simulado):
    """X deve ter exatamente 8 features (sensores)."""
    pipeline = PipelinePreprocessamento()
    X, y = pipeline.fit_transform(dataframe_simulado)
    assert X.shape[1] == 8


def test_pipeline_numero_correto_de_amostras(dataframe_simulado):
    """X e y devem ter o mesmo número de linhas do DataFrame original."""
    pipeline = PipelinePreprocessamento()
    X, y = pipeline.fit_transform(dataframe_simulado)
    assert X.shape[0] == len(dataframe_simulado)
    assert y.shape[0] == len(dataframe_simulado)


def test_pipeline_normaliza_dados(dataframe_simulado):
    """Após normalização, os dados devem ter média próxima de 0 e desvio próximo de 1."""
    pipeline = PipelinePreprocessamento()
    X, _ = pipeline.fit_transform(dataframe_simulado)
    medias = np.abs(X.mean(axis=0))
    assert medias.max() < 1.0, "Médias fora do esperado após normalização"


def test_pipeline_transforma_leitura_unitaria(leitura_normal):
    """Pipeline deve transformar um dicionário de leitura única em array (1, 8)."""
    pipeline = PipelinePreprocessamento()
    # Precisa ser fitado antes de transformar
    df_treino = pd.DataFrame([leitura_normal] * 10)
    df_treino["label"] = 0
    pipeline.fit_transform(df_treino)

    X_novo = pipeline.transformar_leitura(leitura_normal)
    assert X_novo.shape == (1, 8)


def test_pipeline_nao_inclui_label_nas_features(dataframe_simulado):
    """A coluna 'label' não deve estar nas features X."""
    pipeline = PipelinePreprocessamento()
    X, y = pipeline.fit_transform(dataframe_simulado)
    # X deve ter 8 colunas, não 9
    assert X.shape[1] == 8


def test_pipeline_label_preserva_valores(dataframe_simulado):
    """O vetor y deve conter apenas os valores 0 e 1 do label original."""
    pipeline = PipelinePreprocessamento()
    _, y = pipeline.fit_transform(dataframe_simulado)
    assert set(np.unique(y)).issubset({0, 1})
