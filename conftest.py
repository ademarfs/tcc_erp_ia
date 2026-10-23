"""
conftest.py — Fixtures compartilhadas entre todos os testes
"""
import pytest
import pandas as pd
import numpy as np
import sys
import os

# Garante que src/ está no path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))


@pytest.fixture
def leitura_normal():
    """Leitura de sensores dentro dos parâmetros normais do torno CNC."""
    return {
        "temperatura_fuso": 85.0,
        "vibracao_x": 1.2,
        "vibracao_y": 0.9,
        "vibracao_z": 1.1,
        "corrente_motor": 15.0,
        "rpm_fuso": 1500.0,
        "pressao_refrigerante": 4.5,
        "desgaste_ferramenta": 45.0,
    }


@pytest.fixture
def leitura_critica():
    """Leitura de sensores com múltiplos parâmetros críticos."""
    return {
        "temperatura_fuso": 155.0,
        "vibracao_x": 6.8,
        "vibracao_y": 7.2,
        "vibracao_z": 5.5,
        "corrente_motor": 28.0,
        "rpm_fuso": 1500.0,
        "pressao_refrigerante": 1.5,
        "desgaste_ferramenta": 91.0,
    }


@pytest.fixture
def leitura_alerta():
    """Leitura de sensores com um parâmetro em alerta."""
    return {
        "temperatura_fuso": 145.0,
        "vibracao_x": 1.0,
        "vibracao_y": 0.8,
        "vibracao_z": 1.0,
        "corrente_motor": 16.0,
        "rpm_fuso": 1500.0,
        "pressao_refrigerante": 4.5,
        "desgaste_ferramenta": 60.0,
    }


@pytest.fixture
def dataframe_simulado():
    """DataFrame com 200 amostras (150 normais, 50 com falha)."""
    np.random.seed(42)
    n_normal = 150
    n_falha = 50

    normais = pd.DataFrame({
        "temperatura_fuso": np.random.uniform(60, 120, n_normal),
        "vibracao_x": np.random.uniform(0, 2, n_normal),
        "vibracao_y": np.random.uniform(0, 2, n_normal),
        "vibracao_z": np.random.uniform(0, 2, n_normal),
        "corrente_motor": np.random.uniform(10, 20, n_normal),
        "rpm_fuso": np.random.uniform(500, 3000, n_normal),
        "pressao_refrigerante": np.random.uniform(3, 6, n_normal),
        "desgaste_ferramenta": np.random.uniform(0, 70, n_normal),
        "label": 0,
    })

    falhas = pd.DataFrame({
        "temperatura_fuso": np.random.uniform(140, 180, n_falha),
        "vibracao_x": np.random.uniform(5, 10, n_falha),
        "vibracao_y": np.random.uniform(5, 10, n_falha),
        "vibracao_z": np.random.uniform(5, 10, n_falha),
        "corrente_motor": np.random.uniform(25, 35, n_falha),
        "rpm_fuso": np.random.uniform(500, 3000, n_falha),
        "pressao_refrigerante": np.random.uniform(0.5, 2, n_falha),
        "desgaste_ferramenta": np.random.uniform(85, 100, n_falha),
        "label": 1,
    })

    return pd.concat([normais, falhas], ignore_index=True)
