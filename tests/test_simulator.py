"""
FASE RED — test_simulator.py
Testes para o gerador de dados simulados do torno CNC.
"""
import pytest
import pandas as pd
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from data.simulator import SimuladorTornoCNC

COLUNAS_ESPERADAS = [
    "temperatura_fuso", "vibracao_x", "vibracao_y", "vibracao_z",
    "corrente_motor", "rpm_fuso", "pressao_refrigerante",
    "desgaste_ferramenta", "label"
]


def test_simulador_retorna_dataframe():
    """Simulador deve retornar um DataFrame pandas."""
    sim = SimuladorTornoCNC(seed=42)
    df = sim.gerar(n_amostras=100)
    assert isinstance(df, pd.DataFrame)


def test_simulador_quantidade_correta_de_amostras():
    """DataFrame deve ter exatamente o número de amostras solicitado."""
    sim = SimuladorTornoCNC(seed=42)
    df = sim.gerar(n_amostras=200)
    assert len(df) == 200


def test_simulador_contem_todas_as_colunas():
    """DataFrame deve conter todas as colunas de sensores e o label."""
    sim = SimuladorTornoCNC(seed=42)
    df = sim.gerar(n_amostras=50)
    for coluna in COLUNAS_ESPERADAS:
        assert coluna in df.columns, f"Coluna ausente: {coluna}"


def test_simulador_label_binario():
    """Label deve ser binário: apenas 0 (normal) ou 1 (falha)."""
    sim = SimuladorTornoCNC(seed=42)
    df = sim.gerar(n_amostras=500)
    assert set(df["label"].unique()).issubset({0, 1})


def test_simulador_tem_ambas_as_classes():
    """Dataset deve conter amostras normais e com falha."""
    sim = SimuladorTornoCNC(seed=42)
    df = sim.gerar(n_amostras=200)
    assert 0 in df["label"].values
    assert 1 in df["label"].values


def test_simulador_sem_valores_nulos():
    """Nenhum valor nulo deve estar presente nos dados simulados."""
    sim = SimuladorTornoCNC(seed=42)
    df = sim.gerar(n_amostras=100)
    assert df.isnull().sum().sum() == 0


def test_simulador_temperatura_dentro_do_range():
    """Temperatura do fuso deve estar dentro de faixas realistas (50–190°C)."""
    sim = SimuladorTornoCNC(seed=42)
    df = sim.gerar(n_amostras=500)
    assert df["temperatura_fuso"].min() >= 50.0
    assert df["temperatura_fuso"].max() <= 190.0


def test_simulador_salva_csv(tmp_path):
    """Simulador deve salvar o CSV no caminho especificado."""
    sim = SimuladorTornoCNC(seed=42)
    caminho = tmp_path / "dados.csv"
    sim.gerar_e_salvar(n_amostras=100, caminho=str(caminho))
    assert caminho.exists()
    df = pd.read_csv(str(caminho))
    assert len(df) == 100


def test_simulador_reproducivel_com_seed():
    """Mesma seed deve gerar dados idênticos (reprodutibilidade)."""
    sim1 = SimuladorTornoCNC(seed=99)
    sim2 = SimuladorTornoCNC(seed=99)
    df1 = sim1.gerar(n_amostras=50)
    df2 = sim2.gerar(n_amostras=50)
    pd.testing.assert_frame_equal(df1, df2)
