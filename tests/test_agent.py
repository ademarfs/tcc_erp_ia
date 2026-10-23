"""
FASE RED — test_agent.py
Testes para as ferramentas e orquestração do Agente de IA.
"""
import pytest
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from agent.tools import FerramentasAgente
from agent.agent import AgenteManutencaoPreditiva
from preprocessing.pipeline import PipelinePreprocessamento
from model.trainer import TreinadorModelo
from model.predictor import PreditorFalha


# ── Fixtures locais ────────────────────────────────────────────────────

@pytest.fixture
def preditor_treinado(dataframe_simulado):
    pipeline = PipelinePreprocessamento()
    X, y = pipeline.fit_transform(dataframe_simulado)
    treinador = TreinadorModelo(seed=42)
    treinador.treinar(X, y)
    return PreditorFalha(modelo=treinador.modelo, pipeline=pipeline)


@pytest.fixture
def ferramentas(preditor_treinado):
    return FerramentasAgente(preditor=preditor_treinado)


@pytest.fixture
def agente(preditor_treinado):
    ferramentas = FerramentasAgente(preditor=preditor_treinado)
    return AgenteManutencaoPreditiva(ferramentas=ferramentas)


# ── Testes: FerramentasAgente ──────────────────────────────────────────

def test_ferramenta_analisar_sensores_retorna_dict(ferramentas, leitura_normal):
    resultado = ferramentas.analisar_sensores(leitura_normal)
    assert isinstance(resultado, dict)


def test_ferramenta_analisar_sensores_leitura_normal_sem_criticos(ferramentas, leitura_normal):
    resultado = ferramentas.analisar_sensores(leitura_normal)
    assert len(resultado["sensores_criticos"]) == 0


def test_ferramenta_analisar_sensores_leitura_critica_detecta_problemas(ferramentas, leitura_critica):
    resultado = ferramentas.analisar_sensores(leitura_critica)
    assert len(resultado["sensores_criticos"]) > 0


def test_ferramenta_prever_falha_retorna_dict(ferramentas, leitura_normal):
    resultado = ferramentas.prever_falha(leitura_normal)
    assert isinstance(resultado, dict)
    assert "classe" in resultado
    assert "probabilidade_falha" in resultado


def test_ferramenta_gerar_alerta_retorna_string(ferramentas, leitura_critica):
    alerta = ferramentas.gerar_alerta(leitura_critica, nivel="CRITICO")
    assert isinstance(alerta, str)
    assert len(alerta) > 0


def test_ferramenta_abrir_ordem_manutencao_retorna_dict(ferramentas, leitura_critica):
    ordem = ferramentas.abrir_ordem_manutencao(leitura_critica)
    assert isinstance(ordem, dict)
    assert "numero_ordem" in ordem
    assert "status" in ordem
    assert ordem["status"] == "ABERTA"


def test_ferramenta_gerar_relatorio_retorna_dict(ferramentas, leitura_normal):
    analise = ferramentas.analisar_sensores(leitura_normal)
    predicao = ferramentas.prever_falha(leitura_normal)
    relatorio = ferramentas.gerar_relatorio(
        leitura=leitura_normal,
        analise=analise,
        predicao=predicao,
        nivel_risco="NORMAL",
        acoes=[]
    )
    assert isinstance(relatorio, dict)
    assert "timestamp" in relatorio
    assert "nivel_risco" in relatorio
    assert "resumo" in relatorio


# ── Testes: AgenteManutencaoPreditiva ─────────────────────────────────

def test_agente_executar_retorna_dict(agente, leitura_normal):
    resultado = agente.executar(leitura_normal)
    assert isinstance(resultado, dict)


def test_agente_chaves_obrigatorias_no_resultado(agente, leitura_normal):
    resultado = agente.executar(leitura_normal)
    for chave in ["analise", "predicao", "nivel_risco", "acoes", "relatorio"]:
        assert chave in resultado, f"Chave ausente: {chave}"


def test_agente_leitura_normal_nivel_normal(agente, leitura_normal):
    resultado = agente.executar(leitura_normal)
    assert resultado["nivel_risco"] == "NORMAL"


def test_agente_leitura_normal_sem_acoes(agente, leitura_normal):
    resultado = agente.executar(leitura_normal)
    assert len(resultado["acoes"]) == 0


def test_agente_leitura_critica_nivel_critico(agente, leitura_critica):
    resultado = agente.executar(leitura_critica)
    assert resultado["nivel_risco"] == "CRITICO"


def test_agente_leitura_critica_gera_acoes(agente, leitura_critica):
    resultado = agente.executar(leitura_critica)
    assert len(resultado["acoes"]) > 0


def test_agente_leitura_critica_abre_ordem_manutencao(agente, leitura_critica):
    resultado = agente.executar(leitura_critica)
    acoes = resultado["acoes"]
    tipos = [a.get("tipo") if isinstance(a, dict) else "" for a in acoes]
    assert "ORDEM_MANUTENCAO" in tipos


def test_agente_leitura_alerta_nivel_alerta(agente, leitura_alerta):
    resultado = agente.executar(leitura_alerta)
    assert resultado["nivel_risco"] in ["ALERTA", "CRITICO"]
