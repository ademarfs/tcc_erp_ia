"""
FASE RED — test_api.py
Testes de integração para os endpoints FastAPI.
"""
import pytest
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from fastapi.testclient import TestClient


@pytest.fixture
def client():
    from api.main import criar_app
    app = criar_app()
    return TestClient(app)


# ── GET /status ────────────────────────────────────────────────────────

def test_status_retorna_200(client):
    response = client.get("/status")
    assert response.status_code == 200


def test_status_retorna_sistema_operacional(client):
    response = client.get("/status")
    data = response.json()
    assert "status" in data
    assert data["status"] == "operacional"


# ── POST /analisar ─────────────────────────────────────────────────────

def test_analisar_retorna_200(client, leitura_normal):
    response = client.post("/analisar", json=leitura_normal)
    assert response.status_code == 200


def test_analisar_retorna_nivel_risco(client, leitura_normal):
    response = client.post("/analisar", json=leitura_normal)
    data = response.json()
    assert "nivel_risco" in data
    assert data["nivel_risco"] in ["NORMAL", "ALERTA", "CRITICO"]


def test_analisar_leitura_normal_retorna_normal(client, leitura_normal):
    response = client.post("/analisar", json=leitura_normal)
    data = response.json()
    assert data["nivel_risco"] == "NORMAL"


def test_analisar_leitura_critica_retorna_critico(client, leitura_critica):
    response = client.post("/analisar", json=leitura_critica)
    data = response.json()
    assert data["nivel_risco"] == "CRITICO"


def test_analisar_retorna_probabilidade(client, leitura_normal):
    response = client.post("/analisar", json=leitura_normal)
    data = response.json()
    assert "probabilidade_falha" in data
    prob = data["probabilidade_falha"]
    assert 0.0 <= prob <= 1.0


def test_analisar_dados_invalidos_retorna_422(client):
    """Dados incompletos devem retornar erro de validação."""
    response = client.post("/analisar", json={"temperatura_fuso": 85.0})
    assert response.status_code == 422


# ── POST /simular ──────────────────────────────────────────────────────

def test_simular_retorna_200(client):
    response = client.post("/simular", json={"n_amostras": 10, "seed": 42})
    assert response.status_code == 200


def test_simular_retorna_lista_de_leituras(client):
    response = client.post("/simular", json={"n_amostras": 5, "seed": 42})
    data = response.json()
    assert "leituras" in data
    assert len(data["leituras"]) == 5


# ── GET /relatorio ─────────────────────────────────────────────────────

def test_relatorio_retorna_200(client):
    response = client.get("/relatorio")
    assert response.status_code == 200


def test_relatorio_retorna_lista(client):
    response = client.get("/relatorio")
    data = response.json()
    assert "historico" in data
    assert isinstance(data["historico"], list)
