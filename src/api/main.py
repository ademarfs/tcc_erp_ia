"""
main.py — API REST FastAPI para o Agente de Manutenção Preditiva.

Simula os endpoints que seriam expostos para integração
com o ERP TOTVS Protheus via API REST.
"""
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from data.simulator import SimuladorTornoCNC
from preprocessing.pipeline import PipelinePreprocessamento
from model.trainer import TreinadorModelo
from model.predictor import PreditorFalha
from agent.tools import FerramentasAgente
from agent.agent import AgenteManutencaoPreditiva


# ── Schemas de entrada/saída ──────────────────────────────────────────
class LeituraSensores(BaseModel):
    temperatura_fuso: float
    vibracao_x: float
    vibracao_y: float
    vibracao_z: float
    corrente_motor: float
    rpm_fuso: float
    pressao_refrigerante: float
    desgaste_ferramenta: float


class SimularRequest(BaseModel):
    n_amostras: int = 10
    seed: int = 42


# ── Factory: inicializa e treina o agente na startup ──────────────────
def _inicializar_agente() -> AgenteManutencaoPreditiva:
    sim = SimuladorTornoCNC(seed=42)
    df = sim.gerar(n_amostras=1000)

    pipeline = PipelinePreprocessamento()
    X, y = pipeline.fit_transform(df)

    treinador = TreinadorModelo(seed=42)
    treinador.treinar(X, y)

    preditor = PreditorFalha(modelo=treinador.modelo, pipeline=pipeline)
    ferramentas = FerramentasAgente(preditor=preditor)
    return AgenteManutencaoPreditiva(ferramentas=ferramentas)


def criar_app() -> FastAPI:
    app = FastAPI(
        title="Agente de Manutenção Preditiva — Torno CNC",
        description=(
            "API REST para integração do Agente de IA com o ERP TOTVS Protheus. "
            "Recebe leituras de sensores e retorna diagnóstico preditivo de falhas."
        ),
        version="1.0.0",
    )

    agente = _inicializar_agente()

    @app.get("/status")
    def status():
        """Verifica se o sistema está operacional."""
        return {
            "status": "operacional",
            "sistema": "Agente de Manutenção Preditiva — Torno CNC",
            "versao": "1.0.0",
        }

    @app.post("/analisar")
    def analisar(leitura: LeituraSensores):
        """
        Analisa uma leitura de sensores e retorna o diagnóstico do agente.
        Simula o endpoint consumido pelo módulo MNT do TOTVS Protheus.
        """
        resultado = agente.executar(leitura.model_dump())
        return {
            "nivel_risco": resultado["nivel_risco"],
            "probabilidade_falha": resultado["predicao"]["probabilidade_falha"],
            "diagnostico": resultado["predicao"]["diagnostico"],
            "sensores_criticos": resultado["analise"]["sensores_criticos"],
            "acoes_executadas": len(resultado["acoes"]),
            "relatorio": resultado["relatorio"],
        }

    @app.post("/simular")
    def simular(req: SimularRequest):
        """Gera leituras simuladas de sensores do torno CNC."""
        sim = SimuladorTornoCNC(seed=req.seed)
        df = sim.gerar(n_amostras=req.n_amostras)
        leituras = df.drop(columns=["label"]).to_dict(orient="records")
        return {"leituras": leituras, "total": len(leituras)}

    @app.get("/relatorio")
    def relatorio():
        """Retorna o histórico de análises realizadas pelo agente."""
        return {
            "historico": agente.ferramentas.historico,
            "total_analises": len(agente.ferramentas.historico),
        }

    return app


if __name__ == "__main__":
    import uvicorn
    app = criar_app()
    uvicorn.run(app, host="0.0.0.0", port=8000)
