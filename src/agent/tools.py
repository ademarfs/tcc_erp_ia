"""
tools.py — Ferramentas do Agente de IA para manutenção preditiva.

Cada ferramenta encapsula uma ação atômica que o agente pode executar.
O agente orquestra essas ferramentas de forma autônoma com base
no nível de risco calculado a partir das leituras do torno CNC.
"""
import uuid
from datetime import datetime
from model.predictor import PreditorFalha

# Limiares de criticidade por sensor
LIMIARES = {
    "temperatura_fuso":     {"critico": 140.0,  "direcao": "acima"},
    "vibracao_x":           {"critico": 5.0,    "direcao": "acima"},
    "vibracao_y":           {"critico": 5.0,    "direcao": "acima"},
    "vibracao_z":           {"critico": 5.0,    "direcao": "acima"},
    "corrente_motor":       {"critico": 25.0,   "direcao": "acima"},
    "pressao_refrigerante": {"critico": 2.0,    "direcao": "abaixo"},
    "desgaste_ferramenta":  {"critico": 85.0,   "direcao": "acima"},
}


class FerramentasAgente:
    """Conjunto de ferramentas disponíveis para o Agente de IA."""

    def __init__(self, preditor: PreditorFalha):
        self.preditor = preditor
        self._historico: list = []

    # ── Ferramenta 1: Análise de Sensores ─────────────────────────────
    def analisar_sensores(self, leitura: dict) -> dict:
        """
        Verifica cada sensor contra os limiares de criticidade.
        Retorna lista de sensores fora do padrão e seus valores.
        """
        sensores_criticos = []
        for sensor, limiar in LIMIARES.items():
            valor = leitura.get(sensor)
            if valor is None:
                continue
            if limiar["direcao"] == "acima" and valor > limiar["critico"]:
                sensores_criticos.append({
                    "sensor": sensor,
                    "valor": valor,
                    "limiar": limiar["critico"],
                    "situacao": "ACIMA DO LIMITE",
                })
            elif limiar["direcao"] == "abaixo" and valor < limiar["critico"]:
                sensores_criticos.append({
                    "sensor": sensor,
                    "valor": valor,
                    "limiar": limiar["critico"],
                    "situacao": "ABAIXO DO LIMITE",
                })
        return {
            "sensores_criticos": sensores_criticos,
            "total_criticos": len(sensores_criticos),
        }

    # ── Ferramenta 2: Predição de Falha (ML) ──────────────────────────
    def prever_falha(self, leitura: dict) -> dict:
        """Executa o modelo Random Forest e retorna a predição."""
        return self.preditor.prever(leitura)

    # ── Ferramenta 3: Gerar Alerta ────────────────────────────────────
    def gerar_alerta(self, leitura: dict, nivel: str) -> str:
        """
        Gera mensagem de alerta textual para os operadores,
        simulando a notificação que seria gerada no TOTVS Protheus.
        """
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        criticos = self.analisar_sensores(leitura)["sensores_criticos"]
        sensores_str = ", ".join(
            [f"{s['sensor']} ({s['valor']:.1f})" for s in criticos]
        ) or "nenhum"
        return (
            f"[{ts}] ALERTA {nivel} — Torno CNC | "
            f"Sensores fora do padrão: {sensores_str}. "
            f"Verificação imediata recomendada."
        )

    # ── Ferramenta 4: Abrir Ordem de Manutenção ───────────────────────
    def abrir_ordem_manutencao(self, leitura: dict) -> dict:
        """
        Cria uma Ordem de Manutenção corretiva, simulando a criação
        automática de OS no módulo MNT do ERP TOTVS Protheus.
        """
        numero = f"OS-{uuid.uuid4().hex[:8].upper()}"
        criticos = self.analisar_sensores(leitura)["sensores_criticos"]
        return {
            "tipo": "ORDEM_MANUTENCAO",
            "numero_ordem": numero,
            "equipamento": "Torno CNC — Linha de Produção",
            "prioridade": "ALTA",
            "status": "ABERTA",
            "descricao": (
                f"Manutenção preditiva detectada por Agente de IA. "
                f"Sensores críticos: {[s['sensor'] for s in criticos]}"
            ),
            "timestamp": datetime.now().isoformat(),
        }

    # ── Ferramenta 5: Gerar Relatório ─────────────────────────────────
    def gerar_relatorio(self, leitura: dict, analise: dict,
                        predicao: dict, nivel_risco: str,
                        acoes: list) -> dict:
        """Registra a decisão do agente no histórico e retorna o relatório."""
        relatorio = {
            "timestamp": datetime.now().isoformat(),
            "nivel_risco": nivel_risco,
            "probabilidade_falha": predicao.get("probabilidade_falha"),
            "sensores_criticos": analise.get("sensores_criticos", []),
            "acoes_executadas": len(acoes),
            "resumo": (
                f"Nível de risco: {nivel_risco}. "
                f"Probabilidade de falha: {predicao.get('probabilidade_falha', 0):.1%}. "
                f"Ações executadas: {len(acoes)}."
            ),
        }
        self._historico.append(relatorio)
        return relatorio

    @property
    def historico(self) -> list:
        return self._historico
