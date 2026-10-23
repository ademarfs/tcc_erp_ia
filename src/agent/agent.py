"""
agent.py — Orquestrador autônomo do Agente de IA (Modo B: sem LLM).

O agente executa um ciclo de raciocínio determinístico:
  1. Analisar sensores (regras de limiar)
  2. Prever falha (modelo Random Forest)
  3. Calcular nível de risco (combinação de 1 e 2)
  4. Executar ações proporcionais ao risco
  5. Gerar relatório da decisão
"""
# pyrefly: ignore [missing-import]
from agent.tools import FerramentasAgente


class AgenteManutencaoPreditiva:
    """
    Agente autônomo de manutenção preditiva para torno CNC.

    Orquestra ferramentas de análise e age de forma proporcional
    ao nível de risco detectado, simulando a lógica que um agente
    inteligente executaria integrado ao ERP TOTVS Protheus.
    """

    NIVEIS_RISCO = ("NORMAL", "ALERTA", "CRITICO")

    def __init__(self, ferramentas: FerramentasAgente):
        self.ferramentas = ferramentas

    def executar(self, leitura: dict) -> dict:
        """
        Ciclo completo de raciocínio do agente para uma leitura de sensores.

        Args:
            leitura: Dicionário com valores de todos os sensores do torno CNC.

        Returns:
            Dicionário com análise, predição, nível de risco, ações e relatório.
        """
        # Passo 1 — Analisar sensores contra limiares
        analise = self.ferramentas.analisar_sensores(leitura)

        # Passo 2 — Predição probabilística via Random Forest
        predicao = self.ferramentas.prever_falha(leitura)

        # Passo 3 — Decisão autônoma: calcular nível de risco
        nivel_risco = self._calcular_nivel_risco(analise, predicao)

        # Passo 4 — Executar ações proporcionais ao risco
        acoes = self._executar_acoes(leitura, nivel_risco)

        # Passo 5 — Registrar e retornar relatório
        relatorio = self.ferramentas.gerar_relatorio(
            leitura=leitura,
            analise=analise,
            predicao=predicao,
            nivel_risco=nivel_risco,
            acoes=acoes,
        )

        return {
            "analise": analise,
            "predicao": predicao,
            "nivel_risco": nivel_risco,
            "acoes": acoes,
            "relatorio": relatorio,
        }

    # ── Lógica de decisão ──────────────────────────────────────────────
    def _calcular_nivel_risco(self, analise: dict, predicao: dict) -> str:
        """
        Combina análise de limiares com probabilidade do modelo ML
        para determinar o nível de risco de forma robusta.

        Regras:
          CRITICO  → ≥2 sensores críticos OU probabilidade de falha ≥ 0.75
          ALERTA   → 1 sensor crítico    OU probabilidade de falha ≥ 0.45
          NORMAL   → nenhuma das condições acima
        """
        n_criticos = analise["total_criticos"]
        prob_falha = predicao["probabilidade_falha"]

        if n_criticos >= 2 or prob_falha >= 0.75:
            return "CRITICO"
        elif n_criticos >= 1 or prob_falha >= 0.45:
            return "ALERTA"
        return "NORMAL"

    def _executar_acoes(self, leitura: dict, nivel_risco: str) -> list:
        """Executa ações proporcionais ao nível de risco detectado."""
        acoes = []

        if nivel_risco == "ALERTA":
            alerta = self.ferramentas.gerar_alerta(leitura, nivel="ALERTA")
            acoes.append({"tipo": "ALERTA", "mensagem": alerta})

        elif nivel_risco == "CRITICO":
            alerta = self.ferramentas.gerar_alerta(leitura, nivel="CRITICO")
            acoes.append({"tipo": "ALERTA", "mensagem": alerta})

            ordem = self.ferramentas.abrir_ordem_manutencao(leitura)
            acoes.append(ordem)

        return acoes
