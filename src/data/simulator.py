"""
simulator.py — Gerador de dados simulados de sensores do torno CNC.

Simula a exportação de dados do módulo de Manutenção do ERP TOTVS Protheus,
gerando leituras realistas de sensores industriais com distribuições baseadas
em parâmetros reais de tornos CNC do setor metalúrgico.
"""
import numpy as np
import pandas as pd


class SimuladorTornoCNC:
    """
    Gera datasets sintéticos de leituras de sensores de um torno CNC,
    simulando a exportação de dados do ERP TOTVS Protheus.

    Parâmetros dos sensores:
    - temperatura_fuso   : °C  | Normal: 60–120  | Crítico: >140
    - vibracao_x/y/z     : mm/s| Normal: 0–2     | Crítico: >5
    - corrente_motor     : A   | Normal: 10–20   | Crítico: >25
    - rpm_fuso           : RPM | Normal: 500–3000
    - pressao_refrigerante: bar | Normal: 3–6     | Crítico: <2
    - desgaste_ferramenta: %   | Normal: 0–70    | Crítico: >85
    - label              : 0=normal, 1=falha iminente
    """

    def __init__(self, seed: int = 42):
        self.seed = seed
        self.rng = np.random.default_rng(seed)

    def gerar(self, n_amostras: int = 500, proporcao_falha: float = 0.25) -> pd.DataFrame:
        """
        Gera n_amostras de leituras de sensores.

        Args:
            n_amostras: Número total de amostras a gerar.
            proporcao_falha: Proporção de amostras com falha iminente (padrão 25%).

        Returns:
            DataFrame com todas as leituras e o label correspondente.
        """
        n_falha = int(n_amostras * proporcao_falha)
        n_normal = n_amostras - n_falha

        normais = self._gerar_normais(n_normal)
        falhas = self._gerar_falhas(n_falha)

        df = pd.concat([normais, falhas], ignore_index=True)
        df = df.sample(frac=1, random_state=self.seed).reset_index(drop=True)
        return df

    def gerar_e_salvar(self, n_amostras: int, caminho: str,
                       proporcao_falha: float = 0.25) -> pd.DataFrame:
        """Gera os dados e salva em CSV."""
        df = self.gerar(n_amostras=n_amostras, proporcao_falha=proporcao_falha)
        df.to_csv(caminho, index=False)
        return df

    def _gerar_normais(self, n: int) -> pd.DataFrame:
        return pd.DataFrame({
            "temperatura_fuso":      self.rng.uniform(60.0,  120.0, n),
            "vibracao_x":            self.rng.uniform(0.1,   2.0,   n),
            "vibracao_y":            self.rng.uniform(0.1,   2.0,   n),
            "vibracao_z":            self.rng.uniform(0.1,   2.0,   n),
            "corrente_motor":        self.rng.uniform(10.0,  20.0,  n),
            "rpm_fuso":              self.rng.uniform(500.0, 3000.0, n),
            "pressao_refrigerante":  self.rng.uniform(3.0,   6.0,   n),
            "desgaste_ferramenta":   self.rng.uniform(0.0,   70.0,  n),
            "label": 0,
        })

    def _gerar_falhas(self, n: int) -> pd.DataFrame:
        return pd.DataFrame({
            "temperatura_fuso":      self.rng.uniform(140.0, 185.0, n),
            "vibracao_x":            self.rng.uniform(5.1,   10.0,  n),
            "vibracao_y":            self.rng.uniform(5.1,   10.0,  n),
            "vibracao_z":            self.rng.uniform(5.1,   10.0,  n),
            "corrente_motor":        self.rng.uniform(25.1,  35.0,  n),
            "rpm_fuso":              self.rng.uniform(500.0, 3000.0, n),
            "pressao_refrigerante":  self.rng.uniform(0.5,   1.9,   n),
            "desgaste_ferramenta":   self.rng.uniform(85.1,  100.0, n),
            "label": 1,
        })
