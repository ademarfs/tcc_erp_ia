# Agente de Manutenção Preditiva — Torno CNC
## Integração de Inteligência Artificial ao ERP TOTVS Protheus

> **TCC — Engenharia de Software | UNINTER**  
> 🔗 [https://github.com/ademarfs/tcc_erp_ia](https://github.com/ademarfs/tcc_erp_ia)  
> Desenvolvido com metodologia TDD (*Test-Driven Development*)

---

## Sobre o Projeto

Protótipo de *software* funcional que implementa um **Agente de IA autônomo** para **manutenção preditiva** de tornos CNC, simulando a integração com o módulo SIGAMNT (Manutenção de Ativos) do ERP TOTVS Protheus via API REST.

A solução detecta anomalias em sensores industriais, prediz falhas com um modelo *Random Forest* e executa ações automatizadas (alertas e abertura de ordens de manutenção).

---

## Arquitetura

```
[Dados Simulados (Protheus)] → [Pipeline ML] → [Agente de IA] → [API REST]
       sensor_data.csv           Random Forest     Autônomo        FastAPI
```

**Camadas:**
- **Dados:** Simulação realista de sensores do torno CNC (temperatura, vibração, corrente, RPM, pressão, desgaste)
- **Modelo:** *Random Forest Classifier* (scikit-learn) — 100% de acurácia nos dados simulados (critério mínimo nos testes: 80%)
- **Agente:** Orquestrador autônomo com 5 ferramentas de decisão
- **API:** 4 endpoints REST (FastAPI + Swagger)

---

## Tecnologias

| Componente | Tecnologia |
|---|---|
| Linguagem | Python 3.11 |
| Machine Learning | scikit-learn |
| API REST | FastAPI + Uvicorn |
| Testes (TDD) | pytest |
| Dados | pandas + numpy |
| Serialização de modelo | joblib |

---

## Instalação

```bash
# 1. Clonar o repositório
git clone https://github.com/ademarfs/tcc_erp_ia.git
cd tcc_erp_ia

# 2. Instalar dependências
pip install -r requirements.txt
```

---

## Execução

### Rodar os testes (TDD)
```bash
python -m pytest tests/ -v
```
Resultado esperado: **51 testes — 100% PASSED**

### Iniciar a API
```bash
python src/api/main.py
```
Acesse a documentação interativa em: **http://localhost:8000/docs**

---

## Endpoints da API

| Método | Endpoint | Descrição |
|---|---|---|
| GET | `/status` | Saúde do sistema |
| POST | `/analisar` | Analisa leitura de sensores e retorna diagnóstico |
| POST | `/simular` | Gera leituras simuladas do torno CNC |
| GET | `/relatorio` | Histórico de análises do agente |

### Exemplo de uso

**Requisição:**
```json
POST /analisar
{
  "temperatura_fuso": 155.0,
  "vibracao_x": 6.8,
  "vibracao_y": 7.2,
  "vibracao_z": 5.5,
  "corrente_motor": 28.0,
  "rpm_fuso": 1500.0,
  "pressao_refrigerante": 1.5,
  "desgaste_ferramenta": 91.0
}
```

**Resposta:**
```json
{
  "nivel_risco": "CRITICO",
  "probabilidade_falha": 1.0,
  "diagnostico": "FALHA IMINENTE",
  "acoes_executadas": 2
}
```

---

## Estrutura do Projeto

```
tcc_manutencao_preditiva/
├── tests/                    ← 51 testes automatizados (TDD)
│   ├── test_simulator.py     ← 9 testes
│   ├── test_preprocessing.py ← 7 testes
│   ├── test_model.py         ← 8 testes
│   ├── test_agent.py         ← 15 testes
│   └── test_api.py           ← 12 testes
├── src/
│   ├── data/simulator.py     ← Gerador de dados do torno CNC
│   ├── preprocessing/        ← Normalização (StandardScaler)
│   ├── model/                ← Random Forest (treino + predição)
│   ├── agent/                ← Agente autônomo + ferramentas
│   └── api/main.py           ← FastAPI REST
├── conftest.py               ← Fixtures compartilhadas
├── requirements.txt
└── README.md
```

---

## Resultados

- ✅ **51/51 testes** aprovados (100%)
- ✅ **Métricas do modelo:** 100% de acurácia, precisão, recall e F1-score no conjunto de teste (1.000 leituras simuladas, divisão estratificada 80/20). Por serem dados sintéticos, com fronteiras bem definidas entre normal e falha, o desempenho tende a ser menor em dados reais.
- ✅ **Detecção correta** de cenários normais, de alerta e críticos
- ✅ **API funcional** com documentação Swagger automática

---

## Referência Acadêmica

Este repositório foi desenvolvido como parte do Trabalho de Conclusão de Curso em Engenharia de Software pelo Centro Universitário Internacional UNINTER (2026).

**Título:** Integração de Inteligência Artificial ao ERP TOTVS Protheus para Manutenção Preditiva  
**Autor:** Ademar Ferreira / RU: 2809102  
**Orientador:** Allan Christian Krainski Ferrari

---

## Licença

Uso acadêmico — UNINTER 2026
