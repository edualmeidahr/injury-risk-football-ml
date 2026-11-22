# Estimativa de Risco de Lesões em Jogadores de Futebol via ML

Projeto acadêmico (dupla) das disciplinas de **Ciência de Dados** (extração, ETL, análise
exploratória) e **Tópicos Especiais em Machine Learning** (modelagem preditiva).

A pergunta é se dados **públicos** de partidas (minutos jogados, calendário, histórico de
lesões) permitem estimar o risco de lesão de jogadores profissionais, sem GPS nem dados
internos de clube. O recorte é a Série A do Brasileirão de 2022 a 2025, com dados do
Transfermarkt. O alvo é lesão nos 14 dias seguintes a cada partida da liga em que o jogador
atuou. O produto é um *score* de risco relativo, sem valor de diagnóstico médico.

Proposta completa em `proposta_previsao_lesoes_futebol.md`. Artigo em `docs/artigo_ieee.tex`.

## Resultados principais

- **Base:** 47.704 partidas de 1.579 jogadores, com 2.367 casos positivos (4,96%) que vêm de
  1.212 lesões distintas.
- **Deriva de registro:** a taxa de lesões triplica a partir de 2024 (2,0–2,6% → 6,9–8,4%),
  em todas as faixas de gravidade e de forma desigual entre clubes. A causa provável é o
  Transfermarkt registrar lesões de forma mais completa, e não um aumento real.
- **Modelo:** com treino em 2024 e teste em 2025, a regressão logística obteve AUC-PR de
  0,140 (IC 95%: 0,119–0,169) para uma prevalência de 0,084 e AUC-ROC de 0,646. Os 10% de
  jogos de maior risco tiveram o dobro da taxa média de lesões.
- **De onde vem o sinal:** histórico de lesões (recência, gravidade, número) e calendário
  futuro do clube. A carga de minutos (ACWR, minutos em 7/28 dias, EWMA) acrescentou no
  máximo 0,006 de AUC-PR, com efeito significativo em 1 de 4 comparações.
- Modelos simples bastaram: a regressão logística empatou ou superou Random Forest e
  XGBoost; o MLP ficou por último.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

O `xgboost` e o `shap` são opcionais. Sem eles, o notebook de modelagem usa o
`HistGradientBoostingClassifier` do scikit-learn e pula a seção SHAP. Em rede instável, o
download do `xgboost` pode falhar por hash; nesse caso use
`pip install --no-cache-dir --timeout 300 --retries 10 xgboost shap`.

## Como reproduzir

```bash
source .venv/bin/activate

# 1. Coleta (lenta: respeita 2,5–5 s entre requisições; retomável com --retomar)
python scripts/scrape_league.py --temporadas 2022 2023 2024 2025
python scripts/scrape_squad.py --clubes-csv data/processed/clubs_serie_a.csv
python scripts/scrape_injuries.py --retomar --lote 50
python scripts/scrape_performance.py --retomar --lote 50

# 2. Limpeza e atributos -> data/processed/dataset_modelo.csv
python scripts/build_dataset.py --ano-max 2025

# 3. Análises: abrir e executar os notebooks
#    notebooks/01_eda.ipynb        (análise exploratória e auditoria)
#    notebooks/02_modelagem.ipynb  (~10 min)
```

O `dataset_modelo.csv` tem 47.872 linhas. A **amostra principal** exclui as 168 partidas em
que a lesão começou no próprio dia do jogo: filtre `lesao_no_dia == 0`. A coluna `y_com_dia`
é o alvo alternativo que inclui o dia do jogo, usado na análise de sensibilidade.

## Roadmap

| Fase | Descrição | Status |
|---|---|---|
| 0 | Teste manual de scraping (`test_scrape.py`) | concluída |
| 1 | Scraper do MVP: Cruzeiro (clube 609), 2022–2024 | concluída |
| 2 | Série A completa, 2022–2024; depois estendida a 2025 | concluída |
| 3 | Limpeza e engenharia de atributos (`scripts/features.py`) | concluída |
| 4 | Análise exploratória e auditoria (`notebooks/01_eda.ipynb`) | concluída |
| 5 | Modelagem preditiva (`notebooks/02_modelagem.ipynb`) | concluída |
| 6 | Artigo IEEE (`docs/artigo_ieee.tex`) | em revisão |

## Estrutura

```
data/raw/                     HTML/JSON bruto cacheado (não versionado)
data/processed/               CSVs limpos (não versionado)
scripts/tm_client.py          HTTP compartilhado: cache, retry, rate limiting
scripts/scrape_league.py      clubes da Série A por temporada
scripts/scrape_squad.py       elenco de um clube por temporada
scripts/scrape_injuries.py    histórico de lesões por jogador
scripts/scrape_performance.py desempenho por partida (via API interna)
scripts/features.py           taxonomia, carga, calendário, histórico, alvo
scripts/build_dataset.py      orquestra limpeza -> dataset_modelo.csv
scripts/modelagem.py          atributos, modelos, métricas e bootstrap da Fase 5
scripts/figuras_artigo.py     gera as figuras do artigo (rótulos em português)
notebooks/01_eda.ipynb        análise exploratória (6 tópicos + síntese)
notebooks/02_modelagem.ipynb  baselines, modelos, ablação, robustez, interpretação
notebooks/figuras/            gráficos gerados pelos notebooks
docs/                         pré-projeto e artigo IEEE (LaTeX)
```

## Detalhes da fonte

- **`saison_id`:** o Transfermarkt usa a convenção europeia de temporada (ano de início),
  então o Brasileirão de 2024 é `saison_id=2023`. Os scripts recebem o ano civil e convertem
  (`tm_client.temporada_para_saison_id`).
- **Lesões:** a aba `/verletzungen/` é HTML estático, com paginação.
- **Desempenho:** a aba `/leistungsdaten/` é renderizada por JavaScript. Os dados vêm da API
  interna `/ceapi/performance-game/<player_id>`, com uma linha por partida (data,
  competição, minutos, titularidade, idade, estado de participação).
- O bruto fica em cache em `data/raw/`; rodar de novo não refaz requisições (use
  `--no-cache` para forçar).

## Observações éticas

Extração apenas para fins acadêmicos, sem redistribuição dos dados brutos, com *rate
limiting* entre requisições.
