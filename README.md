# Estimativa de Risco de Lesões em Jogadores de Futebol via ML

Projeto acadêmico (dupla) que atende às disciplinas de **Ciência de Dados** (extração, ETL,
análise exploratória) e **Tópicos Especiais em Machine Learning** (modelagem preditiva).

Objetivo: estimar o risco de lesão de jogadores profissionais a partir de indicadores de
carga de trabalho derivados de dados **públicos** de partidas (minutos jogados,
congestionamento de calendário, histórico de lesões), sem acesso a dados proprietários de
GPS/fisiologia. Ver `proposta_previsao_lesoes_futebol.md` para a proposta completa.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Fase 2 — Série A completa (2022-2024)

```bash
source .venv/bin/activate
python scripts/scrape_league.py --temporadas 2022 2023 2024
python scripts/scrape_squad.py --clubes-csv data/processed/clubs_serie_a.csv
python scripts/scrape_injuries.py --retomar --lote 50
python scripts/scrape_performance.py --retomar --lote 50
```

`--retomar` pula jogadores já no checkpoint (`.done.txt`) e salva a cada 50.
Se interromper, rode de novo com `--retomar`.

## Fase 1 — pipeline do MVP (Cruzeiro, 2022-2024)

```bash
source .venv/bin/activate
python scripts/scrape_squad.py --clube 609 --temporadas 2022 2023 2024
python scripts/scrape_injuries.py
python scripts/scrape_performance.py
```

Saídas em `data/processed/`: `squad_raw.csv`, `injuries_raw.csv`, `performance_raw.csv`.

> **Pegadinha do `saison_id`**: o Transfermarkt usa a convenção europeia de temporada
> (ano de início), então o Brasileirão de 2024 é `saison_id=2023`. Os scripts recebem o
> **ano civil** e fazem a conversão internamente
> (`tm_client.temporada_para_saison_id`).

## Fase 0 (teste manual de scraping)

Validação dos seletores em uma única página de jogador, antes de construir o pipeline.

```bash
source .venv/bin/activate
python test_scrape.py "https://www.transfermarkt.com.br/<slug>/profil/spieler/<id>"
```

O bruto é cacheado em `data/raw/`, então re-executar o script não gera novas
requisições ao Transfermarkt (use `--no-cache` para forçar o download).

Duas fontes distintas por jogador:

- **Lesões**: a aba `/verletzungen/` ainda é HTML estático, parseada com BeautifulSoup
  (inclui paginação para jogadores com histórico longo).
- **Desempenho**: a aba `/leistungsdaten/` passou a ser renderizada por JavaScript e não
  tem mais tabela no HTML. Os dados vêm da API interna
  `/ceapi/performance-game/<player_id>`, que retorna **uma linha por partida** (data,
  competição, minutos jogados, se estava lesionado, idade na data) — granularidade melhor
  que a tabela agregada por temporada que o site exibe.

## Roadmap

| Fase | Descrição | Status |
|---|---|---|
| 0 | Teste manual de scraping (`test_scrape.py`) | concluída |
| 1 | Scraper do MVP: Cruzeiro (clube 609), temporadas 2022-2024 | concluída |
| 2 | Escalar para toda a Série A 2022-2024 | em validação |
| 2 | Escalar para toda a Série A 2024 (~900 jogadores) | pendente |
| 3 | Limpeza e engenharia de features (`scripts/features.py`) | pendente |
| 4 | Análise exploratória (`notebooks/01_eda.ipynb`) | pendente |
| 5 | Modelagem preditiva (`notebooks/02_modelagem.ipynb`) | pendente |

## Estrutura

```
data/raw/                     HTML/JSON bruto cacheado (não versionado)
data/processed/               CSVs limpos (não versionado)
scripts/tm_client.py          HTTP compartilhado: cache, retry, rate limiting
scripts/scrape_squad.py       elenco de um clube por temporada
scripts/scrape_injuries.py    histórico de lesões por jogador
scripts/scrape_performance.py desempenho por partida (via API interna)
notebooks/                    entregáveis de EDA e modelagem
```

## Observações éticas

Extração apenas para fins acadêmicos, sem redistribuição dos dados brutos, com
*rate limiting* entre requisições.
