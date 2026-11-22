# Handoff — Estimativa de Risco de Lesão (Futebol / ML)

Documento para retomar o projeto em outra sessão (ex.: Claude Code).  
**Última atualização:** 2026-11-15 (EDA, modelagem e artigo concluídos)  
**Autores:** Eduardo Henrique Queiroz Almeida · Sergio Henrique Quedas Ramos  
**Repo:** https://github.com/edualmeidahr/injury-risk-football-ml  
**Workspace local:** `/home/eduardo/Documentos/8 Periodo/Estimativa_risco_lesao`

---

## 1. O que é o projeto

Trabalho acadêmico (dupla) que atende:

- **Ciência de Dados** — extração, ETL, EDA  
- **Tópicos Especiais em ML** — modelagem preditiva  

**Pergunta:** dá para estimar risco de lesão só com dados **públicos** de partidas (minutos, calendário, histórico de lesões), sem GPS/fisiologia?

**Recorte:** Brasileirão Série A, temporadas civis **2022–2025** (2025 incluído em 2026-11-15), fonte **Transfermarkt**.

**Alvo:** \(y=1\) se o jogador sofre lesão (não doença/gestão de esforço) nos **próximos 14 dias** após uma partida BRA1 em que atuou.

**Produto:** *score* de risco relativo — não diagnóstico médico.

Proposta longa: `proposta_previsao_lesoes_futebol.md`  
Pré-projeto LaTeX: `docs/pre_projeto.tex` / `docs/pre_projeto.pdf`  
Artigo IEEE: `docs/artigo_ieee.tex` / `docs/artigo_ieee.pdf` (7 páginas, linguagem simplificada, 6 figuras)

---

## 2. Status por fase

| Fase | Conteúdo | Status |
|------|----------|--------|
| 0 | Teste scraping 1 jogador (`test_scrape.py`) | **Concluída** |
| 1 | MVP Cruzeiro (clube 609), 2022–2024 | **Concluída** |
| 2 | Toda Série A 2022–2024 (~1,9k jogadores) | **Concluída** |
| 3 | Limpeza + features + `dataset_modelo.csv` | **Concluída** |
| 2b | Série A 2025 (+407 jogadores) | **Concluída** (backup 2022–24 em `data/processed/backup_2022_2024/`) |
| 4 | EDA `notebooks/01_eda.ipynb` | **Concluída** — 6 tópicos + síntese final |
| 5 | Modelagem `notebooks/02_modelagem.ipynb` | **Concluída** — 3 cortes, 4 modelos, ablação, robustez, SHAP |
| 6 | Artigo IEEE `docs/artigo_ieee.tex` | **Em revisão** — ver pendências na seção 9 |

### EDA — roteiro (concluído)

1. Panorama da base · 2. Perfil das lesões · 3. Calendário · 4. **Auditoria do alvo e deriva de registro** · 5. Carga × lesão · 6. Comparativos clube/posição · Síntese  

Padrão do notebook: **título → código/gráfico → célula “Leitura”** (não inverter).

---

## 3. Setup

```bash
cd "/home/eduardo/Documentos/8 Periodo/Estimativa_risco_lesao"
python3 -m venv .venv          # se ainda não existir
source .venv/bin/activate
pip install -r requirements.txt
```

`requirements.txt`: `requests`, `beautifulsoup4`, `lxml`, `pandas`, `matplotlib`, `seaborn`, `scikit-learn`, `scipy`.  
Opcionais (instalados no `.venv` local): `xgboost` 3.2 e `shap` 0.49. Sem eles, `scripts/modelagem.py` usa `HistGradientBoostingClassifier` e o notebook pula o SHAP.  
O pip falhou várias vezes com **hash incorreto** (download corrompido pela rede). Funcionou com:
`pip install --no-cache-dir --timeout 300 --retries 10 xgboost shap`.  
Para executar notebooks pela linha de comando: `nbclient` + `nbformat` (já no `.venv`).

---

## 4. Estrutura do repositório

```
Estimativa_risco_lesao/
├── data/
│   ├── raw/              # HTML/JSON cacheados (NÃO versionar)
│   └── processed/        # CSVs (NÃO versionar — .gitignore)
├── scripts/
│   ├── tm_client.py      # HTTP: UA, cache, retry, rate limit, saison_id
│   ├── scrape_league.py  # clubes Série A por temporada
│   ├── scrape_squad.py   # elenco clube × temporada
│   ├── scrape_injuries.py
│   ├── scrape_performance.py
│   ├── features.py       # taxonomia, carga, calendário, histórico, alvo
│   ├── build_dataset.py  # orquestra limpeza → dataset_modelo.csv
│   ├── modelagem.py      # atributos, grupos, modelos, métricas, bootstrap (Fase 5)
│   └── figuras_artigo.py # figuras do artigo (python scripts/figuras_artigo.py)
├── notebooks/
│   ├── 01_eda.ipynb      # EDA (6 tópicos + síntese)
│   ├── 02_modelagem.ipynb  # modelagem (~10 min para rodar)
│   └── figuras/          # PNGs dos dois notebooks (usados no artigo)
├── docs/                 # pré-projeto, artigo IEEE (figuras vêm de ../notebooks/figuras)
├── test_scrape.py        # Fase 0
├── HANDOFF.md            # este arquivo
├── README.md
├── proposta_previsao_lesoes_futebol.md
└── requirements.txt
```

---

## 5. Dados processados (volumes ~)

| Arquivo | Conteúdo aproximado |
|---------|---------------------|
| `clubs_serie_a.csv` | 80 linhas (20 clubes × 4 anos) |
| `squad_raw.csv` | 4.476 linhas · 2.347 jogadores · 28 clubes |
| `injuries_raw.csv` | 6.255 episódios |
| `injuries_clean.csv` | tipado + `categoria_lesao` + `conta_como_alvo` |
| `performance_raw.csv` | ~988k partidas (~109 MB) |
| `dataset_modelo.csv` | **47.704** obs. BRA1 jogadas (2022–25) · **1.579** jogadores · `y=1` = **4,96%** · **1.212 episódios** únicos |
| `*.done.txt` | checkpoints de scraping (player_ids já processados) |

### `dataset_modelo.csv` — colunas

```
player_id, nome, game_id, data, ano, season_id, competicao, clube_id, nome_clube,
adversario_id, local, minutos_jogados, titular, idade, posicao, posicao_grupo,
minutos_7d, minutos_cronica_4sem, acwr, acwr_desacoplada, ewma_aguda, ewma_cronica, acwr_ewma, dias_desde_ultima_partida, jogos_30d,
jogos_clube_prox_14d, n_lesoes_previas, n_lesoes_12m, dias_desde_ultima_lesao, dias_desde_retorno,
jogos_desde_retorno, dias_fora_ultima_lesao, categoria_ultima_lesao, lesao_no_dia, y, y_com_dia
```

### Definições importantes do dataset

- **Observação:** `participacao=played`, competição `BRA1`, minutos > 0, anos 2022–2025. O arquivo tem 47.872 linhas; a **amostra principal** filtra `lesao_no_dia == 0` (47.704 linhas) — sempre aplicar esse filtro.  
- **Alvo alternativo `y_com_dia`:** janela [d, d+14], inclui lesão no dia do jogo (sensibilidade; usar com todas as linhas).  
- **Retorno de lesão:** retorno = 1ª partida jogada após o início da lesão (não usa `data_fim`). `dias_fora_ultima_lesao` = retorno − início.  
- **`jogos_clube_prox_14d`:** jogos de clube agendados em (d, d+14] (calendário público; exclui seleção e `played for a different team`). Checado: 99,4% igual ao resto do elenco no mesmo jogo → sem vazamento individual.  
- **Features de carga:** usam **todo** o histórico de partidas do jogador (todas as competições), mas **só datas anteriores** à partida atual (sem leakage).  
- **ACWR** = `minutos_7d` / (`minutos` nos últimos 28 dias / 4). `NaN` se crônica = 0 (~6% das linhas).  
- **Alvo `y`:** lesão com `conta_como_alvo=True` e `data_inicio` em **(data_partida, data_partida+14]** (exclui o dia do jogo).  
- **Não entram no alvo:** COVID, gripe, gestão de esforço, recuperação de ritmo, etc. (`categoria_lesao=nao_traumatica`).

### Taxonomia (`categoria_lesao`)

`coxa`, `joelho`, `tornozelo`, `panturrilha`, `adutores`, `ombro`, `pe`, `perna`, `quadril`, `muscular_geral`, `cabeca`, `coluna`, `mao_braco`, `outras`, `nao_traumatica`.

---

## 6. Pegadinhas técnicas (não esquecer)

1. **`saison_id` Transfermarkt = ano civil − 1** no Brasileirão  
   (Brasileirão 2024 → `saison_id=2023`). Função: `tm_client.temporada_para_saison_id`.

2. **Aba de desempenho não é HTML estático** — usar  
   `https://www.transfermarkt.com.br/ceapi/performance-game/<player_id>` (JSON por partida).

3. **Lesões** ainda são HTML (`/verletzungen/`), com paginação.

4. **Rate limit:** delays 2,5–5 s; em HTTP 403/429 o client pausa 30–100 s. Scrapers aceitam `--retomar`.

5. **Deriva de registro:** `y=1` = 2,6 / 1,95 / 6,9 / 8,4% (2022–25). Sobe em todas as gravidades, desigual por clube (1,1×–13×) e contamina o histórico (“nunca lesionado” 54% → 19%). Ver EDA tópico 4.

6. **Desbalanceamento:** ~95% / 5% → **nunca** usar acurácia. Métricas usadas: AUC-PR (com *lift* sobre a prevalência), precisão/recall/lift nos 10% de maior risco, AUC-ROC, Brier. Pesos de classe não melhoram a ordenação e pioram a calibração.

7. **Não commitar** `data/raw/` nem `data/processed/` (já no `.gitignore`).

8. **Posição 2025:** o TM passou a grafar `Ponta-Direita`/`Ponta-Esquerda` (já mapeado em `MAPA_POSICAO`).

9. **Top-k com empates:** `modelagem.top_k` usa valor esperado para scores empatados na fronteira; sem isso, regras binárias e o Dummy dependiam da ordem das linhas (data).

10. **SHAP + xgboost 3.x:** o `TreeExplainer` do shap 0.49 não lê o `base_score` do xgboost 3.2. O notebook usa `booster.predict(DMatrix, pred_contribs=True)` (TreeSHAP nativo).

---

## 7. Como regenerar / retomar scraping

```bash
source .venv/bin/activate

# Lista clubes
python scripts/scrape_league.py --temporadas 2022 2023 2024 2025

# Elencos
python scripts/scrape_squad.py --clubes-csv data/processed/clubs_serie_a.csv

# Lesões / desempenho (retomável)
python scripts/scrape_injuries.py --retomar --lote 50
python scripts/scrape_performance.py --retomar --lote 50

# Dataset de modelagem
python scripts/build_dataset.py --ano-max 2025
# opcional: --ano-min 2022 --n-alvo 14
```

Teste unitário de 1 jogador:

```bash
python test_scrape.py "https://www.transfermarkt.com.br/<slug>/profil/spieler/<id>" --limite 10
```

---

## 8. Achados principais da EDA (2022–2025)

Resumo completo na célula **Síntese** no fim do `01_eda.ipynb`.
- **Tamanho efetivo:** 1.212 episódios (150/121/416/525 por ano); treino 2022–23 = 271.
- **Alvo:** 55% das lesões começam no dia do jogo ou no seguinte; dia 0 excluído e dia +1 vira `y=1` → inconsistente.
- **Histórico:** sinal mais forte e estável (AUC 0,58–0,61/ano; lesão ≤30d → risco 2×).
- **Carga:** fraca e instável (AUC 0,51–0,57; sentido inverte em 2022). EWMA é a única ACWR com gradiente; ACWR acoplada ≡ desacoplada em ordenação.
- **Clube:** varia muito, mas sem relação com carga e ranking instável entre anos → registro. Fora do modelo.
- **Posição:** goleiro ~0,6× do risco; zagueiro 1,2× (exposição).

---

## 9. Resultados da modelagem e pendências

### Resultados (`02_modelagem.ipynb`, síntese no fim do notebook)
Cortes: **A** 2022–23 → 2024 · **B** 2024 → 2025 (principal) · **C** 2022–24 → 2025.

| Corte B | AP (IC 95%) | AUC | Lift top 10% |
|---|---|---|---|
| Regressão logística | **0,140** (0,119–0,169) | 0,646 | 2,03 |
| XGBoost | 0,127 (0,109–0,152) | 0,633 | 1,79 |
| Random Forest | 0,126 (0,110–0,148) | 0,639 | 1,76 |
| MLP (10 sementes) | 0,098 (0,087–0,114) | 0,557 | 1,30 |
| Regra recência da lesão | 0,117 | 0,604 | 1,45 |
| Prevalência (acaso) | 0,084 | 0,5 | 1,0 |

- **Critério 1 (modelo útil): atendido.**
- **Critério 2 (carga acrescenta): atendido formalmente, mas frágil** — só XGBoost no corte B tem ΔAP > 0 (+0,0057 [+0,0006; +0,0119]); some no corte C e na regressão logística. Redação adotada: "sinal marginal e instável".
- Sinal vem de **histórico** (permutação: 19–31% da AP) e **calendário** (15–17%); carga 1–5%. No SHAP, `jogos_clube_prox_14d` é o atributo mais usado.
- Robustez: 2022→2023 ainda acima do acaso; alvo `y_com_dia` igual; leave-clubs-out igual; jogadores novos iguais; pesos/undersampling não mudam AP.

### Pendências
1. **Referências novas no artigo** (escritas de memória, conferir volume/páginas/DOI): Williams et al. 2017 (BJSM, EWMA) e van den Goorbergh et al. 2022 (JAMIA, desbalanceamento).
2. ~~Autoria do artigo~~: Sergio incluído como coautor (2026-11-15).
3. `docs/` ainda tem o PDF antigo com nome longo; o atual é `artigo_ieee.pdf`.
4. **Commit** de toda a etapa (nada commitado desde o primeiro commit).
5. Opcional: slides da apresentação; pré-projeto (`docs/pre_projeto.tex`) ainda descreve o plano antigo (2022–24, limiar por F1).

---

## 10. Convenções do assistente / equipe

- Trabalhar **uma fase/tópico por vez**; mostrar resultado e esperar OK antes de avançar (combinado no início do scraping).  
- Código comentado em **português**.  
- Commits só quando o usuário pedir.  
- Scraping: sempre rate limiting; uso acadêmico, sem redistribuir bruto.

---

## 11. Comandos úteis de sanidade

```bash
source .venv/bin/activate
python - <<'EOF'
import pandas as pd
ds = pd.read_csv("data/processed/dataset_modelo.csv", parse_dates=["data"])
ds = ds[ds.lesao_no_dia.eq(0)]          # amostra principal
print(ds.shape, ds.player_id.nunique())
print(ds.y.value_counts(normalize=True))
print(ds.groupby("ano").y.mean())
print(ds[["acwr","minutos_7d","jogos_30d"]].describe())
EOF
```

---

## 12. Prompt sugerido para a próxima sessão

> Continuar o projeto em `Estimativa_risco_lesao`. Ler `HANDOFF.md`.  
> EDA, modelagem e artigo concluídos. Ver pendências na seção 9 (referências, autoria, commit).

---

*Fim do handoff.*
