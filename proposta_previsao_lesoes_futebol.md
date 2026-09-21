---
título: "Previsão de Risco de Lesões em Jogadores de Futebol via Machine Learning"
disciplinas: [Tópicos Especiais em Machine Learning, Ciência de Dados]
status: proposta
tags: [futebol, ml, ciencia-de-dados, sports-analytics, injury-prediction]
data: 2026-09-01
---

# Previsão de Risco de Lesões em Jogadores de Futebol via Machine Learning

> [!info] Sobre este documento
> Proposta de trabalho para as disciplinas de **Tópicos Especiais em Machine Learning** e **Ciência de Dados**, desenvolvida em dupla. Estrutura pensada para ser apresentada como projeto único, com entregáveis distintos por disciplina.

---

## 1. Resumo

Este trabalho propõe o desenvolvimento de um modelo de **estimativa de risco de lesão** em jogadores profissionais de futebol, utilizando dados públicos de desempenho e calendário de partidas como proxy de carga de trabalho (workload). A hipótese central, amplamente documentada na literatura de ciência do esporte, é que o acúmulo de carga física — medido indiretamente por minutos jogados, congestionamento de calendário e histórico de lesões — está associado a um risco elevado de lesão em janelas de tempo subsequentes. O projeto combina uma etapa de **extração e engenharia de dados** (histórico de lesões via scraping, estatísticas de partidas via API) com uma etapa de **modelagem preditiva** (comparação entre modelos estatísticos clássicos, árvores de decisão/ensemble e redes neurais), avaliada sob uma validação temporal rigorosa. O resultado esperado não é um "diagnóstico", mas um **score de risco relativo**, discutido criticamente quanto a suas limitações e vieses.

---

## 2. Introdução

### 2.1 Contexto e motivação
Lesões são um dos fatores mais custosos no futebol profissional — afetam o desempenho esportivo do clube, o valor de mercado do atleta e, no caso de fantasy games e apostas esportivas, também têm impacto financeiro indireto. Historicamente, a gestão de carga de treino era feita de forma empírica pelos departamentos de fisiologia dos clubes; nos últimos anos, a disponibilidade de dados de desempenho e o avanço de técnicas de Machine Learning abriram espaço para modelos preditivos de risco baseados em dados objetivos.

### 2.2 Problema de pesquisa
Clubes profissionais utilizam dados proprietários de GPS e monitoramento fisiológico (frequência cardíaca, percepção subjetiva de esforço) para essa tarefa — dados inacessíveis para um projeto acadêmico. A pergunta que guia este trabalho é:

> **É possível estimar risco de lesão de forma útil utilizando apenas dados públicos de partidas (minutos jogados, calendário, histórico de lesões), sem acesso a dados proprietários de treino?**

### 2.3 Objetivos

**Objetivo geral:** desenvolver e comparar modelos de Machine Learning capazes de estimar a probabilidade de um jogador sofrer uma lesão em uma janela de tempo futura, com base em indicadores de carga de trabalho derivados de dados públicos.

**Objetivos específicos:**
- Construir uma base de dados integrando histórico de lesões, minutos jogados e calendário de partidas;
- Derivar métricas de carga de trabalho utilizadas na literatura esportiva (ex.: razão de carga aguda:crônica, dias de descanso, densidade de jogos);
- Comparar desempenho preditivo entre um modelo estatístico simples, um modelo de ensemble baseado em árvores (Random Forest / XGBoost) e uma rede neural (MLP);
- Avaliar os modelos sob métricas apropriadas para classes desbalanceadas (lesão é evento raro);
- Discutir criticamente as limitações do uso de dados de partida como proxy de carga real de treino.

---

## 3. Trabalhos Relacionados

A literatura de ciência do esporte aplicada a Machine Learning já trata esse problema há alguns anos, o que fornece boa base teórica e metodológica para o projeto.

Estudos revisando o uso de ML para compreensão e previsão de lesões no futebol destacam que o monitoramento de carga é essencial para entender a adaptação a programas de treino, avaliar fadiga e recuperação, e minimizar o risco de lesões, distinguindo entre carga interna (fisiológica/psicológica) e carga externa (velocidade, aceleração, distância percorrida).

Uma revisão publicada em periódico de medicina esportiva reforça que avanços em análise de dados e ML têm sido usados para entender melhor a relação entre carga de jogador e lesão, embora reconheça que a amplitude de variáveis usadas como indicadores de carga ainda carece de padronização entre os estudos.

Um estudo com jogadores profissionais iranianos utilizando dados de GPS mostrou que o número de desacelerações teve o maior poder preditivo para ocorrência de lesão entre as variáveis testadas, usando um algoritmo de árvore de decisão sobre a razão de carga aguda:crônica.

Em um estudo com dados da Premier League inglesa, comparando modelos para prever desempenho e lesões, o XGBoost apresentou desempenho superior para prever lesões, enquanto a Regressão Ridge foi mais eficaz para prever desempenho — os autores também identificaram um vínculo forte entre o aumento de carga de trabalho sob calendários congestionados e a elevação nas taxas de lesão.

Um framework mais recente, publicado em preprint, combinando teoria da decisão com ML, identificou que a média móvel exponencial da distância percorrida nos últimos 21 dias foi o preditor mais forte de risco, seguida da intensidade de corrida em alta velocidade e da frequência de acelerações/desacelerações do dia — o estudo também testou o modelo de forma prospectiva em uma temporada não vista durante o treinamento.

Por fim, material técnico voltado à indústria (Sportmonks) argumenta que **não é necessário acesso a dados de GPS de treino** para construir um modelo de risco de lesão minimamente informativo: dados de partidas públicas (minutos jogados, frequência de jogos, congestionamento de calendário, histórico de lesões, idade e posição) já constituem uma base razoável — o que valida diretamente a viabilidade deste projeto com dados abertos.

> [!tip] Contribuição do projeto em relação à literatura
> Diferente da maioria dos estudos citados (que usam dados de GPS/preparo físico fornecidos por clubes), este projeto testa a hipótese de que **apenas dados públicos de partida** já permitem construir um modelo minimamente útil — um recorte mais realista para um contexto acadêmico sem acesso a dados proprietários.

---

## 4. Base de Dados

### 4.1 Fontes candidatas

| Fonte | Dado fornecido | Tipo de acesso | Observações |
|---|---|---|---|
| **Transfermarkt** | Histórico de lesões por jogador (tipo, data de início/fim, jogos perdidos), minutos jogados, elenco, idade, posição | Scraping (sem API oficial) | Fonte mais utilizada em estudos acadêmicos sobre lesão por não haver alternativa pública equivalente |
| **API-Football** / **FBref** | Calendário de partidas, minutos jogados por jogo, escalações | API REST / scraping estruturado | Complementa o calendário e a carga de jogos |
| **Kaggle** (datasets de lesões no futebol) | Alguns datasets prontos combinando lesão + estatísticas de clube específico | Download direto | Útil para prototipagem rápida antes de investir no scraping completo |

### 4.2 Estratégia de extração (scraping)

O maior desafio de dados deste projeto é o histórico de lesões, que não está disponível via API — precisa ser extraído do Transfermarkt. Proposta de abordagem:

1. **Seleção do escopo**: escolher uma liga e um conjunto de temporadas (ex.: Brasileirão Série A ou uma liga europeia, 3-5 temporadas) para manter o volume de scraping gerenciável.
2. **Mapeamento das páginas**: cada jogador no Transfermarkt possui uma página de perfil com uma aba de histórico de lesões (`/spieler/{id}/verletzungen/spieler/{id}`) e uma aba de estatísticas de participação em partidas por temporada.
3. **Ferramentas**: `requests` + `BeautifulSoup` para páginas estáticas; caso o conteúdo seja carregado via JavaScript, considerar `Selenium` ou `Playwright`.
4. **Boas práticas técnicas**:
   - Respeitar `robots.txt` e aplicar *rate limiting* (delays entre requisições) para não sobrecarregar o servidor;
   - Fazer cache local das páginas já baixadas (HTML bruto) para não precisar re-raspar em caso de erro no parsing;
   - Versionar o dataset extraído (ex.: `injuries_raw_v1.csv`) separado do dataset tratado.
5. **Estrutura de dados esperada por registro de lesão**: `player_id`, `nome`, `data_inicio`, `data_fim`, `tipo_de_lesão`, `jogos_perdidos`, `clube_na_época`.
6. **Estrutura de dados esperada por partida/participação**: `player_id`, `data_da_partida`, `minutos_jogados`, `posição`, `competição`.

> [!warning] Ponto de atenção ético/legal
> Scraping do Transfermarkt deve respeitar os termos de uso do site. Como é um projeto acadêmico sem fins comerciais e sem redistribuição dos dados brutos, o risco é baixo, mas vale mencionar essa limitação explicitamente no relatório final e evitar extração agressiva (muitas requisições simultâneas).

### 4.3 Alternativa caso o scraping seja inviável no prazo
Caso o volume de scraping se mostre grande demais para o cronograma da disciplina, uma alternativa é reduzir o escopo para **um único clube ou um grupo pequeno de ~30-50 jogadores** ao longo de várias temporadas, o que ainda permite construir uma base longitudinal (várias observações por jogador ao longo do tempo) suficiente para os modelos propostos.

---

## 5. Metodologia

### 5.1 Definição do alvo (target)
Variável binária: **o jogador sofrerá uma lesão nos próximos *N* dias?** (sugestão inicial: N = 14 dias, ajustável conforme volume de dados disponível). Cada linha do dataset representa uma "observação" de um jogador em um ponto no tempo (ex.: a cada partida disputada), com o alvo calculado a partir da data da próxima lesão registrada.

### 5.2 Engenharia de features

| Categoria | Features propostas |
|---|---|
| Carga aguda | Minutos jogados nos últimos 7 dias; nº de partidas nos últimos 7 dias |
| Carga crônica | Média de minutos jogados nas últimas 3-4 semanas |
| Razão aguda:crônica | Razão entre carga aguda e crônica (métrica clássica na literatura de ciência do esporte) |
| Congestionamento de calendário | Dias desde a última partida; nº de jogos nos últimos 30 dias |
| Perfil do jogador | Idade; posição; histórico de lesões prévias (contagem e recência) |
| Contexto | Competição (liga vs. copa/torneios internacionais, que aumentam densidade de jogos) |

### 5.3 Modelos a comparar
- **Baseline**: Regressão Logística — interpretável, referência mínima de comparação.
- **Ensemble de árvores**: Random Forest e/ou XGBoost — literatura indica desempenho superior para esse tipo de problema.
- **Rede Neural (MLP)**: para atender ao requisito de rede neural da disciplina de ML, comparando se a complexidade adicional se justifica frente aos modelos de árvore (discussão crítica esperada, já que a literatura tende a favorecer XGBoost em dados tabulares como este).

### 5.4 Tratamento do desbalanceamento de classes
Lesões são eventos raros comparados ao volume de partidas sem lesão — isso torna o problema uma classificação com classes fortemente desbalanceadas. Estratégias a testar:
- Ponderação de classes (`class_weight`) nos modelos;
- Reamostragem (oversampling da classe minoritária, ex. SMOTE, ou undersampling da classe majoritária);
- Métricas de avaliação **não devem incluir acurácia como métrica principal** — priorizar AUC-ROC, AUC-PR (precision-recall), F1-score da classe positiva.

### 5.5 Validação
- **Split temporal**: treinar em temporadas passadas, testar na temporada mais recente disponível — nunca embaralhar aleatoriamente entre partidas do mesmo jogador (risco de vazamento de dados/data leakage).
- Validação cruzada agrupada por jogador (*group k-fold*) como alternativa complementar, garantindo que dados do mesmo jogador não apareçam simultaneamente em treino e teste.

### 5.6 Interpretabilidade
Uso de **SHAP values** (ou importância de features nativa dos modelos de árvore) para identificar quais variáveis mais contribuem para o risco estimado — item que agrega valor acadêmico e conecta com a discussão da literatura sobre quais fatores são mais preditivos (carga aguda recente, congestionamento, histórico prévio).

---

## 6. Divisão de Tarefas Entre Disciplinas

| Etapa | Ciência de Dados | Tópicos Especiais em ML |
|---|---|---|
| Extração (scraping Transfermarkt + API de calendário) | ✅ Entregável principal | Reaproveitado como insumo |
| Limpeza e integração das bases | ✅ Entregável principal | Reaproveitado como insumo |
| Engenharia de features (carga aguda/crônica etc.) | ✅ Parcial (cálculo das métricas) | ✅ Parcial (seleção/uso nas features do modelo) |
| Análise exploratória (distribuição de lesões por idade, posição, época da temporada) | ✅ Entregável principal | — |
| Modelagem preditiva (Regressão, Random Forest/XGBoost, MLP) | — | ✅ Entregável principal |
| Tratamento de desbalanceamento e validação temporal | — | ✅ Entregável principal |
| Interpretabilidade (SHAP) | — | ✅ Entregável principal |

---

## 7. Cronograma Sugerido *(seção adicional sugerida)*

| Semana | Atividade |
|---|---|
| 1-2 | Definição do escopo (liga/temporadas), setup do ambiente, testes iniciais de scraping |
| 3-4 | Extração completa (lesões + calendário/minutos), armazenamento bruto |
| 5-6 | Limpeza, integração das bases, engenharia de features |
| 7 | Análise exploratória (entregável de Ciência de Dados) |
| 8-9 | Modelagem: baseline, Random Forest/XGBoost |
| 10 | Modelagem: rede neural (MLP) + comparação |
| 11 | Validação temporal, tratamento de desbalanceamento, ajuste de hiperparâmetros |
| 12 | Interpretabilidade (SHAP), discussão crítica dos resultados |
| 13-14 | Redação final dos relatórios e preparação da apresentação |

---

## 8. Riscos e Limitações *(seção adicional sugerida)*

- **Ausência de dados de GPS/preparo físico**: os proxies de carga (minutos jogados, calendário) são uma aproximação da carga real de treino e jogo, não uma medida direta — isso deve ser reconhecido como limitação central do projeto, não escondido.
- **Volume de dados**: histórico de lesões por jogador costuma ser esparso; pode ser necessário agregar múltiplos jogadores/temporadas para ter volume suficiente para os modelos, especialmente a rede neural.
- **Viés de sobrevivência**: jogadores com carreiras mais longas tendem a estar sobrerrepresentados na base, o que pode enviesar o modelo em relação a jogadores mais jovens/menos estabelecidos.
- **Confundidores não observados**: fatores como qualidade do preparo físico do clube, histórico médico individual detalhado e genética não estão disponíveis nos dados públicos e podem explicar parte da variância não capturada pelo modelo.
- **Generalização entre ligas**: um modelo treinado em uma liga (ex.: europeia, com calendário mais congestionado) pode não generalizar bem para outra (ex.: Brasileirão, com calendário distinto).

---

## 9. Considerações Éticas *(seção adicional sugerida)*

Diferente de dados de saúde de pessoas anônimas, este projeto trabalha com **dados de lesões de atletas profissionais já publicamente divulgados** pelos próprios clubes e pela imprensa esportiva, reduzindo a sensibilidade em relação à privacidade. Ainda assim, recomenda-se:
- Tratar o resultado do modelo como **estimativa de risco relativo**, nunca como diagnóstico ou previsão determinística de lesão de um jogador específico;
- Evitar linguagem que sugira certeza médica (ex.: "o jogador X vai se lesionar") — preferir formulações como "o modelo indica um padrão de carga associado a risco elevado";
- Deixar claro nas limitações que o modelo não substitui avaliação de profissionais de fisiologia/medicina esportiva.

---

## 10. Resultados Esperados

- Uma base de dados estruturada e documentada, combinando histórico de lesões e carga de trabalho — reaproveitável para trabalhos futuros;
- Comparação empírica entre modelos estatísticos, ensemble de árvores e rede neural para esse problema específico, com discussão sobre qual abordagem é mais adequada dado o volume/tipo de dado disponível;
- Identificação (via interpretabilidade) das variáveis mais associadas a risco de lesão na amostra estudada, contrastando com o que a literatura já aponta (carga aguda recente, congestionamento de calendário, histórico prévio);
- Discussão crítica sobre a viabilidade de se estimar risco de lesão apenas com dados públicos, sem acesso a dados proprietários de GPS/fisiologia.

---

## 11. Referências

- Sportmonks. *Building an Injury Risk Model With Football Data*. Disponível em: https://www.sportmonks.com/blogs/building-an-injury-risk-model-with-football-data/
- *Prediction of football injuries using GPS-based data in Iranian professional football players: a machine learning approach*. Frontiers in Sports and Active Living, 2025. Disponível em: https://www.frontiersin.org/journals/sports-and-active-living/articles/10.3389/fspor.2025.1425180/full
- *Machine Learning for Understanding and Predicting Injuries in Football*. Sports Medicine - Open, 2022. Disponível em: https://link.springer.com/article/10.1186/s40798-022-00465-4
- *Football Analytics: Assessing the Correlation between Workload, Injury and Performance of Football Players in the English Premier League*. Applied Sciences, 2024. Disponível em: https://doi.org/10.3390/app14167217
- *Informed Injury Prediction in Elite Football: Decision Theory meets Machine Learning*. medRxiv, 2025 (preprint). Disponível em: https://www.medrxiv.org/content/10.1101/2025.04.23.25326218v1.full

---

## 12. Seções Adicionais Sugeridas (não incluídas em detalhe)

Pontos que valem a pena desenvolver mais à frente, conforme o projeto avança:

- **Anexo com dicionário de dados**: tabela completa de todas as colunas do dataset final, tipos e fonte de origem — facilita muito a correção pelo professor.
- **Anexo de reprodutibilidade**: ambiente (`requirements.txt`), seed fixa para os experimentos, instruções de execução do pipeline de scraping → tratamento → modelagem.
- **Seção de trabalhos futuros**: ex. incorporar dados de clima/gramado, ou expandir para múltiplas ligas para testar generalização.

> [!note] Nota final
> Este documento é uma proposta viva — recomenda-se revisar as seções de Metodologia e Riscos conforme o scraping avançar e o volume real de dados disponível ficar mais claro.
