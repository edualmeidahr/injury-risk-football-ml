"""
Limpeza de lesoes, taxonomia e engenharia de features de carga.

Usado por build_dataset.py. Comentarios em portugues (relatorio final tambem e).
"""

from __future__ import annotations

import re
from typing import Iterable

import numpy as np
import pandas as pd

# Janela do alvo: lesao nos proximos N dias
N_ALVO_DIAS = 14

# Carga: aguda = 7 dias; cronica = media semanal das ultimas 4 semanas (28 dias)
DIAS_AGUDA = 7
DIAS_CRONICA = 28
SEMANAS_CRONICA = 4
DIAS_CALENDARIO = 30

# EWMA (Williams et al., 2017): lambda = 2 / (N + 1), com N = 7 (aguda) e 28 (cronica)
LAMBDA_AGUDA = 2 / (DIAS_AGUDA + 1)
LAMBDA_CRONICA = 2 / (DIAS_CRONICA + 1)

# Historico de lesoes em janela fixa (dias)
DIAS_HISTORICO = 365

# ---------------------------------------------------------------------------
# Taxonomia de lesoes (referencia: categorias do estudo UNICESUMAR / EPCC)
# Ordem importa: regras mais especificas primeiro.
# ---------------------------------------------------------------------------
REGRAS_TAXONOMIA: list[tuple[str, re.Pattern]] = [
    ("joelho", re.compile(r"joelho|cruzad|menisc|ligamento.*joelho|patel|rotulian", re.I)),
    ("tornozelo", re.compile(r"tornozelo|tibiot[aá]rsic|entorse.*p[eé]|ligamento.*tornoz", re.I)),
    ("coxa", re.compile(r"coxa|isquiotib|b[ií]ceps femoral|quadr[ií]ceps|posterior.*cox", re.I)),
    ("panturrilha", re.compile(r"panturrilha|gastrocn[eê]mio|s[oó]leo", re.I)),
    ("adutores", re.compile(r"adutor|virilha|pubalg|p[uú]bis|inguinal", re.I)),
    ("ombro", re.compile(r"ombro|manguito|clav[ií]cul|acromio", re.I)),
    ("pe", re.compile(r"\bp[eé]\b|metatars|plantaris|calc[aâ]neo|dedo do p[eé]", re.I)),
    ("perna", re.compile(r"perna|t[ií]bia|f[ií]bula|canela|compartiment", re.I)),
    ("quadril", re.compile(r"quadril|flexor.*quadril|ilio|sacro", re.I)),
    ("muscular_geral", re.compile(
        r"muscul|distens[aã]o|estiramento|ruptura muscular|cãibra|caimbra|fadiga muscular",
        re.I,
    )),
    ("cabeca", re.compile(r"cabe[cç]a|concuss|cr[aâ]nio|face|nariz|olho|mand[ií]bul", re.I)),
    ("coluna", re.compile(r"coluna|lombar|cervical|costas|h[eé]rnia", re.I)),
    ("mao_braco", re.compile(r"m[aã]o|pulso|antebra[cç]o|cotovelo|bra[cç]o|dedo(?! do p)", re.I)),
]

# Ausencias que NAO entram no alvo de "lesao" (doenca / gestao / fora de escopo)
RE_NAO_TRAUMATICA = re.compile(
    r"coronav[ií]rus|covid|gripe|gripal|resfriado|influenza|doente|doen[cç]a|"
    r"v[ií]rus|infec[cç][aã]o|febre|gastro|intestin|g[aá]stric|"
    r"gest[aã]o de esfor[cç]o|recupera[cç][aã]o ritmo|f[eé]rias|"
    r"motivos pessoais|problema[s]? dent|odont|"
    r"nascimento|paternidade|casamento|"
    r"suspens[aã]o|castigo|sem clube|rescis|"
    r"problemas f[ií]sicos$",  # genérico demais / ambíguo — tratado aparte se quiser
    re.I,
)

# Mapeamento de posicao Transfermarkt -> grupo analitico
MAPA_POSICAO = {
    "Goleiro": "goleiro",
    "Zagueiro": "zagueiro",
    "Defensores": "zagueiro",
    "Lateral Dir.": "lateral",
    "Lateral Esq.": "lateral",
    "Volante": "volante",
    "Meia Central": "meia",
    "Meio-Campo": "meia",
    "Meia Ofensivo": "meia",
    "Meia Direita": "meia",
    "Meia Esquerda": "meia",
    "Ponta Direita": "ponta",
    "Ponta Esquerda": "ponta",
    # a partir de 2025 o Transfermarkt passou a grafar com hifen
    "Ponta-Direita": "ponta",
    "Ponta-Esquerda": "ponta",
    "Seg. Atacante": "atacante",
    "Centroavante": "atacante",
    "Atacante": "atacante",
}


def categorizar_lesao(tipo: str) -> str:
    """Mapeia o texto livre do Transfermarkt para uma categoria anatomica."""
    if not isinstance(tipo, str) or not tipo.strip():
        return "desconhecida"
    if RE_NAO_TRAUMATICA.search(tipo):
        return "nao_traumatica"
    for categoria, padrao in REGRAS_TAXONOMIA:
        if padrao.search(tipo):
            return categoria
    if re.search(r"desconhecid|pequena les[aã]o|choque|contus[aã]o|trauma", tipo, re.I):
        return "outras"
    return "outras"


def eh_alvo_lesao(tipo: str) -> bool:
    """True se o episodio conta como evento positivo do alvo (lesao, nao doenca)."""
    return categorizar_lesao(tipo) not in {"nao_traumatica"}


def limpar_lesoes(injuries: pd.DataFrame) -> pd.DataFrame:
    """Tipa datas, deduplica e adiciona categoria anatomica + flag de alvo."""
    df = injuries.copy()
    df["player_id"] = df["player_id"].astype(str)
    df["data_inicio"] = pd.to_datetime(df["data_inicio"], errors="coerce")
    df["data_fim"] = pd.to_datetime(df["data_fim"], errors="coerce")
    df = df.dropna(subset=["data_inicio"])
    df["categoria_lesao"] = df["tipo_lesao"].map(categorizar_lesao)
    df["conta_como_alvo"] = df["categoria_lesao"] != "nao_traumatica"
    df = df.drop_duplicates(subset=["player_id", "data_inicio", "tipo_lesao"], keep="first")
    return df.sort_values(["player_id", "data_inicio"]).reset_index(drop=True)


def limpar_performance(performance: pd.DataFrame) -> pd.DataFrame:
    """Tipa datas/ids e define minutos jogados (0 se nao atuou)."""
    df = performance.copy()
    df["player_id"] = df["player_id"].astype(str)
    df["clube_id"] = df["clube_id"].astype(str)
    df["game_id"] = df["game_id"].astype(str)
    df["data"] = pd.to_datetime(df["data"], errors="coerce")
    df = df.dropna(subset=["data", "player_id"])
    df["minutos"] = pd.to_numeric(df["minutos"], errors="coerce")
    df["minutos_jogados"] = np.where(
        df["participacao"].eq("played"), df["minutos"].fillna(0.0), 0.0
    )
    df = df.sort_values(["player_id", "data", "game_id"]).reset_index(drop=True)
    return df


def posicao_por_jogador_temporada(squad: pd.DataFrame) -> pd.DataFrame:
    """Uma linha por player_id x temporada civil, com posicao agregada."""
    s = squad.copy()
    s["player_id"] = s["player_id"].astype(str)
    s["posicao_grupo"] = s["posicao"].map(MAPA_POSICAO).fillna("outras")
    # se o jogador mudou de posicao no mesmo ano (raro), fica a mais frequente
    agg = (
        s.groupby(["player_id", "temporada"], as_index=False)
        .agg(
            posicao=("posicao", lambda x: x.mode().iloc[0] if len(x) else None),
            posicao_grupo=("posicao_grupo", lambda x: x.mode().iloc[0] if len(x) else "outras"),
            nome=("nome", "first"),
            data_nascimento=("data_nascimento", "first"),
        )
    )
    return agg


def _features_um_jogador(
    g: pd.DataFrame,
    inicios_lesao: np.ndarray,
    categorias_lesao: np.ndarray | None = None,
) -> pd.DataFrame:
    """Calcula features de carga e alvo para as linhas de um unico jogador.

    Historico de minutos usa apenas partidas ANTERIORES a data da observacao
    (exclui a partida atual), para evitar vazamento. A unica informacao
    "futura" usada como atributo e o calendario do clube (jogos agendados),
    que e publico e conhecido antes da partida.
    """
    g = g.sort_values("data").reset_index(drop=True)
    n = len(g)
    dates = g["data"].values.astype("datetime64[D]")
    mins = g["minutos_jogados"].to_numpy(dtype=float)
    jogou = mins > 0
    if categorias_lesao is None:
        categorias_lesao = np.full(inicios_lesao.size, None, dtype=object)

    cum = np.concatenate([[0.0], np.cumsum(mins)])
    # cum_jogos[j] = nº de partidas com minutos > 0 em indices [0, j)
    cum_jogos = np.concatenate([[0], np.cumsum(jogou.astype(int))])
    # datas das partidas jogadas (para achar o retorno de cada lesao)
    datas_jogadas = dates[jogou]

    # --- calendario do clube: datas de jogos de clube em que o jogador estava
    # vinculado ao elenco, qualquer que seja a participacao (jogou, banco, fora,
    # lesionado). Exclui selecao (a API lista jogos da selecao ate para quem nao
    # foi convocado) e "played for a different team" (jogo do clube principal
    # enquanto o atleta atuava por outra equipe, ex.: sub-20).
    eh_clube = ~g["jogo_de_selecao"].fillna(False).astype(bool).to_numpy() \
        if "jogo_de_selecao" in g.columns else np.ones(n, dtype=bool)
    if "participacao" in g.columns:
        eh_clube &= g["participacao"].ne("played for a different team").to_numpy()
    datas_clube = np.unique(dates[eh_clube])

    # --- EWMA (Williams et al., 2017) sobre a serie DIARIA de minutos ---
    # Dias sem jogo entram com zero. O valor usado na observacao i e o do dia
    # anterior a partida (d-1), entao a partida corrente nao entra.
    dia_idx = (dates - dates[0]).astype(int)
    serie_diaria = np.zeros(dia_idx[-1] + 1)
    np.add.at(serie_diaria, dia_idx, mins)
    serie_diaria = pd.Series(serie_diaria)
    ewma_a_dia = serie_diaria.ewm(alpha=LAMBDA_AGUDA, adjust=False).mean().to_numpy()
    ewma_c_dia = serie_diaria.ewm(alpha=LAMBDA_CRONICA, adjust=False).mean().to_numpy()
    ontem = dia_idx - 1
    tem_ontem = ontem >= 0
    ewma_a = np.where(tem_ontem, ewma_a_dia[np.clip(ontem, 0, None)], 0.0)
    ewma_c = np.where(tem_ontem, ewma_c_dia[np.clip(ontem, 0, None)], 0.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        acwr_ewma = np.where(ewma_c > 0, ewma_a / ewma_c, np.nan)

    aguda = np.full(n, np.nan)
    cronica = np.full(n, np.nan)
    acwr = np.full(n, np.nan)
    acwr_desac = np.full(n, np.nan)
    dias_desde = np.full(n, np.nan)
    jogos_30 = np.full(n, np.nan)
    jogos_clube_prox = np.zeros(n, dtype=int)
    n_lesoes_prev = np.zeros(n, dtype=int)
    n_lesoes_12m = np.zeros(n, dtype=int)
    dias_desde_lesao = np.full(n, np.nan)
    dias_desde_retorno = np.full(n, np.nan)
    jogos_desde_retorno = np.full(n, np.nan)
    dias_fora_ultima = np.full(n, np.nan)
    cat_ultima = np.full(n, None, dtype=object)
    alvo = np.zeros(n, dtype=int)
    lesao_no_dia = np.zeros(n, dtype=int)

    # ultimo indice com minutos > 0 visto ate agora
    ultimo_jogo = -1

    for i in range(n):
        d = dates[i]

        # --- carga: janelas abertas a direita em d (nao incluem a partida i) ---
        ini_7 = d - np.timedelta64(DIAS_AGUDA, "D")
        ini_28 = d - np.timedelta64(DIAS_CRONICA, "D")
        ini_30 = d - np.timedelta64(DIAS_CALENDARIO, "D")

        # searchsorted: primeiro indice com data > limite
        l7 = int(np.searchsorted(dates, ini_7, side="right"))
        l28 = int(np.searchsorted(dates, ini_28, side="right"))
        l30 = int(np.searchsorted(dates, ini_30, side="right"))

        aguda[i] = cum[i] - cum[l7]
        minutos_28 = cum[i] - cum[l28]
        cronica[i] = minutos_28 / SEMANAS_CRONICA
        if cronica[i] > 0:
            acwr[i] = aguda[i] / cronica[i]
        # versao desacoplada: cronica = media semanal dos dias 8..28 (sem a semana aguda)
        cronica_desac = (cum[l7] - cum[l28]) / (SEMANAS_CRONICA - 1)
        if cronica_desac > 0:
            acwr_desac[i] = aguda[i] / cronica_desac

        jogos_30[i] = cum_jogos[i] - cum_jogos[l30]

        if ultimo_jogo >= 0:
            dias_desde[i] = (d - dates[ultimo_jogo]).astype(int)
        # atualiza depois de ler (partida atual conta para a proxima linha)
        if jogou[i]:
            ultimo_jogo = i

        # --- exposicao futura: jogos de clube agendados em (d, d+N] ---
        jogos_clube_prox[i] = int(
            np.searchsorted(datas_clube, d + np.timedelta64(N_ALVO_DIAS, "D"), side="right")
            - np.searchsorted(datas_clube, d, side="right")
        )

        # --- historico de lesoes anteriores a d ---
        if inicios_lesao.size:
            n_prev = int(np.searchsorted(inicios_lesao, d, side="left"))
            n_lesoes_prev[i] = n_prev
            # janela fixa: menos sensivel a cobertura antiga do Transfermarkt
            l365 = int(np.searchsorted(inicios_lesao, d - np.timedelta64(DIAS_HISTORICO, "D"), side="left"))
            n_lesoes_12m[i] = n_prev - l365
            if n_prev > 0:
                inicio_ult = inicios_lesao[n_prev - 1]
                dias_desde_lesao[i] = (d - inicio_ult).astype(int)
                cat_ultima[i] = categorias_lesao[n_prev - 1]
                # retorno = 1a partida jogada DEPOIS do inicio da lesao. Usa a data
                # observada em campo, nao `data_fim` (que tem erros, ver EDA 4.3).
                k = int(np.searchsorted(datas_jogadas, inicio_ult, side="right"))
                if k < datas_jogadas.size and datas_jogadas[k] <= d:
                    retorno = datas_jogadas[k]
                    dias_desde_retorno[i] = (d - retorno).astype(int)
                    # partidas jogadas desde o retorno, sem contar a atual
                    jogos_desde_retorno[i] = int(np.searchsorted(datas_jogadas, d, side="left")) - k
                    dias_fora_ultima[i] = (retorno - inicio_ult).astype(int)

            # alvo: lesao em (d, d+N], sem o dia da partida.
            # Lesao no proprio jogo encurta os minutos e vazaria o rotulo.
            ini = d + np.timedelta64(1, "D")
            fim = d + np.timedelta64(N_ALVO_DIAS, "D")
            a = int(np.searchsorted(inicios_lesao, ini, side="left"))
            b = int(np.searchsorted(inicios_lesao, fim, side="right"))
            alvo[i] = int(b > a)
            a0 = int(np.searchsorted(inicios_lesao, d, side="left"))
            b0 = int(np.searchsorted(inicios_lesao, d, side="right"))
            lesao_no_dia[i] = int(b0 > a0)

    out = g.copy()
    out["minutos_7d"] = aguda
    out["minutos_cronica_4sem"] = cronica
    out["acwr"] = acwr
    out["acwr_desacoplada"] = acwr_desac
    out["ewma_aguda"] = ewma_a
    out["ewma_cronica"] = ewma_c
    out["acwr_ewma"] = acwr_ewma
    out["dias_desde_ultima_partida"] = dias_desde
    out["jogos_30d"] = jogos_30
    out["jogos_clube_prox_14d"] = jogos_clube_prox
    out["n_lesoes_previas"] = n_lesoes_prev
    out["n_lesoes_12m"] = n_lesoes_12m
    out["dias_desde_ultima_lesao"] = dias_desde_lesao
    out["dias_desde_retorno"] = dias_desde_retorno
    out["jogos_desde_retorno"] = jogos_desde_retorno
    out["dias_fora_ultima_lesao"] = dias_fora_ultima
    out["categoria_ultima_lesao"] = cat_ultima
    out["lesao_prox_14d"] = alvo
    out["lesao_no_dia"] = lesao_no_dia
    # alvo alternativo [d, d+N]: inclui lesao no proprio jogo (analise de sensibilidade)
    out["y_com_dia"] = np.maximum(alvo, lesao_no_dia)
    return out


def engenharia_features(
    performance: pd.DataFrame,
    injuries_limpas: pd.DataFrame,
) -> pd.DataFrame:
    """Percorre todos os jogadores e empilha as features."""
    lesoes_alvo = injuries_limpas.loc[
        injuries_limpas["conta_como_alvo"], ["player_id", "data_inicio", "categoria_lesao"]
    ].sort_values(["player_id", "data_inicio"])
    inicios_por_jogador = {
        pid: (
            grp["data_inicio"].values.astype("datetime64[D]"),
            grp["categoria_lesao"].to_numpy(dtype=object),
        )
        for pid, grp in lesoes_alvo.groupby("player_id")
    }
    vazio = (np.array([], dtype="datetime64[D]"), np.array([], dtype=object))

    partes: list[pd.DataFrame] = []
    grupos = list(performance.groupby("player_id", sort=False))
    total = len(grupos)
    for i, (pid, g) in enumerate(grupos, start=1):
        if i % 200 == 0 or i == total:
            print(f"  features [{i}/{total}]", flush=True)
        inicios, categorias = inicios_por_jogador.get(pid, vazio)
        partes.append(_features_um_jogador(g, inicios, categorias))

    return pd.concat(partes, ignore_index=True)


def montar_dataset_modelo(
    performance_feat: pd.DataFrame,
    squad: pd.DataFrame,
    clubs: pd.DataFrame | None = None,
    ano_min: int = 2022,
    ano_max: int = 2024,
) -> pd.DataFrame:
    """Filtra observacoes de modelagem e anexa perfil do jogador.

    Observacao = partida em que o jogador JOGOU (participacao=played) na Série A
    (BRA1), entre ano_min e ano_max. Features de carga ja consideram todas as
    competicoes do historico do jogador.
    """
    df = performance_feat.copy()
    df["ano"] = df["data"].dt.year

    mask = (
        df["participacao"].eq("played")
        & df["competicao"].eq("BRA1")
        & df["ano"].between(ano_min, ano_max)
        & (df["minutos_jogados"] > 0)
    )
    obs = df.loc[mask].copy()
    # Lesao que comeca na propria partida (lesao_no_dia=1): na amostra PRINCIPAL
    # a linha sai (y=(d, d+N] nao a enxerga). Ela fica no arquivo, marcada, para a
    # analise de sensibilidade com y_com_dia = [d, d+N]. Quem consome o dataset
    # filtra: ds = ds[ds.lesao_no_dia.eq(0)] para o protocolo principal.

    # temporada civil do Brasileirao ~= ano do jogo
    pos = posicao_por_jogador_temporada(squad)
    obs = obs.merge(
        pos,
        left_on=["player_id", "ano"],
        right_on=["player_id", "temporada"],
        how="left",
        suffixes=("", "_squad"),
    )

    # nome do clube na Serie A daquele ano (se disponivel)
    if clubs is not None:
        c = clubs.copy()
        c["clube_id"] = c["clube_id"].astype(str)
        obs = obs.merge(
            c[["temporada", "clube_id", "nome_clube"]],
            left_on=["ano", "clube_id"],
            right_on=["temporada", "clube_id"],
            how="left",
            suffixes=("", "_liga"),
        )
        if "nome_clube_liga" in obs.columns:
            obs["nome_clube"] = obs["nome_clube_liga"].fillna(obs.get("nome_clube"))

    # idade: preferir a da API na data do jogo
    if "idade" not in obs.columns or obs["idade"].isna().all():
        obs["idade"] = np.nan

    colunas = [
        "player_id",
        "nome",
        "game_id",
        "data",
        "ano",
        "season_id",
        "competicao",
        "clube_id",
        "nome_clube",
        "adversario_id",
        "local",
        "minutos_jogados",
        "titular",
        "idade",
        "posicao",
        "posicao_grupo",
        "minutos_7d",
        "minutos_cronica_4sem",
        "acwr",
        "acwr_desacoplada",
        "ewma_aguda",
        "ewma_cronica",
        "acwr_ewma",
        "dias_desde_ultima_partida",
        "jogos_30d",
        "jogos_clube_prox_14d",
        "n_lesoes_previas",
        "n_lesoes_12m",
        "dias_desde_ultima_lesao",
        "dias_desde_retorno",
        "jogos_desde_retorno",
        "dias_fora_ultima_lesao",
        "categoria_ultima_lesao",
        "lesao_no_dia",
        "lesao_prox_14d",
        "y_com_dia",
    ]
    existentes = [c for c in colunas if c in obs.columns]
    out = obs[existentes].copy()
    out = out.rename(columns={"lesao_prox_14d": "y"})
    return out.sort_values(["data", "player_id"]).reset_index(drop=True)
