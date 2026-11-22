"""
Funcoes de apoio da Fase 5 (modelagem) — usadas por notebooks/02_modelagem.ipynb.

Concentra aqui: matriz de atributos, grupos para ablacao, fabrica de modelos,
metricas (sem limiar e top-k) e bootstrap por jogador.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

try:  # xgboost e opcional: se nao estiver instalado, usa o HistGradientBoosting
    from xgboost import XGBClassifier

    TEM_XGB = True
except ImportError:  # pragma: no cover
    TEM_XGB = False

SEMENTE = 42

# Limites (caps) para atributos de "tempo desde": evitam sentinela gigante no
# StandardScaler. O indicador `nunca_lesionado` carrega a informacao do nulo.
CAP_DIAS = 730
CAP_JOGOS = 100

POSICOES = ["zagueiro", "lateral", "volante", "meia", "ponta", "atacante"]  # goleiro = referencia


def preparar_atributos(ds: pd.DataFrame) -> pd.DataFrame:
    """Matriz X a partir do dataset_modelo. Transformacoes sem ajuste (sem vazamento).

    Nulos que restam (ACWR sem cronica) sao imputados DENTRO do Pipeline,
    com estatisticas do treino de cada fold.
    """
    X = pd.DataFrame(index=ds.index)

    # --- perfil ---
    X["idade"] = ds["idade"].astype(float)
    X["titular"] = ds["titular"].astype(float)
    X["goleiro"] = ds["posicao_grupo"].eq("goleiro").astype(float)
    for p in POSICOES:
        X[f"pos_{p}"] = ds["posicao_grupo"].eq(p).astype(float)

    # --- historico de lesoes ---
    X["n_lesoes_previas"] = ds["n_lesoes_previas"].clip(upper=15).astype(float)
    X["n_lesoes_12m"] = ds["n_lesoes_12m"].astype(float)
    X["nunca_lesionado"] = ds["dias_desde_ultima_lesao"].isna().astype(float)
    X["log_dias_desde_lesao"] = np.log1p(ds["dias_desde_ultima_lesao"].fillna(CAP_DIAS).clip(upper=CAP_DIAS))
    X["log_dias_desde_retorno"] = np.log1p(ds["dias_desde_retorno"].fillna(CAP_DIAS).clip(upper=CAP_DIAS))
    X["log_jogos_desde_retorno"] = np.log1p(ds["jogos_desde_retorno"].fillna(CAP_JOGOS).clip(upper=CAP_JOGOS))
    X["log_dias_fora_ultima"] = np.log1p(ds["dias_fora_ultima_lesao"].fillna(0))

    # --- calendario / exposicao ---
    X["jogos_30d"] = ds["jogos_30d"].astype(float)
    X["log_dias_desde_partida"] = np.log1p(ds["dias_desde_ultima_partida"].fillna(60).clip(upper=60))
    X["jogos_clube_prox_14d"] = ds["jogos_clube_prox_14d"].astype(float)

    # --- carga (minutos) ---
    X["minutos_7d"] = ds["minutos_7d"].astype(float)
    X["minutos_cronica_4sem"] = ds["minutos_cronica_4sem"].astype(float)
    X["acwr"] = ds["acwr"].astype(float)  # NaN = sem cronica -> imputador com indicador
    X["ewma_aguda"] = ds["ewma_aguda"].astype(float)
    X["ewma_cronica"] = ds["ewma_cronica"].astype(float)
    X["acwr_ewma"] = ds["acwr_ewma"].clip(upper=3).astype(float)
    return X


GRUPOS = {
    "perfil": ["idade", "titular", "goleiro"] + [f"pos_{p}" for p in POSICOES],
    "historico": [
        "n_lesoes_previas", "n_lesoes_12m", "nunca_lesionado", "log_dias_desde_lesao",
        "log_dias_desde_retorno", "log_jogos_desde_retorno", "log_dias_fora_ultima",
    ],
    "calendario": ["jogos_30d", "log_dias_desde_partida", "jogos_clube_prox_14d"],
    "carga": ["minutos_7d", "minutos_cronica_4sem", "acwr", "ewma_aguda", "ewma_cronica", "acwr_ewma"],
}
TODAS = [c for g in GRUPOS.values() for c in g]


# ---------------------------------------------------------------------------
# Modelos: (pipeline base, grade de hiperparametros)
# ---------------------------------------------------------------------------
def _imputador() -> SimpleImputer:
    return SimpleImputer(strategy="median", add_indicator=True)


def fabrica_modelos(n_neg_por_pos: float) -> dict[str, tuple[Pipeline, dict]]:
    """Modelos e grades. A grade e pequena de proposito: ~270 episodios no menor treino."""
    peso = float(np.sqrt(n_neg_por_pos))  # meio-termo entre sem peso e n0/n1
    modelos = {
        "Regressao logistica": (
            Pipeline([("imp", _imputador()), ("esc", StandardScaler()),
                      ("clf", LogisticRegression(max_iter=3000))]),
            {"clf__C": [0.01, 0.1, 1.0], "clf__class_weight": [None, "balanced"]},
        ),
        "Random Forest": (
            Pipeline([("imp", _imputador()),
                      ("clf", RandomForestClassifier(n_estimators=400, class_weight="balanced_subsample",
                                                     n_jobs=8, random_state=SEMENTE))]),
            {"clf__min_samples_leaf": [25, 100], "clf__max_features": ["sqrt", 0.5]},
        ),
    }
    if TEM_XGB:
        modelos["XGBoost"] = (
            Pipeline([("clf", XGBClassifier(n_estimators=300, learning_rate=0.05, subsample=0.8,
                                            colsample_bytree=0.8, reg_lambda=5.0, eval_metric="aucpr",
                                            n_jobs=8, random_state=SEMENTE))]),
            {"clf__max_depth": [2, 3, 4], "clf__min_child_weight": [5, 20],
             "clf__scale_pos_weight": [1.0, peso]},
        )
    else:
        modelos["Gradient Boosting"] = (
            Pipeline([("clf", HistGradientBoostingClassifier(learning_rate=0.05, l2_regularization=1.0,
                                                             min_samples_leaf=50, random_state=SEMENTE))]),
            {"clf__max_depth": [2, 3, 4], "clf__max_iter": [150, 300],
             "clf__class_weight": [None, {0: 1.0, 1: peso}]},
        )
    modelos["MLP"] = (
        Pipeline([("imp", _imputador()), ("esc", StandardScaler()),
                  ("clf", MLPClassifier(hidden_layer_sizes=(32, 16), early_stopping=True,
                                        validation_fraction=0.15, max_iter=300, random_state=SEMENTE))]),
        {"clf__alpha": [1e-3, 1e-2, 1e-1]},
    )
    return modelos


def ajustar(pipe: Pipeline, grade: dict, X: pd.DataFrame, y: pd.Series, grupos: pd.Series) -> GridSearchCV:
    """Busca em grade com CV estratificada e agrupada por jogador, otimizando AUC-PR."""
    cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEMENTE)
    busca = GridSearchCV(pipe, grade, scoring="average_precision", cv=cv, n_jobs=1, refit=True)
    busca.fit(X, y, groups=grupos)
    return busca


# ---------------------------------------------------------------------------
# Metricas
# ---------------------------------------------------------------------------
def top_k(y: np.ndarray, p: np.ndarray, frac: float = 0.10) -> tuple[float, float]:
    """Precisao e recall entre as `frac` observacoes de maior score.

    Empates na fronteira do top-k sao tratados pelo VALOR ESPERADO: entram todos
    os scores acima do corte e, do grupo empatado no corte, a fracao necessaria
    para completar k, com a taxa media de positivos desse grupo. Sem isso, regras
    binarias e o Dummy dependeriam da ordem das linhas (que segue a data).
    """
    y = np.asarray(y)
    p = np.asarray(p, dtype=float)
    k = max(1, int(round(frac * len(p))))
    corte = np.sort(p)[::-1][k - 1]
    acima = p > corte
    empatados = p == corte
    faltam = k - acima.sum()
    tp = y[acima].sum() + faltam * y[empatados].mean()
    return tp / k, tp / max(1, y.sum())


def metricas(y, p, frac: float = 0.10) -> dict:
    y = np.asarray(y)
    p = np.asarray(p, dtype=float)
    prev = y.mean()
    ap = average_precision_score(y, p)
    prec_k, rec_k = top_k(y, p, frac)
    return {
        "AP": ap,
        "lift_AP": ap / prev,
        "AUC_ROC": roc_auc_score(y, p),
        f"P@{int(frac * 100)}%": prec_k,
        f"R@{int(frac * 100)}%": rec_k,
        f"lift@{int(frac * 100)}%": prec_k / prev,
        "Brier": brier_score_loss(y, np.clip(p, 0, 1)),
        "prevalencia": prev,
    }


def bootstrap_jogador(y, scores: dict[str, np.ndarray], jogadores, n_boot: int = 500,
                      frac: float = 0.10, semente: int = SEMENTE) -> dict[str, pd.DataFrame]:
    """Bootstrap por cluster (jogador): reamostra jogadores do teste com reposicao.

    Devolve, por modelo, as amostras de AP e P@k — os mesmos sorteios para todos
    os modelos, o que permite comparacao PAREADA (diferencas por amostra).
    """
    y = np.asarray(y)
    jog = np.asarray(jogadores)
    unicos, inv = np.unique(jog, return_inverse=True)
    indices = [np.flatnonzero(inv == j) for j in range(len(unicos))]
    rng = np.random.default_rng(semente)
    saida = {m: {"AP": np.empty(n_boot), "Pk": np.empty(n_boot)} for m in scores}
    for b in range(n_boot):
        sorteio = rng.integers(0, len(unicos), len(unicos))
        idx = np.concatenate([indices[j] for j in sorteio])
        yb = y[idx]
        for m, s in scores.items():
            sb = np.asarray(s)[idx]
            saida[m]["AP"][b] = average_precision_score(yb, sb)
            saida[m]["Pk"][b] = top_k(yb, sb, frac)[0]
    return {m: pd.DataFrame(v) for m, v in saida.items()}


def ic(amostras: np.ndarray, nivel: float = 0.95) -> tuple[float, float]:
    a = (1 - nivel) / 2
    return float(np.quantile(amostras, a)), float(np.quantile(amostras, 1 - a))
