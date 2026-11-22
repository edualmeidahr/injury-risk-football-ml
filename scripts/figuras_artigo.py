"""
Gera as figuras do artigo IEEE em formato de coluna (rótulos em português).

Uso:
    python scripts/figuras_artigo.py

Saídas em notebooks/figuras/:
    artigo_historico_vs_acwr.png  risco relativo por recência da lesão e por faixa de ACWR
    artigo_lift_decil.png         taxa de lesão por decil de risco (regressão logística, corte B)
    artigo_shap.png               SHAP do XGBoost no teste de 2025 (corte B)

Reajusta os modelos do corte B com a mesma busca do notebook 02 (alguns segundos).
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent))
import modelagem as M
from tm_client import PROCESSED_DIR, RAIZ

warnings.filterwarnings("ignore")
FIG = RAIZ / "notebooks" / "figuras"
FIG.mkdir(parents=True, exist_ok=True)

# Estilo compatível com o IEEE (coluna de ~3,5 pol.)
plt.rcParams.update({
    "font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8, "xtick.labelsize": 7.5,
    "ytick.labelsize": 7.5, "legend.fontsize": 7.5, "axes.spines.top": False,
    "axes.spines.right": False, "figure.dpi": 300, "savefig.bbox": "tight",
})
VERDE, AMARELO, VERMELHO, CINZA = "#1d6a44", "#c98f00", "#b93a28", "#8a958d"


def virgula(eixo) -> None:
    """Decimais com vírgula nos rótulos de um eixo (padrão do artigo em português)."""
    from matplotlib.ticker import FuncFormatter
    eixo.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}".replace(".", ",").replace("-", "\u2212")))


def carregar() -> pd.DataFrame:
    ds = pd.read_csv(PROCESSED_DIR / "dataset_modelo.csv", parse_dates=["data"])
    return ds.loc[ds["lesao_no_dia"].eq(0)].reset_index(drop=True)


def risco_relativo(ds: pd.DataFrame, faixa: pd.Series) -> pd.DataFrame:
    """Taxa da faixa / taxa média do ano, com IC 95% aproximado."""
    rr = ds["y"] / ds.groupby("ano")["y"].transform("mean")
    g = pd.DataFrame({"f": faixa, "rr": rr}).groupby("f", observed=True)["rr"]
    out = g.agg(["mean", "std", "size"])
    out["ic"] = 1.96 * out["std"] / np.sqrt(out["size"])
    return out


def fig_historico_vs_acwr(ds: pd.DataFrame) -> None:
    dias = pd.cut(ds["dias_desde_ultima_lesao"], [0, 30, 90, 180, 365, 1e6],
                  labels=["≤30", "31–90", "91–180", "181–365", ">365"]).astype(object)
    dias[ds["dias_desde_ultima_lesao"].isna()] = "nunca"
    dias = pd.Categorical(dias, ["≤30", "31–90", "91–180", "181–365", ">365", "nunca"], ordered=True)

    a = ds["acwr"]
    acwr = pd.cut(a, [0, 0.8, 1.3, 1.5, np.inf], right=False,
                  labels=["0–0,8", "0,8–1,3", "1,3–1,5", "≥1,5"]).astype(object)
    acwr[ds["minutos_7d"].eq(0) & a.notna()] = "0 min\n(7 dias)"
    acwr[a.isna()] = "sem\nbase"
    ordem = ["sem\nbase", "0 min\n(7 dias)", "0–0,8", "0,8–1,3", "1,3–1,5", "≥1,5"]
    acwr = pd.Categorical(acwr, ordem, ordered=True)

    t1, t2 = risco_relativo(ds, dias), risco_relativo(ds, acwr)
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.3), sharey=True)
    for ax, t, cor, titulo, xl in [
        (axes[0], t1, VERDE, "(a) Tempo desde a última lesão", "dias"),
        (axes[1], t2, AMARELO, "(b) ACWR (carga recente ÷ carga de base)", "faixa"),
    ]:
        x = np.arange(len(t))
        ax.bar(x, t["mean"], yerr=t["ic"], color=cor, capsize=2, width=0.7, error_kw={"lw": 0.8})
        ax.axhline(1, color=CINZA, ls="--", lw=0.8)
        ax.set_xticks(x, t.index.astype(str))
        ax.set_title(titulo, loc="left")
        ax.set_xlabel(xl)
    axes[0].set_ylabel("risco relativo ao ano")
    virgula(axes[0].yaxis)
    axes[0].set_ylim(0, 2.4)
    fig.tight_layout()
    fig.savefig(FIG / "artigo_historico_vs_acwr.png")
    plt.close(fig)
    print("  historico:", t1["mean"].round(2).to_dict())
    print("  acwr:", {k.replace("\n", " "): v for k, v in t2["mean"].round(2).to_dict().items()})


def ajustar_corte_b(ds: pd.DataFrame, X: pd.DataFrame, nome: str):
    tr, te = ds["ano"].eq(2024), ds["ano"].eq(2025)
    ytr = ds.loc[tr, "y"]
    pipe, grade = M.fabrica_modelos((1 - ytr.mean()) / ytr.mean())[nome]
    busca = M.ajustar(pipe, grade, X[tr], ytr, ds.loc[tr, "player_id"])
    return busca.best_estimator_, X[te], ds.loc[te, "y"]


def fig_lift_decil(ds: pd.DataFrame, X: pd.DataFrame) -> None:
    est, Xte, yte = ajustar_corte_b(ds, X, "Regressao logistica")
    p = est.predict_proba(Xte)[:, 1]
    dec = pd.qcut(pd.Series(p).rank(method="first"), 10, labels=range(1, 11))
    lift = pd.Series(yte.to_numpy()).groupby(dec.to_numpy()).mean() / yte.mean()
    cores = [VERDE] * 5 + [AMARELO] * 3 + [VERMELHO] * 2
    fig, ax = plt.subplots(figsize=(3.45, 2.1))
    ax.bar(lift.index.astype(str), lift.values, color=cores, width=0.72)
    ax.axhline(1, color=CINZA, ls="--", lw=0.8)
    for i, v in enumerate(lift.values):
        ax.text(i, v + 0.04, f"{v:.2f}".replace(".", ","), ha="center", fontsize=6.5)
    ax.set_xlabel("decil de risco previsto (10 = maior risco)")
    ax.set_ylabel("taxa observada ÷ média")
    ax.set_ylim(0, 2.35)
    virgula(ax.yaxis)
    fig.tight_layout()
    fig.savefig(FIG / "artigo_lift_decil.png")
    plt.close(fig)
    print("  lift por decil:", lift.round(2).tolist())


NOMES_PT = {
    "jogos_clube_prox_14d": "jogos do clube nos próx. 14 dias",
    "log_dias_desde_lesao": "tempo desde a última lesão",
    "log_dias_fora_ultima": "duração da última lesão",
    "n_lesoes_previas": "nº de lesões anteriores",
    "n_lesoes_12m": "lesões nos últimos 12 meses",
    "idade": "idade", "titular": "titular", "goleiro": "goleiro",
    "minutos_cronica_4sem": "carga de base (28 dias)", "minutos_7d": "minutos em 7 dias",
    "acwr": "ACWR", "acwr_ewma": "ACWR (EWMA)", "ewma_aguda": "EWMA recente",
    "ewma_cronica": "EWMA de base", "log_jogos_desde_retorno": "jogos desde o retorno",
    "log_dias_desde_retorno": "tempo desde o retorno", "jogos_30d": "jogos em 30 dias",
    "log_dias_desde_partida": "dias desde o último jogo", "nunca_lesionado": "sem lesão anterior",
}


def fig_shap(ds: pd.DataFrame, X: pd.DataFrame) -> None:
    if not M.TEM_XGB:
        print("  xgboost ausente: figura SHAP não gerada")
        return
    import shap
    import xgboost as xgb

    est, Xte, _ = ajustar_corte_b(ds, X, "XGBoost")
    booster = est.named_steps["clf"].get_booster()
    # TreeSHAP nativo (o TreeExplainer do shap 0.49 não lê o base_score do xgboost 3.x)
    sv = booster.predict(xgb.DMatrix(Xte), pred_contribs=True)[:, :-1]
    nomes = [NOMES_PT.get(c, c.replace("pos_", "posição: ")) for c in Xte.columns]
    shap.summary_plot(sv, Xte.to_numpy(), feature_names=nomes, max_display=10, show=False,
                      plot_size=(3.6, 3.0), color_bar_label="valor do atributo")
    fig = plt.gcf()
    ax = plt.gca()
    ax.set_xlabel("efeito no risco previsto (SHAP)", fontsize=7.5)
    ax.tick_params(labelsize=7)
    virgula(ax.xaxis)
    for cax in fig.axes[1:]:
        cax.tick_params(labelsize=6.5)
        cax.set_yticklabels(["baixo", "alto"])
        cax.yaxis.label.set_size(7)
    fig.savefig(FIG / "artigo_shap.png")
    plt.close(fig)
    top = pd.Series(np.abs(sv).mean(0), index=nomes).sort_values(ascending=False).head(5)
    print("  SHAP top 5:", top.round(3).to_dict())


def main() -> None:
    ds = carregar()
    X = M.preparar_atributos(ds)
    print("1/3 histórico × ACWR"); fig_historico_vs_acwr(ds)
    print("2/3 lift por decil"); fig_lift_decil(ds, X)
    print("3/3 SHAP"); fig_shap(ds, X)
    print("figuras em", FIG)


if __name__ == "__main__":
    main()
