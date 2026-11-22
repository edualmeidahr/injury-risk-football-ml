"""
Análises complementares pedidas na Etapa III (confiabilidade experimental).

Uso:
    python scripts/complementos_etapa3.py

Para cada corte (A, B, C) e modelo, repete a busca em grade do notebook 02 e informa:
  * AUC e AP no próprio treino (ajuste dentro da amostra);
  * AUC e AP na validação cruzada (5 partes agrupadas por jogador), média e desvio-padrão;
  * AUC e AP no teste.
A AUC não depende da proporção de positivos, então é a medida usada para comparar
treino e teste (a AP muda com a prevalência, que difere entre os anos).

Também gera, para o corte B (treino 2024, teste 2025):
  * matriz de confusão da regressão logística com alerta para os 10% de maior risco;
  * curva de precisão e recall (notebooks/figuras/artigo_curva_pr.png).

Saída em texto: logs/complementos_etapa3.txt
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold

sys.path.append(str(Path(__file__).resolve().parent))
import modelagem as M
from tm_client import PROCESSED_DIR, RAIZ

warnings.filterwarnings("ignore")
FIG = RAIZ / "notebooks" / "figuras"
LOG = RAIZ / "logs" / "complementos_etapa3.txt"
plt.rcParams.update({
    "font.size": 8, "axes.labelsize": 8, "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
    "legend.fontsize": 7.5, "axes.spines.top": False, "axes.spines.right": False,
    "figure.dpi": 300, "savefig.bbox": "tight",
})

CORTES = {"A": ([2022, 2023], [2024]), "B": ([2024], [2025]), "C": ([2022, 2023, 2024], [2025])}

bruto = pd.read_csv(PROCESSED_DIR / "dataset_modelo.csv", parse_dates=["data"])
ds = bruto.loc[bruto["lesao_no_dia"].eq(0)].reset_index(drop=True)
X_all = M.preparar_atributos(ds)
y_all = ds["y"]

saida: list[str] = []


def log(msg: str = "") -> None:
    print(msg, flush=True)
    saida.append(msg)


linhas = []
scores_b = {}
for c, (tr_anos, te_anos) in CORTES.items():
    tr, te = ds["ano"].isin(tr_anos), ds["ano"].isin(te_anos)
    Xtr, ytr, gtr = X_all[tr], y_all[tr], ds.loc[tr, "player_id"]
    Xte, yte = X_all[te], y_all[te]
    modelos = M.fabrica_modelos(n_neg_por_pos=(1 - ytr.mean()) / ytr.mean())
    for nome, (pipe, grade) in modelos.items():
        busca = M.ajustar(pipe, grade, Xtr, ytr, gtr)
        best = busca.best_estimator_
        # validação cruzada com a melhor configuração, mesmas partes da busca
        cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=M.SEMENTE)
        auc_f, ap_f = [], []
        for i_tr, i_va in cv.split(Xtr, ytr, gtr):
            est = clone(best).fit(Xtr.iloc[i_tr], ytr.iloc[i_tr])
            p = est.predict_proba(Xtr.iloc[i_va])[:, 1]
            auc_f.append(roc_auc_score(ytr.iloc[i_va], p))
            ap_f.append(average_precision_score(ytr.iloc[i_va], p))
        p_tr = best.predict_proba(Xtr)[:, 1]
        p_te = best.predict_proba(Xte)[:, 1]
        linhas.append({
            "corte": c, "modelo": nome,
            "AUC_treino": roc_auc_score(ytr, p_tr), "AUC_cv_media": np.mean(auc_f), "AUC_cv_dp": np.std(auc_f),
            "AUC_teste": roc_auc_score(yte, p_te),
            "AP_treino": average_precision_score(ytr, p_tr), "AP_cv_media": np.mean(ap_f), "AP_cv_dp": np.std(ap_f),
            "AP_teste": average_precision_score(yte, p_te),
            "prev_treino": ytr.mean(), "prev_teste": yte.mean(),
        })
        log(f"[{c}] {nome:20s} AUC treino={linhas[-1]['AUC_treino']:.3f} "
            f"cv={np.mean(auc_f):.3f}±{np.std(auc_f):.3f} teste={linhas[-1]['AUC_teste']:.3f}")
        if c == "B":
            scores_b[nome] = p_te
    if c == "B":
        yte_b = yte.to_numpy()
        scores_b["Regra de recência"] = -Xte["log_dias_desde_lesao"].to_numpy()

tab = pd.DataFrame(linhas).set_index(["corte", "modelo"])
log("\n=== Treino x validação cruzada x teste ===")
log(tab.round(3).to_string())

# --- matriz de confusão: regressão logística, corte B, alerta = 10% de maior risco ---
p = scores_b["Regressao logistica"]
k = int(round(0.10 * len(p)))
alerta = np.zeros(len(p), dtype=bool)
alerta[np.argsort(-p, kind="stable")[:k]] = True
vp = int((alerta & (yte_b == 1)).sum())
fp = int((alerta & (yte_b == 0)).sum())
fn = int((~alerta & (yte_b == 1)).sum())
vn = int((~alerta & (yte_b == 0)).sum())
log("\n=== Matriz de confusão (RL, corte B, top 10%) ===")
log(f"VP={vp} FP={fp} FN={fn} VN={vn} | precisão={vp / (vp + fp):.3f} recall={vp / (vp + fn):.3f} "
    f"especificidade={vn / (vn + fp):.3f} | n={len(p)}")

# --- curva de precisão e recall, corte B ---
fig, ax = plt.subplots(figsize=(3.4, 2.5))
cores = {"Regressao logistica": "#4C78A8", "XGBoost": "#F58518", "Random Forest": "#54A24B",
         "Regra de recência": "#9D755D"}
rotulo = {"Regressao logistica": "Reg. logística"}
for nome, cor in cores.items():
    if nome not in scores_b:
        continue
    pr, rc, _ = precision_recall_curve(yte_b, scores_b[nome])
    ap = average_precision_score(yte_b, scores_b[nome])
    ax.plot(rc, pr, color=cor, lw=1.2, label=f"{rotulo.get(nome, nome)} (AP {ap:.3f})".replace("0.", "0,"))
ax.axhline(yte_b.mean(), color="#E45756", ls="--", lw=1, label="acaso")
ax.plot(vp / (vp + fn), vp / (vp + fp), "o", color="#4C78A8", ms=4)
ax.annotate("10% de maior risco", (vp / (vp + fn), vp / (vp + fp)), xytext=(6, 6),
            textcoords="offset points", fontsize=7)
ax.set_xlabel("Recall (fração das lesões capturadas)")
ax.set_ylabel("Precisão (alertas corretos)")
ax.set_xlim(0, 1); ax.set_ylim(0, 0.4)
ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:.1f}".replace(".", ",")))
ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:.2f}".replace(".", ",")))
ax.legend(frameon=False, loc="upper right")
fig.savefig(FIG / "artigo_curva_pr.png")
log(f"\nfigura salva em {FIG / 'artigo_curva_pr.png'}")

LOG.parent.mkdir(exist_ok=True)
LOG.write_text("\n".join(saida) + "\n", encoding="utf-8")
