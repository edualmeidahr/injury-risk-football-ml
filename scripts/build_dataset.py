"""
Fase 3 — limpeza + engenharia de features → dataset de modelagem.

Uso:
    python scripts/build_dataset.py
    python scripts/build_dataset.py --ano-min 2022 --ano-max 2024 --n-alvo 14

Entradas (data/processed/):
    squad_raw.csv, injuries_raw.csv, performance_raw.csv, clubs_serie_a.csv

Saidas (data/processed/):
    injuries_clean.csv   — lesoes tipadas + categoria anatomica
    dataset_modelo.csv   — uma linha por partida BRA1 (features + y)
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent))
from features import (
    N_ALVO_DIAS,
    engenharia_features,
    limpar_lesoes,
    limpar_performance,
    montar_dataset_modelo,
)
from tm_client import PROCESSED_DIR


def main() -> None:
    parser = argparse.ArgumentParser(description="Monta o dataset de modelagem")
    parser.add_argument("--ano-min", type=int, default=2022)
    parser.add_argument("--ano-max", type=int, default=2024)
    parser.add_argument(
        "--n-alvo",
        type=int,
        default=N_ALVO_DIAS,
        help="janela do alvo em dias (default 14)",
    )
    args = parser.parse_args()

    # permite ajustar N sem editar o modulo (monkey-patch pontual)
    import features as feat

    feat.N_ALVO_DIAS = args.n_alvo

    t0 = time.time()
    print("1/4 Lendo CSVs brutos...")
    squad = pd.read_csv(PROCESSED_DIR / "squad_raw.csv")
    injuries = pd.read_csv(PROCESSED_DIR / "injuries_raw.csv")
    performance = pd.read_csv(PROCESSED_DIR / "performance_raw.csv")
    clubs = pd.read_csv(PROCESSED_DIR / "clubs_serie_a.csv")

    print("2/4 Limpando lesoes e performance...")
    inj_clean = limpar_lesoes(injuries)
    perf_clean = limpar_performance(performance)
    inj_path = PROCESSED_DIR / "injuries_clean.csv"
    inj_clean.to_csv(inj_path, index=False, encoding="utf-8")
    print(f"  lesoes: {len(injuries)} -> {len(inj_clean)} (salvas em {inj_path.name})")
    print(f"  categorias:\n{inj_clean['categoria_lesao'].value_counts().to_string()}")
    print(
        f"  contam como alvo: {inj_clean['conta_como_alvo'].sum()} / {len(inj_clean)}"
    )

    # restringe o historico de performance aos jogadores do elenco Serie A
    # (ainda usa TODA a carreira deles para as janelas de carga)
    ids = set(squad["player_id"].astype(str))
    perf_clean = perf_clean[perf_clean["player_id"].isin(ids)].copy()
    print(f"  performance filtrada aos {len(ids)} jogadores do squad: {len(perf_clean)} linhas")

    print("3/4 Calculando features de carga (pode levar alguns minutos)...")
    perf_feat = engenharia_features(perf_clean, inj_clean)

    print("4/4 Montando dataset de modelagem (BRA1, partidas jogadas)...")
    dataset = montar_dataset_modelo(
        perf_feat, squad, clubs, ano_min=args.ano_min, ano_max=args.ano_max
    )

    out_path = PROCESSED_DIR / "dataset_modelo.csv"
    dataset.to_csv(out_path, index=False, encoding="utf-8")

    print(f"\n{'=' * 60}")
    print(f"dataset: {len(dataset)} observacoes -> {out_path}")
    print(f"  com lesao no dia da partida (lesao_no_dia=1): {int(dataset['lesao_no_dia'].sum())}")

    # amostra PRINCIPAL: exclui lesao no proprio dia (alvo y = (d, d+N])
    principal = dataset.loc[dataset["lesao_no_dia"].eq(0)]
    n = len(principal)
    pos = int(principal["y"].sum())
    print(f"\namostra principal: {n} observacoes | jogadores: {principal['player_id'].nunique()}")
    print(f"periodo: {principal['data'].min().date()} a {principal['data'].max().date()}")
    print(f"alvo y=1 (lesao em (d, d+{args.n_alvo}]): {pos} ({100 * pos / n:.2f}%)")
    print(
        f"sensibilidade y_com_dia ([d, d+{args.n_alvo}], todas as linhas): "
        f"{int(dataset['y_com_dia'].sum())} ({100 * dataset['y_com_dia'].mean():.2f}%)"
    )
    print("\npor ano:")
    print(
        principal.groupby("ano")
        .agg(n=("y", "size"), lesoes=("y", "sum"), taxa=("y", "mean"))
        .assign(taxa=lambda d: (100 * d["taxa"]).round(2))
        .join(dataset.groupby("ano")["y_com_dia"].mean().mul(100).round(2).rename("taxa_com_dia"))
        .to_string()
    )
    print(f"\ntempo total: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
