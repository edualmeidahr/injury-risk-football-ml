"""
Extrai o desempenho POR PARTIDA de cada jogador da lista.

Uso:
    python scripts/scrape_performance.py                   # usa data/processed/squad_raw.csv
    python scripts/scrape_performance.py --jogadores 68290 --desde 2021

Saida: data/processed/performance_raw.csv (uma linha por jogador x partida).

A aba /leistungsdaten/ e renderizada por JavaScript e nao tem tabela no HTML; os dados
vem da API interna /ceapi/performance-game/<player_id>, que devolve granularidade por
partida — inclusive os jogos em que o jogador ficou fora (participacao='injured'),
o que e essencial para montar a serie temporal de carga.
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent))
from tm_client import BASE, PROCESSED_DIR, ErroDeExtracao, get_json


def url_desempenho(player_id: str) -> str:
    return f"{BASE}/ceapi/performance-game/{player_id}"


def parse_desempenho(dados: dict) -> list[dict]:
    """Achata a resposta da API em uma linha por partida."""
    registros: list[dict] = []

    for partida in dados.get("data", {}).get("performance", []):
        jogo = partida.get("gameInformation", {})
        clubes = partida.get("clubsInformation", {})
        stats = partida.get("statistics", {})
        geral = stats.get("generalStatistics", {})
        tempo = stats.get("playingTimeStatistics", {})
        gols = stats.get("goalStatistics", {})

        registros.append(
            {
                "game_id": jogo.get("gameId"),
                "data": (jogo.get("date") or {}).get("dateTimeUTC", "")[:10],
                "temporada_tm": (jogo.get("season") or {}).get("nonCyclicalName"),
                "season_id": jogo.get("seasonId"),
                "competicao": jogo.get("competitionId"),
                "rodada": jogo.get("gameDay"),
                "duracao_jogo": jogo.get("gameDuration"),
                "jogo_de_selecao": jogo.get("isNationalGame"),
                "clube_id": (clubes.get("club") or {}).get("clubId"),
                "adversario_id": (clubes.get("opponent") or {}).get("clubId"),
                "local": (clubes.get("club") or {}).get("venue"),
                # played / injured / in squad / not in squad / absent
                "participacao": geral.get("participationState"),
                "minutos": tempo.get("playedMinutes"),
                "titular": tempo.get("isStarting"),
                "posicao_id": geral.get("positionId"),
                "idade": geral.get("age"),
                "injury_id": geral.get("injuryId"),
                "gols": gols.get("goalsScoredTotal"),
                "assistencias": gols.get("assists"),
            }
        )

    registros.sort(key=lambda r: r["data"] or "", reverse=True)
    return registros


def _caminho_checkpoint(saida: str) -> Path:
    return Path(saida).with_suffix(".done.txt")


def _carrega_feitos(saida: str) -> set[str]:
    checkpoint = _caminho_checkpoint(saida)
    if checkpoint.exists():
        return {linha.strip() for linha in checkpoint.read_text(encoding="utf-8").splitlines() if linha.strip()}
    if Path(saida).exists():
        return set(pd.read_csv(saida, dtype={"player_id": str})["player_id"].astype(str))
    return set()


def _salva(registros: list[dict], feitos: set[str], saida: str) -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(registros).to_csv(saida, index=False, encoding="utf-8")
    _caminho_checkpoint(saida).write_text(
        "\n".join(sorted(feitos)) + ("\n" if feitos else ""), encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Extrai desempenho por partida")
    parser.add_argument("--entrada", default=str(PROCESSED_DIR / "squad_raw.csv"))
    parser.add_argument("--jogadores", nargs="+", help="ids avulsos, ignora --entrada")
    parser.add_argument("--saida", default=str(PROCESSED_DIR / "performance_raw.csv"))
    parser.add_argument("--desde", type=int, help="descarta partidas antes deste ano")
    parser.add_argument("--lote", type=int, default=50, help="salva o CSV a cada N jogadores")
    parser.add_argument(
        "--retomar",
        action="store_true",
        help="pula player_ids ja presentes no checkpoint (.done.txt)",
    )
    parser.add_argument("--no-cache", action="store_true")
    args = parser.parse_args()

    if args.jogadores:
        jogadores = [(pid, "") for pid in args.jogadores]
    else:
        elenco = pd.read_csv(args.entrada, dtype={"player_id": str})
        elenco = elenco.drop_duplicates(subset="player_id")
        jogadores = list(zip(elenco["player_id"], elenco["nome"]))

    ja_feitos: set[str] = set()
    registros: list[dict] = []
    if args.retomar:
        ja_feitos = _carrega_feitos(args.saida)
        if Path(args.saida).exists():
            registros = pd.read_csv(args.saida, dtype={"player_id": str}).to_dict("records")
        print(f"[checkpoint] {len(ja_feitos)} jogadores ja processados — retomando")

    pendentes = [(pid, nome) for pid, nome in jogadores if pid not in ja_feitos]
    total = len(pendentes)
    falhas: list[str] = []
    processados_neste_run = 0

    for i, (player_id, nome) in enumerate(pendentes, start=1):
        print(f"[{i}/{total}] desempenho de {nome or player_id} ({player_id})", flush=True)
        try:
            dados = get_json(
                url_desempenho(player_id), f"{player_id}_performance", usa_cache=not args.no_cache
            )
        except ErroDeExtracao as erro:
            print(f"  FALHOU: {erro}")
            falhas.append(player_id)
            continue

        partidas = parse_desempenho(dados)
        if args.desde:
            partidas = [p for p in partidas if p["data"] and int(p["data"][:4]) >= args.desde]

        minutos = sum(p["minutos"] or 0 for p in partidas)
        print(f"  -> {len(partidas)} partidas, {minutos} minutos")
        for partida in partidas:
            registros.append({"player_id": player_id, "nome": nome, **partida})

        ja_feitos.add(player_id)
        processados_neste_run += 1

        if processados_neste_run % args.lote == 0:
            _salva(registros, ja_feitos, args.saida)
            print(f"  [checkpoint] salvo apos {processados_neste_run} jogadores deste run")

    _salva(registros, ja_feitos, args.saida)

    df = pd.DataFrame(registros)
    print(f"\n{len(df)} registros de {len(ja_feitos)} jogadores -> {args.saida}")
    if not df.empty:
        print(f"estados de participacao: {dict(Counter(df['participacao']))}")
        print(f"periodo coberto: {df['data'].min()} a {df['data'].max()}")
    if falhas:
        print(f"jogadores que falharam ({len(falhas)}): {falhas}")


if __name__ == "__main__":
    main()
