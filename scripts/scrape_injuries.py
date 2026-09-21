"""
Extrai a tabela de historico de lesoes de cada jogador da lista.

Uso:
    python scripts/scrape_injuries.py                      # usa data/processed/squad_raw.csv
    python scripts/scrape_injuries.py --jogadores 68290 50144

Saida: data/processed/injuries_raw.csv (uma linha por episodio de lesao).

A aba /verletzungen/ ainda e HTML estatico, e paginada para jogadores com historico
longo (o script segue todas as paginas).
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import pandas as pd
from bs4 import BeautifulSoup

sys.path.append(str(Path(__file__).resolve().parent))
from tm_client import BASE, PROCESSED_DIR, ErroDeExtracao, get_html

# Ordem das colunas na tabela do site: Temporada | Lesao | de | ate | Dias | Jogos perdidos
COLUNAS = ["temporada_tm", "tipo_lesao", "data_inicio", "data_fim", "dias_parado", "jogos_perdidos"]


def url_lesoes(player_id: str, pagina: int = 1) -> str:
    base = f"{BASE}/-/verletzungen/spieler/{player_id}"
    return base if pagina == 1 else f"{base}/plus/1/page/{pagina}"


def _para_data(texto: str) -> str | None:
    """'dd/mm/aaaa' -> 'aaaa-mm-dd'. Vazio (lesao em curso) devolve None."""
    achado = re.match(r"(\d{2})/(\d{2})/(\d{4})", texto or "")
    if not achado:
        return None
    dia, mes, ano = achado.groups()
    return f"{ano}-{mes}-{dia}"


def _para_inteiro(texto: str) -> int | None:
    """'36 dias' -> 36; '8' -> 8; '-' ou vazio -> None."""
    achado = re.search(r"\d+", (texto or "").replace(".", ""))
    return int(achado.group()) if achado else None


def parse_pagina(html: str) -> list[dict]:
    """Extrai as linhas de lesao de uma pagina."""
    soup = BeautifulSoup(html, "lxml")
    tabela = soup.find("table", class_="items")
    if tabela is None:
        return []  # jogador sem nenhuma lesao registrada: a tabela nao existe

    registros: list[dict] = []
    for tr in (tabela.find("tbody") or tabela).find_all("tr", recursive=False):
        celulas = [td.get_text(" ", strip=True) for td in tr.find_all("td", recursive=False)]
        if len(celulas) != len(COLUNAS):
            continue  # linha de cabecalho intermediaria ou rodape

        bruto = dict(zip(COLUNAS, celulas))
        registros.append(
            {
                "temporada_tm": bruto["temporada_tm"],
                "tipo_lesao": bruto["tipo_lesao"],
                "data_inicio": _para_data(bruto["data_inicio"]),
                "data_fim": _para_data(bruto["data_fim"]),
                "dias_parado": _para_inteiro(bruto["dias_parado"]),
                "jogos_perdidos": _para_inteiro(bruto["jogos_perdidos"]),
                "em_curso": _para_data(bruto["data_fim"]) is None,
            }
        )

    return registros


def paginas_extras(html: str) -> list[int]:
    """Numeros das paginas 2..N da paginacao de lesoes."""
    soup = BeautifulSoup(html, "lxml")
    paginas = {
        int(achado.group(1))
        for link in soup.find_all("a", href=True)
        if (achado := re.search(r"/verletzungen/spieler/\d+.*?/page/(\d+)", link["href"]))
    }
    return sorted(p for p in paginas if p > 1)


def extrai_jogador(player_id: str, usa_cache: bool = True) -> list[dict]:
    """Todas as lesoes de um jogador, seguindo a paginacao."""
    html = get_html(url_lesoes(player_id), f"{player_id}_verletzungen", usa_cache)
    lesoes = parse_pagina(html)

    for pagina in paginas_extras(html):
        html_pagina = get_html(
            url_lesoes(player_id, pagina), f"{player_id}_verletzungen_p{pagina}", usa_cache
        )
        lesoes.extend(parse_pagina(html_pagina))

    return lesoes


def _caminho_checkpoint(saida: str) -> Path:
    return Path(saida).with_suffix(".done.txt")


def _carrega_feitos(saida: str) -> set[str]:
    """Lê player_ids ja processados (arquivo .done.txt; fallback no CSV)."""
    checkpoint = _caminho_checkpoint(saida)
    if checkpoint.exists():
        return {linha.strip() for linha in checkpoint.read_text(encoding="utf-8").splitlines() if linha.strip()}
    if Path(saida).exists():
        return set(pd.read_csv(saida, dtype={"player_id": str})["player_id"].astype(str))
    return set()


def _salva(registros: list[dict], feitos: set[str], saida: str) -> None:
    """Persiste o CSV de lesoes e o checkpoint de player_ids."""
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(registros).to_csv(saida, index=False, encoding="utf-8")
    _caminho_checkpoint(saida).write_text(
        "\n".join(sorted(feitos)) + ("\n" if feitos else ""), encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Extrai historico de lesoes")
    parser.add_argument("--entrada", default=str(PROCESSED_DIR / "squad_raw.csv"))
    parser.add_argument("--jogadores", nargs="+", help="ids avulsos, ignora --entrada")
    parser.add_argument("--saida", default=str(PROCESSED_DIR / "injuries_raw.csv"))
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
        print(f"[{i}/{total}] lesoes de {nome or player_id} ({player_id})", flush=True)
        try:
            lesoes = extrai_jogador(player_id, usa_cache=not args.no_cache)
        except ErroDeExtracao as erro:
            print(f"  FALHOU: {erro}")
            falhas.append(player_id)
            continue

        print(f"  -> {len(lesoes)} lesoes")
        for lesao in lesoes:
            registros.append({"player_id": player_id, "nome": nome, **lesao})

        ja_feitos.add(player_id)
        processados_neste_run += 1

        if processados_neste_run % args.lote == 0:
            _salva(registros, ja_feitos, args.saida)
            print(f"  [checkpoint] salvo apos {processados_neste_run} jogadores deste run")

    _salva(registros, ja_feitos, args.saida)
    n_com_lesao = len({r["player_id"] for r in registros})
    print(f"\n{len(registros)} lesoes de {n_com_lesao} jogadores com historico -> {args.saida}")
    print(f"jogadores cobertos no checkpoint: {len(ja_feitos)}")
    if falhas:
        print(f"jogadores que falharam ({len(falhas)}): {falhas}")


if __name__ == "__main__":
    main()
