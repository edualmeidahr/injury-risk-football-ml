"""
Lista os clubes da Serie A (BRA1) em uma ou mais temporadas.

Uso:
    python scripts/scrape_league.py --temporadas 2022 2023 2024

Saida: data/processed/clubs_serie_a.csv
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import pandas as pd
from bs4 import BeautifulSoup

sys.path.append(str(Path(__file__).resolve().parent))
from tm_client import BASE, PROCESSED_DIR, get_html, temporada_para_saison_id


def url_liga(saison_id: int) -> str:
    return f"{BASE}/serie-a/startseite/wettbewerb/BRA1/saison_id/{saison_id}"


def parse_clubes(html: str) -> list[dict]:
    soup = BeautifulSoup(html, "lxml")
    tabela = soup.find("table", class_="items")
    if tabela is None:
        return []

    clubes: dict[str, dict] = {}
    for a in tabela.select('a[href*="/startseite/verein/"]'):
        nome = a.get_text(strip=True)
        if not nome:
            continue
        achado = re.search(r"/verein/(\d+)", a["href"])
        if not achado:
            continue
        clube_id = achado.group(1)
        clubes[clube_id] = {
            "clube_id": int(clube_id),
            "nome_clube": nome,
            "url_elenco": BASE + a["href"].split("/saison_id/")[0],
        }
    return list(clubes.values())


def main() -> None:
    parser = argparse.ArgumentParser(description="Lista clubes da Serie A")
    parser.add_argument("--temporadas", type=int, nargs="+", required=True)
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument("--saida", default=str(PROCESSED_DIR / "clubs_serie_a.csv"))
    args = parser.parse_args()

    registros: list[dict] = []
    for ano in args.temporadas:
        saison_id = temporada_para_saison_id(ano)
        print(f"[liga] Serie A {ano} (saison_id={saison_id})")
        html = get_html(url_liga(saison_id), f"league_BRA1_{saison_id}", not args.no_cache)
        clubes = parse_clubes(html)
        print(f"  -> {len(clubes)} clubes")
        for clube in clubes:
            registros.append({"temporada": ano, **clube})

    df = pd.DataFrame(registros)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.saida, index=False, encoding="utf-8")
    print(f"\n{len(df)} linhas -> {args.saida}")
    print(df.groupby("temporada").size().to_string())


if __name__ == "__main__":
    main()
