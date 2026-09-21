"""
Extrai o elenco de um clube (ou de varios clubes) em uma ou mais temporadas.

Uso:
    python scripts/scrape_squad.py --clube 609 --temporadas 2022 2023 2024
    python scripts/scrape_squad.py --clubes-csv data/processed/clubs_serie_a.csv

Saida: data/processed/squad_raw.csv (uma linha por jogador POR temporada x clube).

As temporadas sao informadas em ANO CIVIL do Brasileirao; a conversao para o
`saison_id` do Transfermarkt (ano - 1) e feita internamente.
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


def url_elenco(clube_id: int, saison_id: int) -> str:
    # o slug do clube e irrelevante para o servidor: '-' funciona e evita depender do nome
    return f"{BASE}/-/startseite/verein/{clube_id}/saison_id/{saison_id}"


def _texto(no) -> str:
    return no.get_text(" ", strip=True) if no else ""


def parse_elenco(html: str) -> list[dict]:
    """Extrai os jogadores da tabela de elenco (class='items')."""
    soup = BeautifulSoup(html, "lxml")
    tabela = soup.find("table", class_="items")
    if tabela is None:
        return []

    jogadores: list[dict] = []
    for tr in tabela.find("tbody").find_all("tr", recursive=False):
        celulas = tr.find_all("td", recursive=False)
        if len(celulas) < 3:
            continue

        # A celula do jogador tem uma tabela interna: linha 1 = nome/link, linha 2 = posicao
        celula_jogador = celulas[1]
        link = celula_jogador.find("a", href=re.compile(r"/profil/spieler/\d+"))
        if link is None:
            continue
        player_id = re.search(r"/spieler/(\d+)", link["href"]).group(1)

        linhas_internas = celula_jogador.select("table.inline-table tr")
        posicao = _texto(linhas_internas[-1]) if len(linhas_internas) > 1 else ""

        # 'dd/mm/aaaa (idade)' -> separa os dois pedacos
        nasc_idade = _texto(celulas[2])
        achado = re.match(r"(\d{2}/\d{2}/\d{4})\s*\((\d+)\)", nasc_idade)

        bandeira = celulas[3].find("img") if len(celulas) > 3 else None

        jogadores.append(
            {
                "player_id": player_id,
                "nome": link.get_text(strip=True),
                "camisa": _texto(celulas[0]),
                "posicao": posicao,
                "data_nascimento": achado.group(1) if achado else None,
                "idade_na_temporada": int(achado.group(2)) if achado else None,
                "nacionalidade": bandeira.get("title") if bandeira else None,
                "valor_mercado": _texto(celulas[-1]),
                "url_perfil": BASE + link["href"],
            }
        )

    return jogadores


def coleta_um(clube_id: int, ano: int, usa_cache: bool = True, nome_clube: str = "") -> list[dict]:
    """Extrai o elenco de um clube em uma temporada."""
    saison_id = temporada_para_saison_id(ano)
    rotulo = nome_clube or str(clube_id)
    print(f"[elenco] {rotulo} (id={clube_id}) temporada={ano} (saison_id={saison_id})")

    html = get_html(url_elenco(clube_id, saison_id), f"squad_{clube_id}_{saison_id}", usa_cache)
    jogadores = parse_elenco(html)
    print(f"  -> {len(jogadores)} jogadores")

    return [
        {
            "temporada": ano,
            "clube_id": clube_id,
            "nome_clube": nome_clube or None,
            **jogador,
        }
        for jogador in jogadores
    ]


def coleta(clube_id: int, temporadas: list[int], usa_cache: bool = True) -> pd.DataFrame:
    """Roda o parser para cada temporada de um unico clube."""
    registros: list[dict] = []
    for ano in temporadas:
        registros.extend(coleta_um(clube_id, ano, usa_cache))
    return pd.DataFrame(registros)


def coleta_de_csv(caminho_clubes: str, usa_cache: bool = True) -> pd.DataFrame:
    """Extrai elenco para cada linha de clubs_serie_a.csv (clube x temporada).

    Em caso de bloqueio (403), pula o clube e continua — reexecute o script depois
    para completar o que faltou (cache cobre o ja baixado).
    """
    from tm_client import ErroDeExtracao

    clubes = pd.read_csv(caminho_clubes)
    registros: list[dict] = []
    falhas: list[str] = []
    total = len(clubes)
    for i, linha in enumerate(clubes.itertuples(index=False), start=1):
        print(f"--- [{i}/{total}] ---")
        try:
            registros.extend(
                coleta_um(
                    int(linha.clube_id),
                    int(linha.temporada),
                    usa_cache,
                    nome_clube=getattr(linha, "nome_clube", "") or "",
                )
            )
        except ErroDeExtracao as erro:
            rotulo = f"{getattr(linha, 'nome_clube', '')} {linha.clube_id}/{linha.temporada}"
            print(f"  PULANDO (falha): {erro}")
            falhas.append(rotulo)
            continue
    if falhas:
        print(f"\n{len(falhas)} elencos falharam (reexecute para completar):")
        for f in falhas:
            print(f"  - {f}")
    return pd.DataFrame(registros)


def main() -> None:
    parser = argparse.ArgumentParser(description="Extrai elenco(s) de clube(s)")
    parser.add_argument("--clube", type=int, help="id do clube no Transfermarkt")
    parser.add_argument("--temporadas", type=int, nargs="+", help="anos civis (com --clube)")
    parser.add_argument(
        "--clubes-csv",
        help="CSV com colunas temporada,clube_id[,nome_clube] (ex.: clubs_serie_a.csv)",
    )
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument(
        "--saida", default=str(PROCESSED_DIR / "squad_raw.csv"), help="CSV de saida"
    )
    args = parser.parse_args()

    if args.clubes_csv:
        df = coleta_de_csv(args.clubes_csv, usa_cache=not args.no_cache)
    elif args.clube and args.temporadas:
        df = coleta(args.clube, args.temporadas, usa_cache=not args.no_cache)
    else:
        parser.error("informe --clube + --temporadas, ou --clubes-csv")

    if df.empty:
        sys.exit("Nenhum jogador extraido — conferir id do clube e seletores.")

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.saida, index=False, encoding="utf-8")

    unicos = df.drop_duplicates(subset="player_id")
    print(f"\n{len(df)} linhas (jogador x temporada x clube) -> {args.saida}")
    print(f"{len(unicos)} jogadores unicos")
    print(f"{df['clube_id'].nunique()} clubes distintos")
    print("\nlinhas por temporada:")
    print(df.groupby("temporada").size().to_string())


if __name__ == "__main__":
    main()
