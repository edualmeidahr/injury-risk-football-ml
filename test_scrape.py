"""
FASE 0 — Teste manual de scraping de UM jogador do Transfermarkt.

Objetivo: conferir visualmente, no console, se os campos extraidos batem com o que a
pagina mostra no navegador. NAO e um pipeline: e um script descartavel de validacao de
seletores, antes de escrever os scrapers de verdade em scripts/.

Uso:
    python test_scrape.py "https://www.transfermarkt.com.br/<slug>/profil/spieler/<id>"

Opcoes:
    --no-cache      ignora o HTML ja salvo em data/raw/ e baixa de novo
    --no-pages      nao segue a paginacao da aba de lesoes (so a primeira pagina)

O bruto de cada resposta baixada fica cacheado em data/raw/, para que reexecutar o
script durante o ajuste do parser nao gere novas requisicoes ao site.

Duas fontes diferentes por jogador:
  - lesoes    -> HTML da aba /verletzungen/ (tabela estatica, parse com BeautifulSoup)
  - desempenho-> JSON de /ceapi/performance-game/<id> (a aba HTML e renderizada por JS)
"""

import argparse
import json
import random
import re
import sys
import time
from collections import Counter
from pathlib import Path

import requests
from bs4 import BeautifulSoup

# Diretorio de cache do HTML bruto (nao versionado no git)
RAW_DIR = Path(__file__).parent / "data" / "raw"

# User-Agent de navegador comum: o Transfermarkt bloqueia o UA default do requests
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "pt-BR,pt;q=0.9,en;q=0.8",
}

BASE = "https://www.transfermarkt.com.br"


# ---------------------------------------------------------------------------
# Requisicao e cache
# ---------------------------------------------------------------------------
def fetch(url: str, cache_name: str, use_cache: bool = True, tentativas: int = 3) -> str:
    """Baixa uma pagina (ou le do cache) e devolve o HTML como string."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = RAW_DIR / f"{cache_name}.html"

    if use_cache and cache_path.exists():
        print(f"[cache] {cache_path.name}")
        return cache_path.read_text(encoding="utf-8")

    for tentativa in range(1, tentativas + 1):
        try:
            print(f"[GET {tentativa}/{tentativas}] {url}")
            resp = requests.get(url, headers=HEADERS, timeout=30)
            resp.raise_for_status()
            resp.encoding = resp.apparent_encoding or "utf-8"
            cache_path.write_text(resp.text, encoding="utf-8")
            return resp.text
        except requests.RequestException as erro:
            print(f"  falhou: {erro}")
            if tentativa == tentativas:
                raise
            # backoff simples antes de tentar de novo
            time.sleep(2 * tentativa + random.random())

    raise RuntimeError("inalcancavel")


def fetch_json(url: str, cache_name: str, use_cache: bool = True, tentativas: int = 3) -> dict:
    """Mesma ideia de fetch(), mas para a API JSON interna do site (ceapi)."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = RAW_DIR / f"{cache_name}.json"

    if use_cache and cache_path.exists():
        print(f"[cache] {cache_path.name}")
        return json.loads(cache_path.read_text(encoding="utf-8"))

    cabecalhos = {**HEADERS, "Accept": "application/json"}
    for tentativa in range(1, tentativas + 1):
        try:
            print(f"[GET {tentativa}/{tentativas}] {url}")
            resp = requests.get(url, headers=cabecalhos, timeout=60)
            resp.raise_for_status()
            dados = resp.json()
            cache_path.write_text(json.dumps(dados, ensure_ascii=False), encoding="utf-8")
            return dados
        except (requests.RequestException, ValueError) as erro:
            print(f"  falhou: {erro}")
            if tentativa == tentativas:
                raise
            time.sleep(2 * tentativa + random.random())

    raise RuntimeError("inalcancavel")


# ---------------------------------------------------------------------------
# Parsing generico das tabelas do Transfermarkt
# ---------------------------------------------------------------------------
def _cabecalhos(tabela) -> list[str]:
    """Le o cabecalho da tabela, expandindo colspan para alinhar com as celulas."""
    linha_cabecalho = tabela.find("thead")
    if linha_cabecalho is None:
        return []

    nomes: list[str] = []
    for th in linha_cabecalho.find_all("th"):
        texto = th.get_text(" ", strip=True)
        span = int(th.get("colspan", 1))
        for i in range(span):
            # colunas sem titulo (icones, bandeiras) recebem nome posicional
            nome = texto or f"col_{len(nomes) + 1}"
            nomes.append(nome if i == 0 else f"{nome}_{i + 1}")
    return nomes


def parse_tabela_items(html: str) -> list[dict]:
    """Extrai a tabela principal (class='items') como lista de dicionarios."""
    soup = BeautifulSoup(html, "lxml")
    tabela = soup.find("table", class_="items")
    if tabela is None:
        return []

    nomes = _cabecalhos(tabela)
    corpo = tabela.find("tbody") or tabela
    registros: list[dict] = []

    for tr in corpo.find_all("tr", recursive=False):
        celulas = tr.find_all("td", recursive=False)
        if not celulas:
            continue  # linha de cabecalho intermediaria
        valores = [td.get_text(" ", strip=True) for td in celulas]
        if len(valores) == 1:
            continue  # linha de separacao/total
        chaves = nomes if len(nomes) == len(valores) else [
            f"col_{i + 1}" for i in range(len(valores))
        ]
        registros.append(dict(zip(chaves, valores)))

    return registros


def paginas_de_lesao(html: str) -> list[int]:
    """Descobre os numeros de pagina extras da aba de lesoes (paginacao)."""
    soup = BeautifulSoup(html, "lxml")
    paginas = set()
    for link in soup.find_all("a", href=True):
        achado = re.search(r"/verletzungen/spieler/\d+.*?/page/(\d+)", link["href"])
        if achado:
            paginas.add(int(achado.group(1)))
    return sorted(p for p in paginas if p > 1)


# ---------------------------------------------------------------------------
# Desempenho por partida (API JSON interna)
# ---------------------------------------------------------------------------
# A aba de desempenho deixou de ser HTML estatico: e renderizada por um componente
# JavaScript que consome /ceapi/performance-game/<player_id>. Usar a API direto e
# melhor que raspar a tabela, porque devolve granularidade POR PARTIDA (e nao o
# agregado por temporada/competicao), que e exatamente o que o modelo precisa.
def parse_desempenho(dados: dict) -> list[dict]:
    """Achata a resposta da API em uma linha por partida do jogador."""
    partidas = dados.get("data", {}).get("performance", [])
    registros: list[dict] = []

    for partida in partidas:
        jogo = partida.get("gameInformation", {})
        clubes = partida.get("clubsInformation", {})
        stats = partida.get("statistics", {})
        geral = stats.get("generalStatistics", {})
        tempo = stats.get("playingTimeStatistics", {})

        registros.append(
            {
                "game_id": jogo.get("gameId"),
                "data": (jogo.get("date") or {}).get("dateTimeUTC", "")[:10],
                "temporada": (jogo.get("season") or {}).get("nonCyclicalName"),
                "season_id": jogo.get("seasonId"),
                "competicao": jogo.get("competitionId"),
                "clube_id": (clubes.get("club") or {}).get("clubId"),
                "adversario_id": (clubes.get("opponent") or {}).get("clubId"),
                "local": (clubes.get("club") or {}).get("venue"),
                # participacao: 'played', 'on the bench', 'injured', 'not in squad', ...
                "participacao": geral.get("participationState"),
                "minutos": tempo.get("playedMinutes"),
                "titular": tempo.get("isStarting"),
                "idade": geral.get("age"),
                "injury_id": geral.get("injuryId"),
            }
        )

    # mais recente primeiro (a API ja devolve nessa ordem, mas garantimos)
    registros.sort(key=lambda r: r["data"] or "", reverse=True)
    return registros


def resumo_por_temporada(registros: list[dict]) -> list[dict]:
    """Agrega minutos/jogos por temporada, para comparar com o site a olho nu."""
    acumulado: dict[str, dict] = {}
    for registro in registros:
        chave = registro["temporada"] or "?"
        linha = acumulado.setdefault(
            chave, {"temporada": chave, "partidas_com_minutos": 0, "minutos": 0, "fora_por_lesao": 0}
        )
        minutos = registro["minutos"] or 0
        if minutos:
            linha["partidas_com_minutos"] += 1
            linha["minutos"] += minutos
        if registro["participacao"] == "injured":
            linha["fora_por_lesao"] += 1
    return sorted(acumulado.values(), key=lambda linha: linha["temporada"], reverse=True)


# ---------------------------------------------------------------------------
# Montagem das URLs a partir da URL de perfil colada pelo usuario
# ---------------------------------------------------------------------------
def identifica_jogador(url: str) -> tuple[str, str]:
    """Extrai (player_id, slug) de qualquer URL de jogador do Transfermarkt."""
    achado_id = re.search(r"/spieler/(\d+)", url)
    if not achado_id:
        sys.exit("URL invalida: nao encontrei o trecho '/spieler/<id>'.")

    achado_slug = re.search(r"transfermarkt\.[^/]+/([^/]+)/(?:profil|verletzungen|leistungsdaten)/", url)
    slug = achado_slug.group(1) if achado_slug else "-"
    return achado_id.group(1), slug


def nome_do_jogador(html: str) -> str:
    """Le o nome exibido no cabecalho da pagina, so para conferencia visual."""
    soup = BeautifulSoup(html, "lxml")
    titulo = soup.find("h1", class_="data-header__headline-wrapper")
    return titulo.get_text(" ", strip=True) if titulo else "(nome nao encontrado)"


# ---------------------------------------------------------------------------
# Impressao
# ---------------------------------------------------------------------------
def imprime(titulo: str, registros: list[dict], limite: int | None = None) -> None:
    print(f"\n{'=' * 78}\n{titulo}  ({len(registros)} registros)\n{'=' * 78}")
    if not registros:
        print(">>> NADA EXTRAIDO — conferir seletor ou possivel bloqueio do site.")
        return
    print(f"colunas: {list(registros[0].keys())}\n")
    mostrar = registros if limite is None else registros[:limite]
    for i, registro in enumerate(mostrar, start=1):
        print(f"{i:>3}. {registro}")
    if limite is not None and len(registros) > limite:
        print(f"... (+{len(registros) - limite} registros omitidos)")


def main() -> None:
    parser = argparse.ArgumentParser(description="Fase 0: teste de scraping de 1 jogador")
    parser.add_argument("url", help="URL do perfil do jogador no Transfermarkt")
    parser.add_argument("--no-cache", action="store_true", help="forca novo download")
    parser.add_argument("--no-pages", action="store_true", help="ignora paginacao de lesoes")
    parser.add_argument("--limite", type=int, default=None, help="limita registros impressos")
    args = parser.parse_args()

    usa_cache = not args.no_cache
    player_id, slug = identifica_jogador(args.url)
    print(f"player_id={player_id}  slug={slug}")

    # ---- aba de lesoes -----------------------------------------------------
    url_lesoes = f"{BASE}/{slug}/verletzungen/spieler/{player_id}"
    html_lesoes = fetch(url_lesoes, f"{player_id}_verletzungen", usa_cache)
    print(f"jogador: {nome_do_jogador(html_lesoes)}")

    lesoes = parse_tabela_items(html_lesoes)
    if not args.no_pages:
        for pagina in paginas_de_lesao(html_lesoes):
            time.sleep(random.uniform(1, 3))  # rate limiting
            html_pagina = fetch(
                f"{url_lesoes}/plus/1/page/{pagina}",
                f"{player_id}_verletzungen_p{pagina}",
                usa_cache,
            )
            lesoes.extend(parse_tabela_items(html_pagina))

    imprime("LESOES", lesoes, args.limite)

    # ---- aba de desempenho (uma linha por partida, via API JSON) -----------
    time.sleep(random.uniform(1, 3))  # rate limiting
    dados = fetch_json(
        f"{BASE}/ceapi/performance-game/{player_id}",
        f"{player_id}_performance",
        usa_cache,
    )
    desempenho = parse_desempenho(dados)
    imprime(
        "DESEMPENHO (uma linha por partida, mais recentes primeiro)",
        desempenho,
        args.limite if args.limite is not None else 10,
    )

    if desempenho:
        estados = Counter(r["participacao"] for r in desempenho)
        print(f"\nestados de participacao: {dict(estados)}")
        imprime("RESUMO POR TEMPORADA (para conferir com o site)", resumo_por_temporada(desempenho))

    print(f"\nDados brutos salvos em: {RAW_DIR}")


if __name__ == "__main__":
    main()
