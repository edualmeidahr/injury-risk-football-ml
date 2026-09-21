"""
Cliente HTTP compartilhado pelos scrapers do Transfermarkt.

Concentra aqui tudo que e infraestrutura (cabecalhos, cache, retry, rate limiting)
para que os scripts de scraping fiquem so com a logica de parsing.
"""

from __future__ import annotations

import json
import random
import time
from pathlib import Path

import requests

BASE = "https://www.transfermarkt.com.br"

# Raiz do projeto = pasta pai de scripts/
RAIZ = Path(__file__).resolve().parent.parent
RAW_DIR = RAIZ / "data" / "raw"
PROCESSED_DIR = RAIZ / "data" / "processed"

# User-Agent de navegador comum: o site bloqueia o UA default do requests
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "pt-BR,pt;q=0.9,en;q=0.8",
    "Referer": "https://www.transfermarkt.com.br/",
}

# Intervalo do delay aleatorio aplicado DEPOIS de cada requisicao real
# (mais conservador na Fase 2 para reduzir 403 Forbidden)
DELAY_MIN, DELAY_MAX = 2.5, 5.0


class ErroDeExtracao(RuntimeError):
    """Falha definitiva ao baixar uma pagina, depois de todas as tentativas."""


def temporada_para_saison_id(ano: int) -> int:
    """Converte o ano civil do Brasileirao no `saison_id` usado pelo Transfermarkt.

    Pegadinha importante: o site usa a convencao europeia de temporada (ano de inicio),
    entao o Brasileirao de 2024 e `saison_id=2023`. Confirmado empiricamente pela API de
    desempenho: seasonId=2023 agrupa jogos de 14/04/2024 a 08/12/2024.
    """
    return ano - 1


def _dorme() -> None:
    """Rate limiting: pausa aleatoria para nao sobrecarregar o servidor."""
    time.sleep(random.uniform(DELAY_MIN, DELAY_MAX))


def _baixa(url: str, json_esperado: bool, tentativas: int):
    """Requisicao com retry e backoff. Devolve texto ou dict, conforme o caso."""
    cabecalhos = {**HEADERS, "Accept": "application/json"} if json_esperado else HEADERS

    for tentativa in range(1, tentativas + 1):
        try:
            resp = requests.get(url, headers=cabecalhos, timeout=60)
            # 403/429: o site pediu pausa — espera mais antes de tentar de novo
            if resp.status_code in (403, 429):
                espera = 30 * tentativa + random.uniform(5, 15)
                print(
                    f"    HTTP {resp.status_code} — pausando {espera:.0f}s "
                    f"(tentativa {tentativa}/{tentativas})"
                )
                if tentativa == tentativas:
                    resp.raise_for_status()
                time.sleep(espera)
                continue
            resp.raise_for_status()
            if json_esperado:
                return resp.json()
            resp.encoding = resp.apparent_encoding or "utf-8"
            return resp.text
        except (requests.RequestException, ValueError) as erro:
            print(f"    tentativa {tentativa}/{tentativas} falhou: {erro}")
            if tentativa == tentativas:
                raise ErroDeExtracao(f"nao foi possivel baixar {url}") from erro
            time.sleep(5 * tentativa + random.random() * 3)


def get_html(url: str, cache_name: str, usa_cache: bool = True, tentativas: int = 3) -> str:
    """Baixa (ou le do cache) uma pagina HTML."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    caminho = RAW_DIR / f"{cache_name}.html"

    if usa_cache and caminho.exists():
        return caminho.read_text(encoding="utf-8")

    texto = _baixa(url, json_esperado=False, tentativas=tentativas)
    caminho.write_text(texto, encoding="utf-8")
    _dorme()
    return texto


def get_json(url: str, cache_name: str, usa_cache: bool = True, tentativas: int = 3) -> dict:
    """Baixa (ou le do cache) uma resposta da API interna (ceapi)."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    caminho = RAW_DIR / f"{cache_name}.json"

    if usa_cache and caminho.exists():
        return json.loads(caminho.read_text(encoding="utf-8"))

    dados = _baixa(url, json_esperado=True, tentativas=tentativas)
    caminho.write_text(json.dumps(dados, ensure_ascii=False), encoding="utf-8")
    _dorme()
    return dados
