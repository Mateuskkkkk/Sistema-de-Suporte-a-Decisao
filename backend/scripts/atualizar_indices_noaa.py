"""Calcula os índices Niño 3.4, TNA e TSA a partir da grade ERSST v5 e grava no banco.

Uso (na pasta backend):
    python scripts/atualizar_indices_noaa.py [--inicio 1900] [--fim ANO] [--conferir]

O script baixa os arquivos anuais em ASCII da ERSST v5 (NOAA NCEI), um por ano,
guarda-os em data/ersst_v5_cache/ e calcula, para cada mês, a média da
temperatura da superfície do mar em cada região, ponderada pelo cosseno da
latitude e só com células de oceano. As anomalias são em relação à média mensal
de 1991 a 2020. O resultado vai para a tabela indices_climaticos (ano, mes,
indice, valor); a origem dos dados, as regiões e o hash dos arquivos vão para
indices_climaticos_fonte.

O backend só lê a tabela: nada é baixado durante as requisições.

Com --conferir, o script baixa os índices prontos da NOAA PSL e informa a
correlação e o erro quadrático médio no período comum, como conferência do
cálculo.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import sqlite3
import sys
import urllib.request
from pathlib import Path

import numpy as np

BACKEND = Path(__file__).resolve().parents[1]
DB_PADRAO = BACKEND / "banco_site.db"
CACHE_PADRAO = BACKEND / "data" / "ersst_v5_cache"
URL_ERSST = "https://www.ncei.noaa.gov/pub/data/cmb/ersst/v5/ascii/ersst.v5.{ano}.asc"
FONTE = "NOAA NCEI Extended Reconstructed SST v5 (ERSST v5), arquivos anuais em ASCII"
BASE = (1991, 2020)

# Grade da ERSST v5: latitudes -88 a 88 e longitudes 0 a 358, de 2 em 2 graus
# (centros das células). Uma célula entra na região quando o centro está dentro
# dos limites, inclusive.
LATS = np.arange(-88.0, 88.0 + 1e-9, 2.0)
LONS = np.arange(0.0, 358.0 + 1e-9, 2.0)

REGIOES = {
    # Niño 3.4: 5°N–5°S, 170°W–120°W
    "nino34": {"lat": (-5.0, 5.0), "lon": [(190.0, 240.0)],
               "descricao": "Niño 3.4: 5°S–5°N, 170°W–120°W"},
    # TNA (Enfield et al., 1999): 5,5°N–23,5°N, 57,5°W–15°W
    "tna": {"lat": (5.5, 23.5), "lon": [(302.5, 345.0)],
            "descricao": "Atlântico Tropical Norte: 5,5°N–23,5°N, 57,5°W–15°W"},
    # TSA (Enfield et al., 1999): 0–20°S, 30°W–10°E
    "tsa": {"lat": (-20.0, 0.0), "lon": [(330.0, 360.0), (0.0, 10.0)],
            "descricao": "Atlântico Tropical Sul: 20°S–0°, 30°W–10°E"},
}

# Referências para conferência: os CSVs da NOAA PSL guardados no repositório
# (Niño 3.4 da ERSST, TNA e TSA da HadISST) e o Niño 3.4 longo da HadISST na PSL.
REFERENCIAS_LOCAIS = {"nino34": "nino34.csv", "tna": "tna.csv", "tsa": "tsa.csv"}
URLS_PSL = {
    "nino34": "https://psl.noaa.gov/data/timeseries/month/data/nino34.long.anom.csv",
    "tna": "https://psl.noaa.gov/data/correlation/tna.data",
    "tsa": "https://psl.noaa.gov/data/correlation/tsa.data",
}
RELATORIO = BACKEND / "resultados" / "previsao_knn" / "conferencia_indices.txt"


def mascara_regiao(regiao: dict) -> np.ndarray:
    """Máscara (180 longitudes x 89 latitudes) das células da região."""
    lat_ok = (LATS >= regiao["lat"][0]) & (LATS <= regiao["lat"][1])
    lon_ok = np.zeros_like(LONS, dtype=bool)
    for a, b in regiao["lon"]:
        lon_ok |= (LONS >= a) & (LONS <= b)
    return lon_ok[:, None] & lat_ok[None, :]


def baixar(ano: int, cache: Path) -> Path:
    destino = cache / f"ersst.v5.{ano}.asc"
    ano_corrente = dt.date.today().year
    # o arquivo do ano corrente e do anterior muda a cada mês; os demais ficam em cache
    if destino.exists() and ano < ano_corrente - 1:
        return destino
    url = URL_ERSST.format(ano=ano)
    with urllib.request.urlopen(url, timeout=120) as resp:
        conteudo = resp.read()
    destino.write_bytes(conteudo)
    return destino


def ler_ersst(caminho: Path) -> np.ndarray:
    """Matriz (12, 180, 89) em °C, com NaN onde não há dado (terra ou mês futuro)."""
    valores = np.array(caminho.read_text().split(), dtype=np.int32)
    if valores.size % (180 * 89) != 0:
        raise ValueError(f"Tamanho inesperado em {caminho.name}: {valores.size} valores.")
    meses = valores.size // (180 * 89)
    grade = valores.reshape(meses, 180, 89).astype(float)
    grade[grade <= -900] = np.nan
    grade /= 100.0
    if meses < 12:
        grade = np.concatenate([grade, np.full((12 - meses, 180, 89), np.nan)])
    return grade


def media_regional(grade: np.ndarray, mascara: np.ndarray) -> np.ndarray:
    """Média mensal ponderada pelo cosseno da latitude; NaN se a região não tem dado."""
    peso = np.cos(np.deg2rad(LATS))[None, :] * mascara
    resultado = np.full(grade.shape[0], np.nan)
    for m in range(grade.shape[0]):
        campo = grade[m]
        valido = np.isfinite(campo) & (peso > 0)
        # a máscara de terra da ERSST é fixa; um mês sem nenhuma célula válida
        # é um mês ainda não publicado
        if not valido.any():
            continue
        resultado[m] = np.sum(campo[valido] * peso[valido]) / np.sum(peso[valido])
    return resultado


def ler_psl(url: str) -> tuple[dict, str]:
    """Lê um índice da PSL em CSV (data, valor) ou no formato de 13 colunas por ano."""
    with urllib.request.urlopen(url, timeout=60) as resp:
        linhas = resp.read().decode("utf-8", "ignore").splitlines()
    return interpretar_indice(linhas)


def interpretar_indice(linhas: list[str]) -> tuple[dict, str]:
    dados, descricao = {}, ""
    for linha in linhas:
        if "," in linha:
            data, _, valor = linha.partition(",")
            try:
                ano, mes, v = int(data[:4]), int(data[5:7]), float(valor)
            except ValueError:
                descricao = descricao or linha.strip()
                continue
            if v > -90:
                dados[(ano, mes)] = v
            continue
        partes = linha.split()
        if len(partes) == 13:
            try:
                ano, vals = int(partes[0]), [float(v) for v in partes[1:]]
            except ValueError:
                continue
            for m, v in enumerate(vals, start=1):
                if v > -90:
                    dados[(ano, m)] = v
        elif partes and not partes[0].lstrip("-").replace(".", "").isdigit():
            descricao = (descricao + " " + linha.strip()).strip()
    return dados, descricao[:160]


def comparar(nosso: dict, ref: dict, periodo: tuple[int, int] | None = None) -> str:
    comuns = sorted(k for k in ref if k in nosso and (periodo is None or periodo[0] <= k[0] <= periodo[1]))
    a = np.array([nosso[k] for k in comuns])
    b = np.array([ref[k] for k in comuns])
    r = np.corrcoef(a, b)[0, 1]
    # as bases das anomalias diferem; o erro é calculado sem a média de cada série
    reqm = float(np.sqrt(np.mean(((a - a.mean()) - (b - b.mean())) ** 2)))
    return f"{comuns[0][0]}–{comuns[-1][0]}, n = {len(comuns)}, r = {r:.3f}, REQM sem viés = {reqm:.3f} °C"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--inicio", type=int, default=1900)
    ap.add_argument("--fim", type=int, default=dt.date.today().year)
    ap.add_argument("--db", default=str(DB_PADRAO))
    ap.add_argument("--cache", default=str(CACHE_PADRAO))
    ap.add_argument("--conferir", action="store_true", help="compara com os índices prontos da NOAA PSL")
    args = ap.parse_args()
    if args.inicio > BASE[0] or args.fim < BASE[1]:
        print(f"O período precisa conter a base {BASE[0]}–{BASE[1]}.", file=sys.stderr)
        return 2

    cache = Path(args.cache)
    cache.mkdir(parents=True, exist_ok=True)
    anos = list(range(args.inicio, args.fim + 1))
    mascaras = {nome: mascara_regiao(reg) for nome, reg in REGIOES.items()}
    brutos = {nome: {} for nome in REGIOES}
    hash_total = hashlib.sha256()
    for ano in anos:
        caminho = baixar(ano, cache)
        hash_total.update(caminho.read_bytes())
        grade = ler_ersst(caminho)
        for nome, mascara in mascaras.items():
            serie = media_regional(grade, mascara)
            for m in range(12):
                if np.isfinite(serie[m]):
                    brutos[nome][(ano, m + 1)] = float(serie[m])
        print(f"\r{ano}", end="", file=sys.stderr)
    print(file=sys.stderr)

    anomalias = {}
    for nome, serie in brutos.items():
        clim = {}
        for m in range(1, 13):
            vals = [serie[(a, m)] for a in range(BASE[0], BASE[1] + 1) if (a, m) in serie]
            if len(vals) != BASE[1] - BASE[0] + 1:
                raise RuntimeError(f"Base incompleta para {nome}, mês {m}.")
            clim[m] = float(np.mean(vals))
        anomalias[nome] = {k: v - clim[k[1]] for k, v in serie.items()}

    con = sqlite3.connect(args.db)
    with con:
        con.execute(
            """CREATE TABLE IF NOT EXISTS indices_climaticos (
                   ano INTEGER NOT NULL, mes INTEGER NOT NULL, indice TEXT NOT NULL, valor REAL NOT NULL,
                   PRIMARY KEY (ano, mes, indice))"""
        )
        con.execute(
            """CREATE TABLE IF NOT EXISTS indices_climaticos_fonte (
                   indice TEXT PRIMARY KEY, descricao TEXT, fonte TEXT, url TEXT, unidade TEXT,
                   periodo_base TEXT, primeiro_mes TEXT, ultimo_mes TEXT, atualizado_em TEXT, sha256_arquivos TEXT)"""
        )
        agora = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        for nome, serie in anomalias.items():
            con.execute("DELETE FROM indices_climaticos WHERE indice = ?", (nome,))
            con.executemany(
                "INSERT INTO indices_climaticos (ano, mes, indice, valor) VALUES (?, ?, ?, ?)",
                [(a, m, nome, round(v, 4)) for (a, m), v in sorted(serie.items())],
            )
            chaves = sorted(serie)
            con.execute(
                "INSERT OR REPLACE INTO indices_climaticos_fonte VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (nome, REGIOES[nome]["descricao"], FONTE, URL_ERSST.format(ano="AAAA"), "°C (anomalia)",
                 f"{BASE[0]}–{BASE[1]}", f"{chaves[0][0]}-{chaves[0][1]:02d}", f"{chaves[-1][0]}-{chaves[-1][1]:02d}",
                 agora, hash_total.hexdigest()),
            )
    for nome, serie in anomalias.items():
        chaves = sorted(serie)
        print(f"{nome}: {len(serie)} meses, de {chaves[0][0]}-{chaves[0][1]:02d} a {chaves[-1][0]}-{chaves[-1][1]:02d}")

    if args.conferir:
        linhas = [f"Conferência dos índices calculados da ERSST v5 ({agora})", ""]
        for nome in REGIOES:
            local = BACKEND / "data" / "indices_climaticos" / REFERENCIAS_LOCAIS[nome]
            if local.exists():
                ref, desc = interpretar_indice(local.read_text(encoding="utf-8", errors="ignore").splitlines())
                linhas.append(f"{nome} × {local.name} do repositório [{desc}]: {comparar(anomalias[nome], ref)}")
            ref, desc = ler_psl(URLS_PSL[nome])
            linhas.append(f"{nome} × {URLS_PSL[nome]} [{desc}]: {comparar(anomalias[nome], ref)}")
            if min(ref)[0] < 1940:
                for periodo in ((max(min(ref)[0], args.inicio), 1949), (1950, max(ref)[0])):
                    linhas.append(f"    subperíodo {periodo[0]}–{periodo[1]}: {comparar(anomalias[nome], ref, periodo)}")
        RELATORIO.parent.mkdir(parents=True, exist_ok=True)
        RELATORIO.write_text("\n".join(linhas) + "\n", encoding="utf-8")
        print("\n".join(linhas))
    return 0


if __name__ == "__main__":
    sys.exit(main())
