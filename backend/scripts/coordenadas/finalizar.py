"""Consolida as coordenadas dos açudes: Wikidata + correções manuais + conferência por distância."""
import csv, json, math, re, sys, unicodedata

CASADOS, WD1, WD2, MUNCSV, SAIDA = sys.argv[1:6]

def norm(s):
    s = unicodedata.normalize("NFKD", s or "").encode("ascii", "ignore").decode().lower()
    return " ".join(re.sub(r"[^a-z0-9 ]", " ", s).split())

itens = {}
for arq in (WD1, WD2):
    for b in json.load(open(arq, encoding="utf-8"))["results"]["bindings"]:
        q = b["item"]["value"].rsplit("/", 1)[1]
        if q not in itens:
            lon, lat = map(float, re.match(r"Point\(([-\d.eE]+) ([-\d.eE]+)\)", b["coord"]["value"]).groups())
            itens[q] = dict(nome=b["itemLabel"]["value"], lat=lat, lon=lon,
                            mun=b.get("munLabel", {}).get("value") or b.get("itemDescription", {}).get("value", ""))

# correções manuais (nome com grafia diferente ou achado na segunda consulta)
MANUAL = {
    "Parambu": ("Q62012778", "busca por nome (barragem em Parambu)"),
    "Atalho": ("Q9641529", "busca por nome (Açude Atalho, Brejo Santo)"),
    "Thomás Osterne": (None, "Thomaz Osterne de Alencar"),
    "Jenipapeiro II": ("Q62015148", "município (no Wikidata: Jenipapeiro I, Baixio)"),
    "Quincoé": (None, "Quinquê"),
}
por_nome = {}
for q, w in itens.items():
    por_nome.setdefault(norm(w["nome"]), []).append(q)

sedes = {}
for r in csv.DictReader(open(MUNCSV, encoding="utf-8")):
    if r["codigo_uf"] == "23":
        sedes[norm(r["nome"])] = (float(r["latitude"]), float(r["longitude"]))
ALIAS = {"dep irapuan pinheiro": "deputado irapuan pinheiro", "mons tabosa": "monsenhor tabosa"}

def km(a, b):
    la1, lo1, la2, lo2 = map(math.radians, (a[0], a[1], b[0], b[1]))
    h = math.sin((la2 - la1) / 2) ** 2 + math.cos(la1) * math.cos(la2) * math.sin((lo2 - lo1) / 2) ** 2
    return 6371 * 2 * math.asin(math.sqrt(h))

linhas = list(csv.DictReader(open(CASADOS, encoding="utf-8")))
for l in linhas:
    if l["acude"] in MANUAL:
        q, obs = MANUAL[l["acude"]]
        if q is None:                      # procura pelo nome indicado
            q = por_nome[norm(obs)][0]
            obs = f"nome com grafia diferente ({obs})"
        w = itens[q]
        l.update(lat=w["lat"], lon=w["lon"], wikidata=q, nome_wikidata=w["nome"], municipio_wikidata=w["mun"], metodo=obs)

# uma barragem do Wikidata para um único açude: mantém o casamento pelo município
uso = {}
for l in linhas:
    if l["wikidata"]:
        uso.setdefault(l["wikidata"], []).append(l)
for q, ls in uso.items():
    if len(ls) > 1:
        manter = [l for l in ls if norm(l["municipio_wikidata"]) == norm(l["municipio"])] or ls[:1]
        for l in ls:
            if l is not manter[0]:
                l.update(lat="", lon="", wikidata="", nome_wikidata="", municipio_wikidata="", metodo="não encontrado (homônimo já usado)")

saida = []
for l in linhas:
    m = ALIAS.get(norm(l["municipio"]), norm(l["municipio"]))
    sede = sedes[m]
    if l["wikidata"]:
        d = km((float(l["lat"]), float(l["lon"])), sede)
        fonte = "Wikidata"
        situacao = "conferir" if d > 30 else "ok"
        lat, lon = float(l["lat"]), float(l["lon"])
    else:
        d, fonte, situacao = 0.0, "sede municipal (aproximada)", "aproximada"
        lat, lon = sede
    saida.append(dict(cod=l["cod"], acude=l["acude"], municipio=l["municipio"], latitude=round(lat, 6), longitude=round(lon, 6),
                      fonte=fonte, wikidata=l["wikidata"], nome_wikidata=l["nome_wikidata"], metodo=l["metodo"],
                      dist_sede_km=round(d, 1), situacao=situacao))

with open(SAIDA, "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=list(saida[0].keys()))
    w.writeheader(); w.writerows(saida)
from collections import Counter
print(Counter(s["situacao"] for s in saida))
for s in saida:
    if s["situacao"] != "ok":
        print(f'{s["acude"]:24s} {s["municipio"]:24s} {s["situacao"]:10s} {s["dist_sede_km"]:6.1f} km  {s["nome_wikidata"]}  [{s["metodo"]}]')
