"""Cruza os açudes do banco do SSD com as barragens do Wikidata (nome + município)."""
import json, re, sqlite3, sys, unicodedata, difflib, csv

WD, DB, SAIDA = sys.argv[1], sys.argv[2], sys.argv[3]

def norm(s):
    s = unicodedata.normalize("NFKD", s or "").encode("ascii", "ignore").decode().lower()
    s = re.sub(r"[^a-z0-9 ]", " ", s)
    s = re.sub(r"\b(acude|barragem|reservatorio|publico|de|do|da|dos|das)\b", " ", s)
    s = s.replace("mons ", "monsenhor ").replace("dep ", "deputado ").replace("sao ", "sao ")
    return " ".join(s.split())

# barragens do Wikidata (uma por item)
itens = {}
for b in json.load(open(WD, encoding="utf-8"))["results"]["bindings"]:
    uri = b["item"]["value"]
    if uri in itens:
        continue
    lon, lat = map(float, re.match(r"Point\(([-\d.eE]+) ([-\d.eE]+)\)", b["coord"]["value"]).groups())
    itens[uri] = dict(qid=uri.rsplit("/", 1)[1], nome=b["itemLabel"]["value"], mun=b.get("munLabel", {}).get("value", ""),
                      lat=lat, lon=lon)
wd = list(itens.values())
por_nome = {}
for w in wd:
    por_nome.setdefault(norm(w["nome"]), []).append(w)
print("barragens distintas no Wikidata:", len(wd), file=sys.stderr)

db = sqlite3.connect(DB)
linhas = []
for nome, cod, mun in db.execute('select CORPO, COD, MUNICIPIO from acudes order by CORPO'):
    n, m = norm(nome), norm(mun)
    cand = por_nome.get(n, [])
    escolha, metodo = None, ""
    mesmos = [w for w in cand if norm(w["mun"]) == m]
    if len(mesmos) == 1:
        escolha, metodo = mesmos[0], "nome e município"
    elif len(cand) == 1:
        escolha, metodo = cand[0], "nome (município diferente)"
    elif len(mesmos) > 1:
        escolha, metodo = mesmos[0], f"nome e município ({len(mesmos)} homônimos)"
    else:
        # nome aproximado dentro do mesmo município
        mesmo_mun = [w for w in wd if norm(w["mun"]) == m]
        prox = difflib.get_close_matches(n, [norm(w["nome"]) for w in mesmo_mun], n=1, cutoff=0.8)
        if prox:
            escolha = next(w for w in mesmo_mun if norm(w["nome"]) == prox[0])
            metodo = "nome aproximado e município"
    linhas.append(dict(cod=cod, acude=nome, municipio=mun,
                       lat=escolha["lat"] if escolha else "", lon=escolha["lon"] if escolha else "",
                       wikidata=escolha["qid"] if escolha else "", nome_wikidata=escolha["nome"] if escolha else "",
                       municipio_wikidata=escolha["mun"] if escolha else "", metodo=metodo or "não encontrado"))

with open(SAIDA, "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=list(linhas[0].keys()))
    w.writeheader(); w.writerows(linhas)
from collections import Counter
print(Counter(l["metodo"] for l in linhas), file=sys.stderr)
