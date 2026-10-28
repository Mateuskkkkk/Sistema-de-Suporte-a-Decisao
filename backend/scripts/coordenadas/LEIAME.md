# Coordenadas dos açudes

`backend/data/coordenadas_acudes.csv` traz latitude e longitude dos 155 açudes do banco.

- **147 açudes**: coordenada do Wikidata (barragens localizadas no Ceará, dados CC0), casadas pelo nome e pelo município.
- **8 açudes** (Barragem do Batalhão, Caldeirões, Facundo, Fogareiro, Jatobá II, Monte Belo, Prazeres, Trussu): não encontrados; posição aproximada na sede do município (`fonte = sede municipal (aproximada)`).

Colunas: `cod`, `acude`, `municipio`, `latitude`, `longitude`, `fonte`, `wikidata`, `nome_wikidata`, `metodo`,
`dist_sede_km` (distância à sede municipal), `situacao` e `municipio_do_ponto` (município em que o ponto cai).
Os 12 pontos que caem em município vizinho ao do cadastro são barragens em divisa municipal.

## Como regenerar

1. Baixar as barragens do Ceará no Wikidata (endpoint `https://query.wikidata.org/sparql`):

   ```sparql
   SELECT ?item ?itemLabel ?tipoLabel ?munLabel ?coord WHERE {
     VALUES ?classe { wd:Q12323 wd:Q131681 wd:Q23397 }
     ?item wdt:P31/wdt:P279* ?classe ; wdt:P131+ wd:Q40123 ; wdt:P625 ?coord .
     OPTIONAL { ?item wdt:P31 ?tipo . }
     OPTIONAL { ?item wdt:P131 ?mun . }
     SERVICE wikibase:label { bd:serviceParam wikibase:language "pt,en". }
   }
   ```

2. `python casar_wikidata.py wd_ce.json ../../banco_site.db casados.csv`
3. `python finalizar.py casados.csv wd_ce.json wd_faltantes.json municipios.csv ../../data/coordenadas_acudes.csv`
   (`wd_faltantes.json`: segunda consulta por nome; `municipios.csv`: sedes municipais de
   https://github.com/kelvins/municipios-brasileiros).

As correções manuais (grafias diferentes) estão no dicionário `MANUAL` de `finalizar.py`.
