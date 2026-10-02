from simlib import *
import json
c=sqlite3.connect(main.DB_PATH)
rows=c.execute("select Ano, \"Mês\", \"Vazão (m³/s)\" from vazoes where nome_reservatorio in ('Mundaú',?)",(main.texto_para_legado('Mundaú'),)).fetchall()
serie={}
for a,m,q in rows:
    m=main.ordem_meses.get(main.corrigir_mojibake(str(m)).upper()[:3]); a=int(float(a))
    if 1911<=a<=2021 and m: serie[f"{a}-{m}"]=float(q)
print(len(serie))
json.dump({"serie":serie,"casos":[
  {"demanda":0.25,"vol":50,"arquivo":"Verificacao_Mundau_250Ls.xlsx","ano":1915},
  {"demanda":0.50,"vol":50,"arquivo":"Verificacao_Mundau_500Ls.xlsx","ano":1915}]},open("cfg_planilha.json","w"))
