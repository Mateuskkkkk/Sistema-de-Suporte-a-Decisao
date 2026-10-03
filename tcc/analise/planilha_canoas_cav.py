"""Reexecuta Canoas na planilha substituindo a curva CAV da planilha pela curva do banco do sistema."""
import json, os, shutil, uno, sqlite3
from planilha_uno import conectar, prop
SRC="/tmp/lo_work/todos/Verificacao_13_Canoas.xlsx"; DST="/tmp/lo_work/todos/Verificacao_13_Canoas_CAV_banco.xlsx"; shutil.copy(SRC,DST)
db=sqlite3.connect('/home/user/Sistema-de-Suporte-a-Decisao/backend/banco_site.db')
cav=db.execute("select COTA, \"VOLUME (m³)\", \"AREA (km²)\" from cav where COD='127'").fetchall()
proc,desktop=conectar()
try:
    doc=desktop.loadComponentFromURL(uno.systemPathToFileUrl(DST),"_blank",0,(prop("Hidden",True),prop("FilterName","Calc MS Excel 2007 XML")))
    sh=doc.Sheets.getByName("Dados_CAV")
    for k,(cota,vol,area) in enumerate(cav):
        r=2133-1+k
        for col,val in enumerate([127,cota,vol,area,vol/1e6]): sh.getCellByPosition(col,r).setValue(float(val))
    ac=doc.Sheets.getByName("Dados_Acudes")
    for r in range(1,200):
        if ac.getCellByPosition(0,r).getString()=="Canoas": ac.getCellByPosition(11,r).setValue(len(cav))
    doc.calculateAll()
    v=doc.Sheets.getByName("Verificacao")
    ind={k:v.getCellRangeByName(x).getValue() for k,x in dict(meses="E6",falhas="E7",atend="E8",vert="E9",evap="E10",resmax="E11",B18="B18",B17="B17").items()}
    dados=v.getCellRangeByName("A24:AC1367").getDataArray()
    linhas=[list(l[:26])+[l[21]] for l in dados if l[21] in ('Sim','Não')]
    doc.storeToURL(uno.systemPathToFileUrl(DST),(prop("FilterName","Calc MS Excel 2007 XML"),)); doc.close(True)
finally:
    try: desktop.terminate()
    except Exception: pass
    proc.wait(timeout=60)
print(ind)
P=json.load(open('resultados_planilha_todos.json'))
P['13_Canoas.xlsx#cav_banco']=dict(ind=ind,linhas=linhas)
json.dump(P,open('resultados_planilha_todos.json','w'))
