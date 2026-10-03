"""Executa a planilha independente para os 25 hidrossistemas (Q normal, V0 = 50%, sem racionamento)."""
import json, os, shutil, uno, sqlite3, sys
from planilha_uno import conectar, prop, SRC
sys.path.insert(0, "/home/user/Sistema-de-Suporte-a-Decisao/backend")
OUT = "/tmp/lo_work/todos"; os.makedirs(OUT, exist_ok=True)
MES = {'JAN':1,'FEV':2,'MAR':3,'ABR':4,'MAI':5,'JUN':6,'JUL':7,'AGO':8,'SET':9,'OUT':10,'NOV':11,'DEZ':12}
def corrigir(v):
    try: return v.encode('latin1').decode('utf-8') if any(m in v for m in ('Ã','Â')) else v
    except Exception: return v
cfg = json.load(open('cfg_todos.json'))
db = sqlite3.connect('/home/user/Sistema-de-Suporte-a-Decisao/backend/banco_site.db')
proc, desktop = conectar(); res = {}
try:
    for c in cfg:
        p0 = c['ini'][1]*12+MES[c['ini'][0]]; p1 = c['fim'][1]*12+MES[c['fim'][0]]
        serie = {}
        for n, a, m, q in db.execute('select * from vazoes'):
            if corrigir(str(n)) != c['banco']: continue
            mm = MES.get(corrigir(str(m)).upper()[:3]); a = int(float(a))
            if mm and p0 <= a*12+mm <= p1: serie[(a, mm)] = float(q)
        dst = os.path.join(OUT, f"Verificacao_{c['arq'].replace('.xlsx','')}.xlsx"); shutil.copy(SRC, dst)
        doc = desktop.loadComponentFromURL(uno.systemPathToFileUrl(dst), "_blank", 0, (prop("Hidden", True), prop("FilterName", "Calc MS Excel 2007 XML")))
        sh = doc.Sheets.getByName("Vazoes_COLAR")
        for r in range(1, 113):
            for col in range(1, 13): sh.getCellByPosition(col, r).setString("")
        for (a, mm), q in serie.items(): sh.getCellByPosition(mm, a-1910+1).setValue(q)
        v = doc.Sheets.getByName("Verificacao")
        v.getCellRangeByName("B6").setString(c['banco'])
        v.getCellRangeByName("B12").setValue(c['q']/1000.0)
        v.getCellRangeByName("B10").setValue(50)
        doc.calculateAll()
        ind = {k: v.getCellRangeByName(x).getValue() for k, x in dict(meses="E6", falhas="E7", atend="E8", vert="E9", evap="E10", resmax="E11", vmin="E12", vmax="E13", cap="B8", cod="B7").items()}
        ind["situacao"] = v.getCellRangeByName("E14").getString(); ind["limites"] = v.getCellRangeByName("E17").getString()
        dados = v.getCellRangeByName("A24:AC1367").getDataArray()
        linhas = [list(l[:26]) + [l[21]] for l in dados if l[21] in ('Sim', 'Não')]
        doc.storeToURL(uno.systemPathToFileUrl(dst), (prop("FilterName", "Calc MS Excel 2007 XML"),)); doc.close(True)
        res[c['arq']] = dict(ind=ind, linhas=linhas, n_serie=len(serie)); print(c['nome'], ind, len(linhas), flush=True)
finally:
    try: desktop.terminate()
    except Exception: pass
    proc.wait(timeout=60)
json.dump(res, open('resultados_planilha_todos.json', 'w'))
