"""Executa a planilha independente para Flor do Campo (55 e 110 L/s, V0 = 50%, 1911–2021)."""
import json, os, shutil, uno, sqlite3, sys
from planilha_uno import conectar, prop, SRC
OUT = "/tmp/lo_work"; os.makedirs(OUT, exist_ok=True)
DB = os.environ.get("SSD_DB")
ANO_AMOSTRA = int(os.environ.get("ANO_AMOSTRA", "1953"))
db = sqlite3.connect(DB)
serie = {}
for n, a, m, q in db.execute('select * from vazoes where nome_reservatorio = ?', ("Flor do Campo",)):
    MES = {'JAN':1,'FEV':2,'MAR':3,'ABR':4,'MAI':5,'JUN':6,'JUL':7,'AGO':8,'SET':9,'OUT':10,'NOV':11,'DEZ':12}
    mm = MES.get(str(m).upper()[:3]) or (int(m) if str(m).isdigit() else None); a = int(float(a))
    if mm and 1911 <= a <= 2021: serie[(a, mm)] = float(q)
print("meses na série:", len(serie))
proc, desktop = conectar(); res = {}
try:
    for q_lps in (55, 110):
        dst = os.path.join(OUT, f"Verificacao_Flor_do_Campo_{q_lps}Ls_CAV_banco.xlsx"); shutil.copy(SRC, dst)
        doc = desktop.loadComponentFromURL(uno.systemPathToFileUrl(dst), "_blank", 0, (prop("Hidden", True), prop("FilterName", "Calc MS Excel 2007 XML")))
        # curva cota-área-volume do banco do sistema (a da planilha original difere para o código 172)
        cav = doc.Sheets.getByName("Dados_CAV")
        pontos = list(db.execute("select COTA, \"VOLUME (m³)\", \"AREA (km²)\" from cav where COD='172' order by \"VOLUME (m³)\""))
        assert cav.getCellRangeByName("A2645").getValue() == 172 and cav.getCellRangeByName("A2668").getValue() == 172
        for i in range(24):
            for col in range(5): cav.getCellByPosition(col, 2644 + i).setString("")
        for i, (cota, vol, area) in enumerate(pontos):
            for col, val in enumerate((172, cota, vol, area, vol / 1e6)): cav.getCellByPosition(col, 2644 + i).setValue(float(val))
        aco = doc.Sheets.getByName("Dados_Acudes")
        linha = [r for r in range(1, 160) if aco.getCellByPosition(0, r).getString() == "Flor do Campo"][0]
        aco.getCellByPosition(11, linha).setValue(len(pontos))
        sh = doc.Sheets.getByName("Vazoes_COLAR")
        for r in range(1, 113):
            for col in range(1, 13): sh.getCellByPosition(col, r).setString("")
        for (a, mm), q in serie.items(): sh.getCellByPosition(mm, a-1910+1).setValue(q)
        v = doc.Sheets.getByName("Verificacao")
        v.getCellRangeByName("B6").setString("Flor do Campo")
        v.getCellRangeByName("B12").setValue(q_lps/1000.0)
        v.getCellRangeByName("B10").setValue(50)
        doc.Sheets.getByName("Amostra_12_meses").getCellRangeByName("B5").setValue(ANO_AMOSTRA)
        doc.calculateAll()
        ind = {k: v.getCellRangeByName(x).getValue() for k, x in dict(meses="E6", falhas="E7", atend="E8", vert="E9", evap="E10", resmax="E11", vmin="E12", vmax="E13", cap="B8", cod="B7").items()}
        ind["situacao"] = v.getCellRangeByName("E14").getString(); ind["limites"] = v.getCellRangeByName("E17").getString()
        doc.storeToURL(uno.systemPathToFileUrl(dst), (prop("FilterName", "Calc MS Excel 2007 XML"),)); doc.close(True)
        res[q_lps] = ind; print(q_lps, ind, flush=True)
finally:
    try: desktop.terminate()
    except Exception: pass
    proc.wait(timeout=60)
json.dump(res, open(os.path.join(OUT, "resultados_planilha_flor.json"), "w"))
