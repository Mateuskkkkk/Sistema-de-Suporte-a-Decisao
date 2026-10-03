"""Preenche a planilha independente de verificação via LibreOffice (UNO), recalcula e exporta os resultados."""
import subprocess, time, sys, os, json, shutil
import uno
from com.sun.star.beans import PropertyValue
SRC = "/root/.claude/uploads/2289c1dd-334d-5daf-aca8-58cbfaaa581c/15aaa383-Planilha_Verificacao_Simulador.xlsx"
OUTDIR = "/tmp/lo_work"; os.makedirs(OUTDIR, exist_ok=True)

def prop(n, v):
    p = PropertyValue(); p.Name = n; p.Value = v; return p

def conectar():
    proc = subprocess.Popen(["soffice", "--headless", "--invisible", "--norestore",
                             "-env:UserInstallation=file:///tmp/lo_prof_tcc",
                             '--accept=socket,host=localhost,port=2002;urp;'])
    local = uno.getComponentContext()
    resolver = local.ServiceManager.createInstanceWithContext("com.sun.star.bridge.UnoUrlResolver", local)
    for _ in range(60):
        try:
            ctx = resolver.resolve("uno:socket,host=localhost,port=2002;urp;StarOffice.ComponentContext"); break
        except Exception:
            time.sleep(1)
    desktop = ctx.ServiceManager.createInstanceWithContext("com.sun.star.frame.Desktop", ctx)
    return proc, desktop

def rodar(desktop, serie, demanda_m3s, vol_pct, nome_saida, ano_amostra):
    dst = os.path.join(OUTDIR, nome_saida)
    shutil.copy(SRC, dst)
    doc = desktop.loadComponentFromURL(uno.systemPathToFileUrl(dst), "_blank", 0, (prop("Hidden", True), prop("FilterName", "Calc MS Excel 2007 XML")))
    sh = doc.Sheets.getByName("Vazoes_COLAR")
    # limpa B2:M113 e cola série (ano -> 12 valores)
    for r in range(1, 113):
        for c in range(1, 13):
            sh.getCellByPosition(c, r).setString("")
    for (ano, mes), q in serie.items():
        sh.getCellByPosition(mes, ano - 1910 + 1).setValue(q)
    v = doc.Sheets.getByName("Verificacao")
    v.getCellRangeByName("B12").setValue(demanda_m3s)
    v.getCellRangeByName("B10").setValue(vol_pct)
    doc.Sheets.getByName("Amostra_12_meses").getCellRangeByName("B5").setValue(ano_amostra)
    doc.calculateAll()
    res = {k: v.getCellRangeByName(c).getValue() for k, c in
           dict(meses="E6", falhas="E7", atend="E8", vert="E9", evap="E10", resmax="E11", vmin="E12", vmax="E13").items()}
    res["situacao"] = v.getCellRangeByName("E14").getString()
    res["limites"] = v.getCellRangeByName("E17").getString()
    linhas = []
    r = 23
    while True:
        a = v.getCellByPosition(0, r)
        if v.getCellByPosition(3, r).getString() == "" and r > 30:
            if r > 1366: break
            r += 1; continue
        if r > 1366: break
        vals = [v.getCellByPosition(c, r).getValue() for c in range(0, 26)]
        txt = v.getCellByPosition(21, r).getString()
        linhas.append(vals + [txt])
        r += 1
    amostra = []
    am = doc.Sheets.getByName("Amostra_12_meses")
    for r in range(7, 19):
        amostra.append([am.getCellByPosition(c, r).getString() if c in (0, 13) else am.getCellByPosition(c, r).getValue() for c in range(0, 15)])
    doc.calculateAll()
    doc.storeToURL(uno.systemPathToFileUrl(dst), (prop("FilterName", "Calc MS Excel 2007 XML"),))
    doc.close(True)
    return res, linhas, amostra

if __name__ == "__main__":
    cfg = json.load(open(sys.argv[1]))
    proc, desktop = conectar()
    out = {}
    try:
        for caso in cfg["casos"]:
            serie = {(int(k.split("-")[0]), int(k.split("-")[1])): q for k, q in cfg["serie"].items()}
            res, linhas, amostra = rodar(desktop, serie, caso["demanda"], caso["vol"], caso["arquivo"], caso["ano"])
            out[caso["arquivo"]] = dict(res=res, linhas=linhas, amostra=amostra)
            print(caso["arquivo"], res)
    finally:
        try: desktop.terminate()
        except Exception: pass
        proc.wait(timeout=30)
    json.dump(out, open("resultados_planilha.json", "w"))
