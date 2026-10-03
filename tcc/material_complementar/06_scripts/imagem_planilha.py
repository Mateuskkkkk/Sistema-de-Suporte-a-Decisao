"""Exporta para PDF um recorte da aba Verificacao da planilha preenchida (Mundaú, 500 L/s)."""
import shutil, uno
from com.sun.star.beans import PropertyValue
from planilha_uno import conectar, prop
SRC = "/tmp/lo_work/Verificacao_Mundau_500Ls.xlsx"; DST = "/tmp/lo_work/img_planilha.xlsx"; shutil.copy(SRC, DST)
proc, desktop = conectar()
try:
    doc = desktop.loadComponentFromURL(uno.systemPathToFileUrl(DST), "_blank", 0, (prop("Hidden", True), prop("FilterName", "Calc MS Excel 2007 XML")))
    doc.calculateAll()
    sh = doc.Sheets.getByName("Verificacao")
    # oculta colunas auxiliares (Z em diante) e linhas de meses além do recorte
    cols = sh.getColumns()
    for c in range(25, 40):
        cols.getByIndex(c).IsVisible = False
    cols.getByIndex(0).Width = 4600; cols.getByIndex(1).Width = 2700
    cols.getByIndex(3).Width = 3900
    for c in range(4, 25):
        cols.getByIndex(c).Width = 2450
    rows = sh.getRows()
    for r in range(23, 35):
        rows.getByIndex(r).IsVisible = False
    rows.getByIndex(22).Height = 1550
    hdr = sh.getCellRangeByName("A23:Y23"); hdr.IsTextWrapped = True
    estilo = doc.StyleFamilies.getByName("PageStyles").getByName(sh.PageStyle)
    estilo.IsLandscape = True; w, h = estilo.Width, estilo.Height
    if w < h: estilo.Width, estilo.Height = h, w
    estilo.ScaleToPagesX = 1; estilo.ScaleToPagesY = 1
    estilo.LeftMargin = estilo.RightMargin = estilo.TopMargin = estilo.BottomMargin = 500
    estilo.HeaderIsOn = False; estilo.FooterIsOn = False
    rng = sh.getCellRangeByName("A1:Y58")
    fd = uno.Any("[]com.sun.star.beans.PropertyValue", (prop("Selection", rng),))
    doc.storeToURL(uno.systemPathToFileUrl("/tmp/lo_work/planilha_verificacao.pdf"),
                   (prop("FilterName", "calc_pdf_Export"), prop("FilterData", fd)))
    doc.close(True)
finally:
    try: desktop.terminate()
    except Exception: pass
    proc.wait(timeout=60)
print("ok")
