import sys, json
from playwright.sync_api import sync_playwright
CHROME = "/opt/pw-browsers/chromium-1194/chrome-linux/chrome"
OUT = sys.argv[1]
MESES = ["JAN","FEV","MAR","ABR","MAI","JUN","JUL","AGO","SET","OUT","NOV","DEZ"]
FAIXAS = [("Seca Severa", "55", [7.9,7.0,6.5,6.3,6.3,5.9,18.3,16.6,14.8,12.9,11.0,9.2]),
          ("Seca", "33.33", [22.2,20.7,19.7,19.2,19.0,18.3,39.9,37.2,34.1,31.0,27.6,24.6]),
          ("Alerta", "8.33", [36.6,34.6,33.3,32.5,32.0,30.9,59.7,56.2,52.3,48.3,43.9,39.9]),
          ("Normal", "0", [100]*12)]
with sync_playwright() as p:
    b = p.chromium.launch(executable_path=CHROME)
    pg = b.new_page(viewport={"width": 1400, "height": 820}, accept_downloads=True, device_scale_factor=1)
    erros = []; pg.on("pageerror", lambda e: erros.append(str(e)))
    pg.goto("http://127.0.0.1:5174/"); pg.wait_for_timeout(2500)
    def so_simulador():
        pg.evaluate("""() => {
            for (const b of document.querySelectorAll('button')) if (['Otimizador','Vazões','Previsão'].includes(b.textContent.trim())) b.style.display = 'none';
            for (const e of document.querySelectorAll('div,span,p')) if (e.children.length === 0 && e.textContent.trim().startsWith('Simulação de Série Histórica e Otimização')) e.textContent = 'Simulação de Série Histórica de Reservatórios.';
        }""")
    busca = pg.get_by_placeholder("Digite para buscar…")
    busca.click(); busca.fill("Flor do Campo"); pg.wait_for_timeout(600)
    pg.get_by_text("Flor do Campo", exact=True).last.dispatch_event("mousedown"); pg.wait_for_timeout(600)
    nums = pg.locator("aside input[type=number], .sim-sidebar input[type=number]")
    print("inputs numéricos:", nums.count())
    # volume inicial e demanda: primeiros dois campos do cartão do reservatório
    card_inputs = pg.locator("input[type=number]")
    vals = [card_inputs.nth(i).input_value() for i in range(min(6, card_inputs.count()))]; print("valores:", vals)
    card_inputs.nth(1).fill("60")
    anos = pg.get_by_placeholder("Ano"); anos.nth(0).fill("1911"); anos.nth(1).fill("2021")
    pg.get_by_role("button", name="Simulação com Níveis Meta").click(); pg.wait_for_timeout(1200)
    pg.get_by_role("button", name="Criar Faixa Manual").click(); pg.wait_for_timeout(300)
    for _ in range(3): pg.get_by_role("button", name="Faixa", exact=True).click(); pg.wait_for_timeout(150)
    linhas = pg.locator("table tbody tr")
    print("linhas:", linhas.count())
    for i, (nome, rac, lim) in enumerate(FAIXAS):
        inps = linhas.nth(i).locator("input")
        inps.nth(0).fill(nome); inps.nth(1).fill(rac)
        for j, v in enumerate(lim): inps.nth(2 + j).fill(str(v))
    pg.get_by_role("button", name="Aplicar na Sessão").click(); pg.wait_for_timeout(800)
    pg.mouse.move(5, 5); pg.wait_for_timeout(2500); so_simulador()
    pg.screenshot(path=f"{OUT}/sim-config-flor.png")
    pg.get_by_role("button", name="Gerar Simulação").click()
    pg.wait_for_selector("text=Volume Armazenado (%)", timeout=120000); pg.wait_for_timeout(2500)
    pg.set_viewport_size({"width": 1400, "height": 1000}); pg.wait_for_timeout(2000); so_simulador()
    # cabeçalho deixa de ser fixo para não cobrir o recorte
    pg.evaluate("""() => { for (const e of document.querySelectorAll('*')) { const cs = getComputedStyle(e); if ((cs.position === 'sticky' || cs.position === 'fixed') && e.getBoundingClientRect().top < 5 && e.offsetHeight < 120) e.style.position = 'relative'; } }""")
    def recorte(nome, alvo, altura):
        alvo.scroll_into_view_if_needed(); pg.wait_for_timeout(300)
        y_doc = pg.evaluate("e => e.getBoundingClientRect().top + window.scrollY", alvo.element_handle())
        pg.evaluate(f"window.scrollTo(0, {y_doc} - 12)"); pg.wait_for_timeout(1500)
        bb = alvo.bounding_box()
        pg.screenshot(path=f"{OUT}/{nome}.png", clip={"x": bb["x"] - 8, "y": bb["y"] - 12, "width": 1060, "height": altura})
    abas = pg.get_by_role("button", name="Gráficos", exact=True).locator("xpath=..")
    recorte("sim-resultados-flor", abas, 1000)
    pg.get_by_role("button", name="Balanço Hídrico").click(); pg.wait_for_timeout(1500); so_simulador()
    card = pg.get_by_text("registros mensais").first.locator("xpath=ancestor::div[contains(@style,'border')][1]")
    recorte("sim-tabela-flor", card, 600)
    print("erros:", erros)
    b.close()
