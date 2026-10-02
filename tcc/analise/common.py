from playwright.sync_api import sync_playwright
EXE='/opt/pw-browsers/chromium-1194/chrome-linux/chrome'
def periodo(pg, a0, m1, a1):
    anos = pg.locator('input[placeholder="Ano"]')
    anos.nth(0).fill(str(a0)); sels = pg.locator('select')
    n = sels.count()
    sels.nth(n-1).select_option(m1); anos.nth(1).fill(str(a1))
