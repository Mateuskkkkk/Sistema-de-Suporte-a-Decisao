from export import nova, exportar
from common import *
def card_export(pg, titulo, aba, nome):
    card=pg.locator(f'xpath=//*[normalize-space(text())="{titulo}"]/ancestor::div[.//div[contains(@class,"recharts-responsive-container")]][1]').first
    card.get_by_role('button',name=aba,exact=True).click(); pg.wait_for_timeout(1200)
    c=card.locator('.recharts-responsive-container').first; c.scroll_into_view_if_needed(); c.click(button='right'); pg.wait_for_timeout(300)
    with pg.expect_download() as d: pg.get_by_role('menuitem',name='Exportar PNG').click()
    d.value.save_as('/tmp/exp/'+nome); print(nome)
with sync_playwright() as p:
    b,pg=nova(p)
    pg.select_option('select >> nth=0', label='Fogareiro/Quixeramobim - PGPS Cenário 1'); pg.wait_for_timeout(1500)
    pg.get_by_role('button',name='▶ Gerar Simulação').click(); pg.wait_for_timeout(7000)
    for aba in ['Fogareiro','Quixeramobim']:
        card_export(pg,'Volume Armazenado (%)',aba,f'fqV_{aba}.png')
    b.close()
    b,pg=nova(p)
    pg.select_option('select >> nth=0', label='Carnaubal'); pg.wait_for_timeout(1500)
    pg.locator('input').nth(7).fill('160'); periodo(pg,1911,'DEZ',2021)
    pg.get_by_role('button',name='▶ Gerar Simulação').click(); pg.wait_for_timeout(7000)
    for aba in ['Carnaubal','Barragem do Batalhão']:
        card_export(pg,'Demanda: Solicitada vs Atendida',aba,f'carnD_{aba.split()[0]}.png')
    b.close()
