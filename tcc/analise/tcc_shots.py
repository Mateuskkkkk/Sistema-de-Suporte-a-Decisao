"""Capturas para o TCC: oculta, apenas na imagem, os botões Otimizador, Vazões e Previsão."""
from common import *
from PIL import Image
OCULTAR = """() => {
  document.querySelectorAll('header nav button').forEach(b => {
    if (['Otimizador','Vazões','Previsão'].includes(b.innerText.trim())) b.style.display = 'none';
  });
  document.querySelectorAll('header div').forEach(d => {
    if (d.children.length === 0 && d.innerText.includes('Otimização de Níveis Meta')) d.innerText = 'Simulação de Série Histórica de Reservatórios.';
  });
}"""
vm={'Seca Severa':[21,18,16,17,21,43,42,39,36,32,28,25],'Seca':[35,32,30,31,35,58,57,54,50,47,43,39],'Alerta':[55,52,50,51,54,79,78,75,71,67,63,59]}
O='/tmp/shots/out2/'; import os; os.makedirs(O,exist_ok=True)
def nova(p,h=900):
    b=p.chromium.launch(executable_path=EXE); pg=b.new_page(viewport={'width':1400,'height':h})
    pg.goto('http://127.0.0.1:5173'); pg.wait_for_timeout(2500); return b,pg
with sync_playwright() as p:
    b,pg=nova(p); pg.evaluate(OCULTAR); pg.evaluate("()=>document.activeElement && document.activeElement.blur()"); pg.mouse.move(5,5); pg.wait_for_timeout(300); pg.screenshot(path=O+'sim-tela-inicial.png'); b.close()
    b,pg=nova(p,3200)
    pg.select_option('select >> nth=0', label='Mundaú'); pg.wait_for_timeout(800)
    pg.get_by_role('button',name='Simulação com Níveis Meta').click(); pg.wait_for_timeout(1500)
    for nome,vals in vm.items():
        row=pg.locator(f'input[value="{nome}"]').first.locator('xpath=ancestor::tr[1]'); inps=row.locator('input')
        for k,v in enumerate(vals): inps.nth(2+k).fill(str(v))
    pg.get_by_role('button',name='Aplicar na Sessão').first.click(); pg.wait_for_timeout(800)
    pg.locator('input').nth(2).fill('250'); periodo(pg,1911,'DEZ',2021)
    pg.evaluate(OCULTAR); pg.evaluate("()=>document.activeElement && document.activeElement.blur()"); pg.mouse.move(5,5); pg.wait_for_timeout(300); pg.screenshot(path=O+'full_mundau.png'); b.close()
    Image.open(O+'full_mundau.png').crop((0,0,1400,820)).save(O+'sim-config-mundau.png')
    b,pg=nova(p,3200)
    pg.select_option('select >> nth=0', label='Fogareiro/Quixeramobim - PGPS Cenário 1'); pg.wait_for_timeout(1500)
    pg.evaluate(OCULTAR); pg.evaluate("()=>document.activeElement && document.activeElement.blur()"); pg.mouse.move(5,5); pg.wait_for_timeout(300); pg.screenshot(path=O+'full_fq.png'); b.close()
    Image.open(O+'full_fq.png').crop((0,0,1400,1130)).save(O+'sim-config-fq.png')
