from common import *
import re
vm={'Seca Severa':[21,18,16,17,21,43,42,39,36,32,28,25],'Seca':[35,32,30,31,35,58,57,54,50,47,43,39],'Alerta':[55,52,50,51,54,79,78,75,71,67,63,59]}
def nova(p):
    b=p.chromium.launch(executable_path=EXE); ctx=b.new_context(viewport={'width':1400,'height':1000},accept_downloads=True,device_scale_factor=1)
    pg=ctx.new_page(); pg.goto('http://127.0.0.1:5173'); pg.wait_for_timeout(2500); return b,pg
def exportar(pg,prefixo):
    cs=pg.locator('.recharts-responsive-container'); n=cs.count(); print(prefixo,n)
    for i in range(n):
        c=cs.nth(i); c.scroll_into_view_if_needed(); pg.wait_for_timeout(400)
        titulo=c.evaluate("e=>{let x=e; for(let k=0;k<6&&x;k++){x=x.parentElement; const h=x&&x.querySelector('h3,h4,strong,div'); } const card=e.closest('div[style*=\"border\"]')||e.parentElement.parentElement; return (card.innerText||'').split('\\n')[0].slice(0,60)}")
        c.click(button='right'); pg.wait_for_timeout(300)
        with pg.expect_download() as d:
            pg.get_by_role('menuitem',name='Exportar PNG').click()
        nome=f"{prefixo}_{i:02d}_"+re.sub('[^a-zA-Z0-9]+','_',titulo)[:40]+'.png'
        d.value.save_as('/tmp/exp/'+nome); print(nome)
        pg.wait_for_timeout(300)
if __name__=="__main__":
  with sync_playwright() as p:
      b,pg=nova(p)
      pg.select_option('select >> nth=0', label='Mundaú'); pg.wait_for_timeout(800)
      pg.get_by_role('button',name='Simulação com Níveis Meta').click(); pg.wait_for_timeout(1500)
      for nome,vals in vm.items():
          row=pg.locator(f'input[value="{nome}"]').first.locator('xpath=ancestor::tr[1]'); inps=row.locator('input')
          for k,v in enumerate(vals): inps.nth(2+k).fill(str(v))
      pg.get_by_role('button',name='Aplicar na Sessão').first.click(); pg.wait_for_timeout(800)
      pg.locator('input').nth(2).fill('250'); periodo(pg,1911,'DEZ',2021)
      pg.get_by_role('button',name='▶ Gerar Simulação').click(); pg.wait_for_timeout(6000)
      exportar(pg,'mundau')
      pg.get_by_role('button',name='Garantia').click(); pg.wait_for_timeout(1500); exportar(pg,'mundauG')
      pg.screenshot(path='/tmp/exp/mundau_garantia_tab.png',full_page=True)
      b.close()
      b,pg=nova(p)
      pg.select_option('select >> nth=0', label='Carnaubal'); pg.wait_for_timeout(1500)
      pg.locator('input').nth(7).fill('160'); periodo(pg,1911,'DEZ',2021)
      pg.get_by_role('button',name='▶ Gerar Simulação').click(); pg.wait_for_timeout(7000)
      exportar(pg,'carn'); b.close()
      b,pg=nova(p)
      pg.select_option('select >> nth=0', label='Fogareiro/Quixeramobim - PGPS Cenário 1'); pg.wait_for_timeout(1500)
      pg.get_by_role('button',name='▶ Gerar Simulação').click(); pg.wait_for_timeout(7000)
      exportar(pg,'fq'); b.close()
