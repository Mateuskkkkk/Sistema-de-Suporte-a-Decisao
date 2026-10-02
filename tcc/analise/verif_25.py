import openpyxl, glob, json, numpy as np, pandas as pd, os
D='/tmp/claude-0/-home-user-Sistema-de-Suporte-a-Decisao/2289c1dd-334d-5daf-aca8-58cbfaaa581c/scratchpad/x/exp/exportacoes'
K=2.592; MES=['JAN','FEV','MAR','ABR','MAI','JUN','JUL','AGO','SET','OUT','NOV','DEZ']
rows=[]
for f in sorted(glob.glob(D+'/*.xlsx')):
    wb=openpyxl.load_workbook(f,data_only=True); ws=wb['Parâmetros e notas']
    p={}; curvas=[]
    for r in range(1,ws.max_row+1):
        k=ws.cell(r,1).value
        if k in ('VM1','VM2','VM3'):
            curvas.append(dict(curva=k,estado=ws.cell(r,2).value,rac=float(ws.cell(r,3).value),lim=[float(ws.cell(r,c).value) for c in range(6,18)]))
        elif k: p[k]=ws.cell(r,2).value
    nome=p['Hidrossistema']
    if p['Status da execução']!='Executado':
        rows.append(dict(n=os.path.basename(f)[:2],nome=nome,exec=False,motivo=p.get('Bloqueio'))); continue
    cap=float(p['Capacidade no simulador (hm³)'])
    df=pd.read_excel(f,sheet_name='Resultados')
    Vi=df['Armazenamento Inicial (hm³)'].values; Vf=df['Armazenamento Final (hm³)'].values
    I=df['Afluências (hm³/mês)'].values; E=df['Evaporação (hm³)'].values; S=df['Vertimento (hm³)'].values
    R=df['Demanda Atendida (hm³)'].values; Rq=df['Demanda Atendida (m³/s)'].values*K
    eps=Vf-(Vi+I-R-E-S)
    cont=np.abs(Vi[1:]-Vf[:-1]).max()
    lim_viol=int(((Vf< -1e-9)|(Vf>cap+1e-9)|(Vi<-1e-9)|(Vi>cap+1e-9)).sum())
    mes=df['Mês/Ano'].astype(str).str[5:7].astype(int).values-1
    pct=Vi/cap*100
    # classificação esperada: menor limite >= pct
    ordem=sorted(curvas,key=lambda c:0)  # VM3 < VM2 < VM1 em geral; aplica-se a mesma regra do motor: ordena por limite do mês
    est_esp=[];rac_esp=[]
    for t in range(len(df)):
        regras=sorted([(c['lim'][mes[t]],c['rac'],c['estado']) for c in curvas])
        e,r='Normal',0.0
        for lim,rc,st in regras:
            if pct[t]<=lim: e,r=st,rc; break
        est_esp.append(e); rac_esp.append(r)
    div_est=int((np.array(est_esp)!=df['Modo Operação'].values).sum())
    div_rac=int((np.abs(np.array(rac_esp)-df['Racionamento (%)'].values)>1e-9).sum())
    apl=df['Demanda Solicitada (m³/s)'].values*(1-df['Racionamento (%)'].values/100)
    atd=df['Demanda Atendida (m³/s)'].values
    excesso=int((atd>apl+1e-9).sum())
    falha_esp=np.where(np.round(atd*K,6)<np.round(apl*K,6),'Sim','Não')
    div_falha=int((falha_esp!=df['Falha'].values).sum())
    est=df['Modo Operação'].value_counts()
    rows.append(dict(n=os.path.basename(f)[:2],nome=nome,exec=True,cenario=p['Cenário/Regra'],periodo=p['Período'],meses=len(df),cap=cap,q=p['Q normal base (L/s)'],
        rac='/'.join(f"{c['rac']:.0f}" for c in curvas),
        eps=float(np.abs(eps).max()),epsR=float(np.abs(R-Rq).max()),cont=float(cont),lim=lim_viol,div_est=div_est,div_rac=div_rac,excesso=excesso,div_falha=div_falha,
        falhas=int((df['Falha']=='Sim').sum()),vmin_pct=float(Vf.min()/cap*100),
        N=int(est.get('Normal',0)),A=int(est.get('Alerta',0)),Se=int(est.get('Seca',0)),SS=int(est.get('Seca Severa',0))))
json.dump(rows,open('verif_25.json','w'),indent=1)
t=pd.DataFrame(rows); pd.set_option('display.width',250); print(t.to_string())
ok=t[t.exec==True]
print('total meses',ok.meses.sum(),'eps max',ok.eps.max(),'cont',ok.cont.max(),'lim',ok.lim.sum(),'div est',ok.div_est.sum(),'div rac',ok.div_rac.sum(),'exc',ok.excesso.sum(),'divf',ok.div_falha.sum())
