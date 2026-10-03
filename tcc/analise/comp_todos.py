from simlib import *
import json
cfg=json.load(open('cfg_todos.json')); P=json.load(open('resultados_planilha_todos.json'))
out=[]
for c in cfg:
    L=P[c['arq']]['linhas']
    pl=pd.DataFrame({'Vi':[l[7] for l in L],'I':[l[4] for l in L],'E':[l[14] for l in L],'R':[l[16] for l in L],'S':[l[19] for l in L],'Vf':[l[20] for l in L],'F':[l[26] for l in L]})
    a=acude(c['cod']); r=dict(a, vol_inicial=a['capacidade']*0.5, demanda=c['q']/1000, gatilho=0.0, plano_secas_custom=None)
    d=list(simular([r],ini=tuple(c['ini']),fim=tuple(c['fim'])).values())[0]
    s=pd.DataFrame({'Vi':d['Armazenamento Inicial'],'I':d['Afluências (hm³/mês)'],'E':d['Evaporação (hm³)'],'R':d['Demanda Atendida (m³/s)']*K,'S':d['Vertimento (hm³)'],'Vf':d['Armazenamento Final'],'F':d['Falha']})
    assert len(pl)==len(s),(c['nome'],len(pl),len(s))
    dif=max(float((pl[k]-s[k]).abs().max()) for k in ['Vi','I','E','R','S','Vf'])
    ind=indicadores(d,a['capacidade'])
    o=dict(nome=c['nome'],periodo=f"{c['ini'][1]}--{c['fim'][1]}",padrao=c['padrao'],meses=len(s),q=c['q'],cap=a['capacidade'],
        f_pl=int(P[c['arq']]['ind']['falhas']),f_sim=ind['falhas'],div_f=int((pl.F!=s.F).sum()),
        at_pl=P[c['arq']]['ind']['atend'],at_sim=ind['atend_aplicada_pct'],dif=dif,
        res_pl=P[c['arq']]['ind']['resmax'],res_sim=float(residuo(d).abs().max()),
        evap_pl=P[c['arq']]['ind']['evap'],evap_sim=ind['evap_hm3'],vert_pl=P[c['arq']]['ind']['vert'],vert_sim=ind['vert_hm3'])
    out.append(o); print({k:(round(v,6) if isinstance(v,float) else v) for k,v in o.items()})
# exportações faltantes: simulação com níveis meta e conferências independentes
extra=[]
for c in cfg:
    if not c['padrao']: continue
    a=acude(c['cod'])
    plano=[{"Faixa":e,"Racionamento":rr,"NomeFaixaNormal":"Normal",**{m:v for m,v in zip(MESES,l)}} for e,rr,l in c['curvas']]
    r=dict(a, vol_inicial=a['capacidade']*0.5, demanda=c['q']/1000, gatilho=0.0, plano_secas_custom=plano)
    d=list(simular([r],ini=tuple(c['ini']),fim=tuple(c['fim']),niveis_meta=True).values())[0]
    cap=a['capacidade']; pct=d['Armazenamento Inicial']/cap*100; mi=d['Ordem_Mês'].values-1
    est=[];rac=[]
    for t in range(len(d)):
        e,rc='Normal',0.0
        for lim,rr,st in sorted([(cv[2][mi[t]],cv[1],cv[0]) for cv in c['curvas']]):
            if pct.iloc[t]<=lim: e,rc=st,rr; break
        est.append(e); rac.append(rc)
    apl=d['Demanda Solicitada (m³/s)']*(1-d['Racionamento (%)']/100); atd=d['Demanda Atendida (m³/s)']
    vi=d['Armazenamento Inicial'].values; vf=d['Armazenamento Final'].values
    i=indicadores(d,cap)
    x=dict(nome=c['nome'],meses=len(d),cap=cap,q=c['q'],rac='/'.join(f"{cv[1]:.0f}" for cv in c['curvas']),
      eps=float(residuo(d).abs().max()),cont=float(np.abs(vi[1:]-vf[:-1]).max()),lim=int(((vf<-1e-9)|(vf>cap+1e-9)).sum()),
      div_est=int((np.array(est)!=d['Modo Operação'].values).sum()),div_rac=int((np.abs(np.array(rac)-d['Racionamento (%)'].values)>1e-9).sum()),
      excesso=int((atd>apl+1e-9).sum()),falhas=i['falhas'],estados={k:int(v) for k,v in d['Modo Operação'].value_counts().items()})
    extra.append(x); print(x)
json.dump(dict(planilha=out,extra=extra),open('verif_todos.json','w'),indent=1,ensure_ascii=False)
print('max dif',max(o['dif'] for o in out),'div falha',sum(o['div_f'] for o in out),'meses',sum(o['meses'] for o in out))
