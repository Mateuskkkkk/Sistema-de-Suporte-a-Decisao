from simlib import *
import json
P=json.load(open('resultados_planilha.json'))
out={}
for arq,dem in [("Verificacao_Mundau_250Ls.xlsx",250),("Verificacao_Mundau_500Ls.xlsx",500)]:
    L=[l for l in P[arq]['linhas'] if l[26] in ('Sim','Não')]
    pl=pd.DataFrame({'Vi':[l[7] for l in L],'I':[l[4] for l in L],'E':[l[14] for l in L],'R':[l[16] for l in L],'S':[l[19] for l in L],'Vf':[l[20] for l in L],'F':[l[26] for l in L],'res':[l[22] for l in L]})
    d=simular([res(61,50,dem)])["Mundaú"]
    s=pd.DataFrame({'Vi':d['Armazenamento Inicial'],'I':d['Afluências (hm³/mês)'],'E':d['Evaporação (hm³)'],'R':d['Demanda Atendida (m³/s)']*K,'S':d['Vertimento (hm³)'],'Vf':d['Armazenamento Final'],'F':d['Falha']})
    assert len(pl)==len(s), (len(pl),len(s))
    dif={c:float((pl[c]-s[c]).abs().max()) for c in ['Vi','I','E','R','S','Vf']}
    fal=int((pl.F!=s.F).sum())
    ind=indicadores(d,21.3)
    out[dem]=dict(dif=dif,div_falha=fal,plan=P[arq]['res'],sim=dict(falhas=ind['falhas'],atend=ind['atend_aplicada_pct'],vert=ind['vert_hm3'],evap=ind['evap_hm3'],vmin=ind['vmin_hm3'],res=float(residuo(d).abs().max())),
                  soma_plan=dict(I=pl.I.sum(),E=pl.E.sum(),R=pl.R.sum(),S=pl.S.sum()),soma_sim=dict(I=s.I.sum(),E=s.E.sum(),R=s.R.sum(),S=s.S.sum()))
    print(dem, json.dumps(out[dem],indent=1,default=float))
json.dump(out,open('verif_planilha.json','w'),default=float)
