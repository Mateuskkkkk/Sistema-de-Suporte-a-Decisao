from simlib import *
vm1=[55,52,50,51,54,79,78,75,71,67,63,59]; vm2=[35,32,30,31,35,58,57,54,50,47,43,39]; vm3=[21,18,16,17,21,43,42,39,36,32,28,25]
r=res(61,50,250,plano=faixas(vm1,vm2,vm3))
d=simular([r],niveis_meta=True)["Mundaú"]
print(d.columns.tolist()); print(indicadores(d,21.3)); print(residuo(d).abs().max())
import openpyxl
ex=pd.read_excel('/tmp/claude-0/-home-user-Sistema-de-Suporte-a-Decisao/2289c1dd-334d-5daf-aca8-58cbfaaa581c/scratchpad/x/exp/exportacoes/18_Mundau.xlsx',sheet_name='Resultados')
print((ex['Armazenamento Final (hm³)'].values-d['Armazenamento Final'].values).__abs__().max(), (ex['Modo Operação'].values!=d['Modo Operação'].values).sum())
