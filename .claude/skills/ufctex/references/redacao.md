# Redação técnica no trabalho acadêmico

Resumo das regras de escrita usadas neste modelo. Se o repositório tiver um `CLAUDE.md` com regras próprias, elas prevalecem.

## Conteúdo
- Diga o fato concreto (número, data, quem fez o quê) em vez de afirmar importância ("marcou um momento decisivo").
- Diga o que a fonte afirma e cite-a; não fale da fonte ("recebeu ampla cobertura").
- Termine a frase no fato. Interpretação vai em frase própria, com argumento ou fonte (evite "…, evidenciando a importância de…").
- Relações exatas ("dirigiu", "financiou", "fica a jusante de"), não "esteve associado a".
- Atribuição nomeada e citada; com uma fonte só, não escreva "diversos estudos".
- Desafios, perspectivas e conclusões só onde há conteúdo específico ou onde o gênero pede (capítulo de conclusões).
- Não feche parágrafo resumindo o que acabou de dizer ("Em suma…", "Dessa forma, fica evidente…").

## Vocabulário
- Evite o acúmulo de: crucial, fundamental, essencial, robusto, valioso, rico, cenário (figurado), panorama, ressaltar, destacar, evidenciar, sublinhar, fomentar, potencializar, alavancar, aprimorar, multifacetado. Uma ocorrência isolada não é problema.
- Prefira "é" e "tem" a "serve como", "constitui", "atua como", "conta com", "dispõe de".
- Palavra comum: "usou", "fez", "analisou", não "utilizou", "efetuou", "realizou a análise".
- Repita o termo técnico; não troque por sinônimos para variar ("o açude… o reservatório… o corpo hídrico").
- Evite fórmulas de contraste ("não apenas X, mas também Y", "não é X, é Y", "em vez de X"); use no máximo uma por seção, quando o leitor de fato pensaria X.
- Liste quantos itens existirem; não force trios.
- Poucos travessões; use vírgula, dois-pontos, parênteses ou ponto. Intervalo numérico com traço meio (`1911--2021`).
- Gradação sustentada é bem-vinda: "o único", "o primeiro", "tende a", "talvez".

## Ressalvas
- Limitação concreta, uma vez, no lugar certo ("A série não inclui 2022--2024, então…"), não "é importante notar que…".
- Sem fonte, não escreva sobre o assunto ou diga que não foi verificado; não preencha com suposição.

## Formatação
- Títulos com só a primeira letra maiúscula (o modelo cuida da caixa alta dos capítulos).
- Sem negrito espalhado, sem emoji, sem listas com rótulo em negrito quando a prosa resolve.
- Nada de Markdown no `.tex` (asteriscos, `#`, `[texto](url)`).
- Sem marcadores esquecidos: `[Nome]`, `INSERIR`, `2025-xx-xx`, `example.com`.

## Parágrafos
- Parágrafo com mais de 12 a 15 linhas no PDF costuma juntar duas ideias; divida.
- Frase com mais de três orações coordenadas por ponto e vírgula costuma ficar melhor em duas ou três frases.
- Cada frase traz um fato, um dado ou um argumento; corte as que só enfatizam.

## Verificação rápida
```bash
grep -oinE "crucia|fundamenta|essencia|robust|ressalt|destac|evidenci|fomenta|potencializ|alavanc|aprimor|serve como|constitui|atua como|conta com|dispõe de|não apenas|mas também|em vez de|é importante|vale ressaltar|em suma|dessa forma|utiliz|efetu" 2-textuais/*.tex 1-pre-textuais/resumo.tex
grep -c "—" 2-textuais/*.tex
```
Confira cada ocorrência no contexto: nomes de capítulo ("Fundamentação teórica") e usos literais ("destaca os reservatórios no mapa") são legítimos.
