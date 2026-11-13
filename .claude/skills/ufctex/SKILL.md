---
name: ufctex
description: >-
  Escrever, editar, compilar e revisar trabalhos acadêmicos no modelo LaTeX da Universidade Federal do Ceará (ufcTex, derivado do abnTeX2), como TCC de graduação, monografia de especialização, dissertação ou tese. Use sempre que a tarefa mexer em um projeto com lib/ufcTex.sty, lib/abntex2.cls ou documento.tex no padrão da UFC (por exemplo a pasta tcc/), mesmo que o usuário diga só "o TCC", "a monografia", "o trabalho": incluir figura, tabela, quadro, código-fonte, equação, sigla, símbolo, apêndice ou referência; ajustar capa, folha de aprovação, banca ou ficha catalográfica; corrigir formatação ABNT; recompilar o PDF; ou conferir citações e referências.
---

# Modelo ufcTex (UFC)

O ufcTex é o modelo LaTeX da Biblioteca Universitária da UFC, adaptado do abnTeX2. Ele já aplica as normas da ABNT (NBR 14724, 6023, 6024, 6027, 10520) na formatação; o trabalho do autor é preencher o conteúdo com os comandos do modelo. Usar `\begin{figure}` + `\caption` comuns, ou montar a fonte da figura à mão, quebra o padrão de legenda e de "Fonte:" exigido pela biblioteca. Por isso, siga os padrões abaixo, copiados do que o modelo define.

## Estrutura do projeto

```
documento.tex          configuração (tipo de trabalho, autor, banca) e ordem dos elementos
latexmkrc              regra que roda makeglossaries (lista de siglas)
lib/                   abntex2.cls, ufcTex.sty, ufcTexcite.sty, abntex2-alf.bst, preambulo.tex
1-pre-textuais/        resumo, abstract, agradecimentos, lista-de-simbolos, lista-de-abreviaturas-e-siglas
2-textuais/            um arquivo por capítulo (1-introducao ... 5-conclusao)
3-pos-textuais/        referencias.bib, apendices/, anexos/
figuras/               imagens (PNG, PDF)
```

Antes de editar, leia `documento.tex` para saber o tipo de trabalho (`\trabalhoacademico{tccgraduacao|tccespecializacao|dissertacao|tese}`) e quais elementos estão ativos. Mudanças de conteúdo vão nos arquivos dos capítulos; `lib/` só se altera para corrigir o próprio modelo (e então diga ao usuário o que mudou, porque afeta todas as páginas).

## Compilar e conferir

Rode o script da skill, que compila com latexmk, resume o log e limpa os auxiliares:

```bash
bash <caminho-da-skill>/scripts/compilar.sh <pasta-do-trabalho>
```

Leia o resumo antes de entregar. O que importa:

- **Referências ou citações indefinidas**: um `\ref` ou `\cite` com rótulo errado aparece como "??" no PDF.
- **Overfull vbox**: neste modelo quase sempre é uma listagem em `minipage` que não coube na página depois de uma figura flutuante. As linhas finais somem do PDF sem erro de compilação. Corrija com a figura em `[H]` (ver abaixo) e confira a página no PDF.
- **Overfull hbox grande**: texto ou tabela saindo da margem.
- **Número de páginas**: mudanças no texto deslocam a paginação; avise o usuário se ele cita páginas em outro documento.

Não faça commit de auxiliares (`.aux`, `.bbl`, `.log`, `.toc`, `.gls` etc.); o script já os remove. Se não houver `pdfinfo` ou `latexmk`, compile com `pdflatex`, `bibtex`, `makeglossaries` e `pdflatex` duas vezes, como no Makefile do modelo.

Para conferir o resultado, extraia o texto (`pdftotext -layout documento.pdf -`) e, para figuras e páginas suspeitas, renderize a página (`pdftoppm -f N -l N -r 60 -png`) e olhe a imagem. A página impressa no rodapé costuma ser diferente da página do PDF (os pré-textuais não são numerados); ao citar páginas para o usuário, use a impressa.

## Elementos do texto

Leia `references/elementos.md` para os modelos completos de figura, tabela, quadro, código-fonte, equação, alíneas, siglas, símbolos e apêndices. O essencial:

- **Figura**: `\Caption{\label{fig:x}Título}` + `\UFCfig{}{conteúdo}{\Fonte{Elaborada pelo autor.}}`, dentro de `figure`.
- **Tabela**: `\caption` antes do `tabular`, `\Fonte{...}` e `\Nota{...}` depois; tabelas sem linhas verticais (booktabs). Dado em tabela, texto em quadro.
- **Quadro**: ambiente `quadro`, com bordas (`|` e `\hline`), para conteúdo textual.
- **Código-fonte**: `\noindent\begin{minipage}{\linewidth}` + `lstlisting` com `caption` e `label`, seguido de `\Fonte{...}`.
- **Equação**: `equation` com `\label{eq:x}`; o texto continua com "em que ..." definindo cada símbolo e sua unidade. Todo símbolo novo entra em `1-pre-textuais/lista-de-simbolos.tex`, com unidade.
- **Sigla**: `\newacronym` em `lista-de-abreviaturas-e-siglas.tex`, uso com `\gls{SIGLA}`. Uma sigla que aparece só em tabela ou figura precisa de `\glsadd{SIGLA}` em `documento.tex` para entrar na lista.
- **Remissões**: sempre `\ref`, nunca o número escrito à mão ("Equação 3.4" quebra quando outra equação é inserida antes). A exceção são números dentro de `lstlisting`, que não aceitam `\ref`; ao inserir equações, confira esses comentários.

## Citações e referências

A bibliografia usa BibTeX com o estilo `abntex2-alf.bst` e o pacote `ufcTexcite`, no sistema autor-data:

- `\cite{chave}` → (Hashimoto; Stedinger; Loucks, 1982)
- `\citeonline{chave}` → Hashimoto, Stedinger e Loucks (1982)
- `\cite[p. 45]{chave}` → com página; use página sempre que a afirmação for específica de um livro ou relatório.
- `\citeyear{chave}` → só o ano, para "Companhia ... (2023, 2026c)".

Cada entrada citada precisa estar no `.bib` e vice-versa (o BibTeX só imprime o que é citado, mas o `.bib` não deve guardar lixo). Leia `references/bibliografia.md` antes de criar ou corrigir entradas: o modelo tem campos próprios (`org-short`, `urlaccessdate`, `conference-*`, `subtitle`) e convenções para autor institucional, DOI, "[S. l.]" e "[s. n.]". A geração automática não cobre toda a NBR 6023, e a biblioteca da UFC revisa as referências; quando o estilo não produzir o formato correto, ajuste a entrada e diga ao usuário o que precisa de conferência manual.

Ao incluir uma referência nova, confira a fonte: o DOI resolve para o trabalho citado, o ISBN é válido, o link abre e não tem parâmetros de rastreamento (`utm_source` e similares). Não invente páginas, volume ou editora; se faltar dado, use as marcas da ABNT ("[S. l.]", "[s. n.]", "[20--]") e avise o que ficou pendente.

## Pré-textuais e configuração

- **Folha de aprovação**: `\dataaprovacao{...}` fica em branco (linhas) até a defesa; banca em `\membrodabancadois{}` ... `\membrodabancaseis{}`, com `\...ies{}`; membros vazios não são impressos. O orientador usa `\orientadorfeminino{sim|nao}`. O texto "Aprovado(a) em" vem de `lib/ufcTex.sty`, na variante do tipo de trabalho.
- **Ficha catalográfica**: é gerada pela biblioteca da UFC a partir dos dados do trabalho. Não a fabrique; quando o usuário tiver o PDF, coloque em `1-pre-textuais/` e ative `\imprimirfichacatalografica{...}` em `documento.tex`.
- **Resumo e abstract**: parágrafo único de 150 a 500 palavras (NBR 6028:2021), seguido de `\palavraschave{termo; termo; termo.}`, com os termos separados por ponto e vírgula e o último seguido de ponto. O abstract traduz o resumo, sem conteúdo a mais ou a menos; quando o resumo mudar, atualize o abstract.
- **Títulos**: o modelo já põe capítulos em caixa alta e seções com só a inicial maiúscula. Escreva os títulos em caixa normal ("Resultados da aplicação"); `\apendice{}` e `\anexo{}` recebem o título em maiúsculas.

## Redação

Se o repositório tiver um `CLAUDE.md` com regras de escrita, elas valem para todo texto do trabalho. Na falta dele, leia `references/redacao.md`, um resumo das regras de redação técnica usadas neste modelo. Ao entregar uma revisão, informe à parte o que mudou de fato; o texto do trabalho não deve conter comentários sobre o processo.

## Antes de entregar

1. O script de compilação não acusou erro, referência indefinida nem Overfull vbox.
2. Toda figura, tabela, quadro e código tem título, rótulo, "Fonte:" e é citado no texto antes de aparecer.
3. Símbolos e siglas novos estão nas listas; citações e referências se correspondem.
4. Resumo, abstract e conclusões continuam coerentes com os números do texto se algum resultado mudou.
5. Auxiliares removidos; só fontes `.tex`, `.bib`, figuras e o PDF vão para o commit.
