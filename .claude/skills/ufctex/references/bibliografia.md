# Referências no ufcTex (abntex2-alf.bst + ufcTexcite)

O estilo gera as referências pela NBR 6023:2018 e as citações autor-data pela NBR 10520:2023 (sobrenome só com a inicial maiúscula na citação, caixa alta na lista). A geração não cobre todos os casos; a biblioteca da UFC revisa a lista.

## Campos próprios do modelo

| Campo | Uso |
|---|---|
| `subtitle` | subtítulo, impresso após dois-pontos e sem negrito |
| `organization` | autor institucional, em caixa alta e entre chaves duplas: `{{COMPANHIA DE GESTÃO DOS RECURSOS HÍDRICOS}}` |
| `org-short` | forma do autor institucional na citação: `Companhia de Gest{\~a}o dos Recursos H{\'\i}dricos` |
| `urlaccessdate` | data de acesso, no formato `3 out. 2026` (mês abreviado, minúsculo) |
| `conference-number`, `conference-year`, `conference-location` | número, ano e cidade do evento em `@inproceedings` |
| `editortype` | `ed.` ou `org.` para obra organizada |
| `note` | DOI (`DOI 10.xxxx/...`), descrição de software, versão, "Apresentação de slides" |

## Modelos por tipo

Artigo:
```bibtex
@article{pontesFilho2020,
    author = {{\relax Pontes Filho}, João Dehon and {\relax Souza Filho}, F. A.},
    title = {Copula-Based Multivariate Frequency Analysis of the 2012--2018 Drought in {Northeast Brazil}},
    journal = {Water},
    address = {Basel},
    volume = {12},
    number = {3},
    pages = {834},
    month = {mar.},
    year = {2020},
    note = {DOI 10.3390/w12030834},
    url = {https://doi.org/10.3390/w12030834},
    urlaccessdate = {20 jul. 2026}
}
```

Livro e livro organizado:
```bibtex
@book{deNys2016,
    editor = {{\relax De Nys}, Erwin and Engle, Nathan L.},
    editortype = {ed.},
    title = {Secas no Brasil},
    subtitle = {política e gestão proativas},
    address = {Brasília, DF},
    publisher = {Centro de Gestão e Estudos Estratégicos; Banco Mundial},
    year = {2016},
    pages = {292},
    isbn = {978-85-5569-116-4}
}
```

Capítulo:
```bibtex
@incollection{souzaFilhoGouveia2003,
    author = {{\relax Souza Filho}, F. A. and Gouveia, Sila Xavier},
    title = {Sistemas de suporte às decisões},
    booktitle = {Gestão de águas: princípios e práticas},
    editor = {Campos, Nilson and Studart, T. M. C.},
    editortype = {ed.},
    address = {Fortaleza},
    publisher = {[s. n.]},
    pages = {87--110},
    year = {2003}
}
```

Trabalho em evento:
```bibtex
@inproceedings{lam2015numba,
    author = {Lam, Siu Kwan and Pitrou, Antoine and Seibert, Stanley},
    title = {Numba},
    subtitle = {a {LLVM}-based {Python} {JIT} compiler},
    organization = {Workshop on the LLVM Compiler Infrastructure in HPC},
    booktitle = {Proceedings [...]},
    conference-number = {2},
    conference-year = {2015},
    conference-location = {Austin},
    address = {New York},
    publisher = {Association for Computing Machinery},
    year = {2015},
    pages = {1--6},
    note = {DOI 10.1145/2833157.2833162},
    url = {https://doi.org/10.1145/2833157.2833162},
    urlaccessdate = {3 out. 2026}
}
```
Sai como "In: WORKSHOP ON ..., 2., 2015, Austin. Proceedings [...]". Em português, `booktitle = {Anais [...]}`.

Documento institucional (plano, relatório):
```bibtex
@book{pgpsMundau2026,
    organization = {{COMPANHIA DE GESTÃO DOS RECURSOS HÍDRICOS}},
    org-short = {Companhia de Gest{\~a}o dos Recursos H{\'\i}dricos},
    title = {Plano de Gestão Proativa de Secas},
    subtitle = {{Hidrossistema Mundaú}},
    address = {Fortaleza},
    publisher = {Companhia de Gestão dos Recursos Hídricos},
    year = {2026c},
    pages = {82}
}
```

Software e site:
```bibtex
@misc{python,
    organization = {{PYTHON SOFTWARE FOUNDATION}},
    org-short = {Python Software Foundation},
    title = {Python},
    address = {Wilmington, DE},
    publisher = {Python Software Foundation},
    year = {2022},
    url = {https://www.python.org},
    urlaccessdate = {3 out. 2026},
    note = {Linguagem de programação. Versão 3.11}
}
```

## Convenções

- **Sobrenomes compostos e partículas**: `{\relax Souza Filho}, F. A.` e `{\relax De Nys}, Erwin`; o `\relax` impede o BibTeX de abreviar ou reordenar.
- **Maiúsculas em títulos**: o estilo mantém a caixa do `.bib`; escreva o título como na norma (só a inicial e nomes próprios em maiúscula, em português) e proteja nomes próprios e siglas com chaves: `{Brazil}`, `{SIGA}`, `{Python}`.
- **Sem local ou editora**: `address = {[S. l.]}`, `publisher = {[s. n.]}`. Data incerta: `year = {[20--]}` ou `{[2024]}`.
- **Mesmo autor e mesmo ano**: letras no próprio `year` (`2026a`, `2026b`). As letras seguem a ordem alfabética dos títulos (e subtítulos, quando o título é igual), porque a lista sai nessa ordem. Ao inserir uma obra nova, reveja as letras das outras; as citações se atualizam sozinhas.
- **DOI**: em `note` como `DOI 10.xxxx/yyyy` e em `url` como `https://doi.org/10.xxxx/yyyy`. Confira que o DOI resolve para o trabalho citado.
- **Link**: só o endereço do documento, sem parâmetros de rastreamento (`utm_source=...`, `referrer=...`).
- **Páginas**: intervalo com `--` (`87--110`); total de páginas de livro ou relatório em `pages = {292}`.
- **Documento não publicado** (planilha cedida, comunicação): não vai na lista de referências; registre em nota de rodapé no texto, dizendo quem forneceu e quando.

## Conferência

```bash
# chaves citadas no texto
grep -ohE "\\\\cite[a-z]*(\[[^]]*\])?\{[^}]+\}" -r 2-textuais 3-pos-textuais/apendices 1-pre-textuais | sed -E 's/.*\{//; s/\}$//' | tr ',' '\n' | sed 's/ //g' | sort -u > /tmp/citadas.txt
# chaves no .bib
grep -oE "^@[a-z]+\{[^,]+" 3-pos-textuais/referencias.bib | sed 's/.*{//' | sort -u > /tmp/bib.txt
comm -23 /tmp/citadas.txt /tmp/bib.txt   # citadas sem entrada
comm -13 /tmp/citadas.txt /tmp/bib.txt   # entradas não citadas
```

Depois de compilar, `documento.blg` lista campos ausentes e chaves não encontradas.
