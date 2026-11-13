# Elementos do texto no ufcTex

Modelos testados no TCC deste repositório. Copie e troque o conteúdo.

## Sumário
- Figura
- Figura com partes (a), (b)
- Tabela
- Quadro
- Código-fonte
- Equação
- Alíneas
- Siglas
- Símbolos
- Apêndices e anexos
- Posicionamento de flutuantes

## Figura

```latex
\begin{figure}[H]
    \centering
    \Caption{\label{fig:arquitetura}Arquitetura do sistema desenvolvido}
    \UFCfig{}{
        \includegraphics[width=0.9\textwidth]{figuras/arquitetura.png}
    }{
        \Fonte{Elaborada pelo autor.}
    }
\end{figure}
```

- `\Caption` (com C maiúsculo) põe o título acima, em espaço simples, na largura da figura.
- O primeiro argumento de `\UFCfig` fica vazio; o segundo é o conteúdo (imagem, TikZ via `\input`); o terceiro, a fonte.
- Fonte de figura adaptada: `\Fonte{Adaptada de \citeonline{chave}.}`; de terceiros: `\Fonte{\citeonline{chave}.}`; captura de tela do próprio sistema: `\Fonte{Elaborada pelo autor.}`.
- Para título curto na lista de figuras, use `\Caption{\label{...}Título}` sem versão curta; se precisar de forma curta, prefira encurtar o título.
- Prefira PNG com resolução suficiente (largura da imagem em pixels ≥ 2× a largura impressa em pontos) ou PDF vetorial para gráficos.

## Figura com partes (a), (b)

```latex
\begin{figure}[H]
    \centering
    \Caption{\label{fig:planilha}Aba de cálculo: (a) parâmetros e (b) primeiros meses}
    \UFCfig{}{
        \begin{minipage}{\textwidth}
            \centering
            {\footnotesize (a) Parâmetros}\\[2pt]
            \includegraphics[width=0.68\textwidth]{figuras/parte-a.png}\\[8pt]
            {\footnotesize (b) Primeiros meses}\\[2pt]
            \includegraphics[width=\textwidth]{figuras/parte-b.png}
        \end{minipage}
    }{
        \Fonte{Elaborada pelo autor.}
    }
\end{figure}
```

No texto, cite a parte como `Figura~\ref{fig:planilha}(b)`.

## Tabela

```latex
\begin{table}[H]
    \captionsetup{width=15cm}
    \caption{Indicadores da aplicação de Flor do Campo}
    \label{tab:flor-indicadores}
    \centering
    \footnotesize
    \begin{tabular}{lrr}
        \toprule
        Indicador & Com curvas-guia & Sem racionamento \\
        \midrule
        Meses com falha & 106 & 188 \\
        Confiabilidade (\%) & 92,04 & 85,89 \\
        \bottomrule
    \end{tabular}
    \Fonte{Elaborada pelo autor.}
    \Nota{Déficits calculados em relação à demanda aplicada no mês.}
\end{table}
```

- Norma do IBGE adotada pela ABNT: sem linhas verticais nem fechamento lateral; `\toprule`, `\midrule`, `\bottomrule`.
- `\caption` vem antes do `\label`, e os dois antes do `tabular`.
- Números com vírgula decimal e ponto de milhar (1.332; 2.248,648). Mesma quantidade de casas decimais na coluna.
- Unidade no cabeçalho da coluna ou no título, entre parênteses.
- Célula sem dado: `---` (traço), não vazio.
- Tabela larga: `\resizebox{\textwidth}{!}{...}` em volta do `tabular`, ou `tabularx`. Tabela longa (mais de uma página): `longtable`, repetindo o cabeçalho.
- Tabela em apêndice segue o mesmo modelo; a numeração continua a do texto.

## Quadro

Quadro é para conteúdo textual (comparações qualitativas, definições). Tem bordas em todas as células.

```latex
\begin{quadro}[H]
    \captionsetup{width=15cm}
    \caption{Características de ferramentas de simulação}
    \label{qua:ferramentas}
    \centering
    \footnotesize
    \renewcommand{\arraystretch}{1.35}
    \begin{tabular}{|>{\raggedright\arraybackslash}p{3cm}|>{\raggedright\arraybackslash}p{5cm}|>{\raggedright\arraybackslash}p{5cm}|}
        \hline
        \textbf{Ferramenta} & \textbf{Abordagem} & \textbf{Acesso} \\
        \hline
        MODSIM & rede de fluxo com otimização & programa instalado \\
        \hline
    \end{tabular}
    \Fonte{Elaborada pelo autor com base em \citeonline{labadie2007}.}
\end{quadro}
```

Rótulo com prefixo `qua:` e remissão "Quadro~\ref{qua:ferramentas}". Use a mesma forma de "Fonte:" em figuras, tabelas e quadros de todo o trabalho (o TCC deste repositório usa "Elaborada pelo autor.").

## Código-fonte

```latex
\noindent\begin{minipage}{\linewidth}
\begin{lstlisting}[
    language=Python,
    basicstyle=\ttfamily\scriptsize,
    columns=fullflexible,
    keepspaces=true,
    captionpos=t,
    caption={Implementação do balanço hídrico mensal},
    label={cod:balanco}
]
def balanco(v, i, d, e):
    return v + i - d - e
\end{lstlisting}
\Fonte{Elaborada pelo autor.}
\end{minipage}
\par\medskip
```

- A `minipage` mantém título, código e fonte na mesma página, mas não quebra. Uma listagem mais longa que a página deve sair da `minipage` (perde a garantia de ficar junta, mas não some).
- Se o log acusar `Overfull \vbox` perto de uma listagem, a causa usual é uma figura flutuante (`[!htbp]`) que caiu no topo da mesma página: as últimas linhas da listagem ficam fora da área impressa. Troque a figura anterior para `[H]`. `[p]` costuma empurrar a figura para muitas páginas adiante.
- Título curto para a lista: `caption={[Título curto]Título completo}`.
- Linguagem sem realce (fórmulas de planilha): `language={}`.
- Listagens mostram números de linha; ao citar uma linha no texto ("linha 10"), confira a numeração depois de editar o código.

## Equação

```latex
O volume preliminar é dado pela Equação \ref{eq:balanco}.

\begin{equation}
    P_t = V_t + I_t + T_t - D^{a}_t - E_t
    \label{eq:balanco}
\end{equation}

em que $T_t$ é o saldo de transferências e $D^{a}_t$ é a demanda aplicada, em hm$^3$.
```

- O parágrafo após a equação começa com "em que" e define cada símbolo novo com unidade.
- Casos: `\begin{cases} ... \end{cases}` dentro de `equation`.
- Equação dentro do texto: `$...$`; unidades fora do modo matemático (`hm$^3$`, `m$^3$/s`).
- Inserir uma equação renumera as seguintes. Procure números escritos à mão (`grep -rn "Equação [0-9]"`) e comentários dentro de listagens.

## Alíneas

```latex
\begin{alineas}
    \item primeiro item;
    \item segundo item; e
    \item último item.
\end{alineas}
```

Itens em minúscula, terminados em ponto e vírgula, o penúltimo com "; e", o último com ponto (NBR 6024). Subdivisão: `subalineas`. Use alíneas só para itens paralelos; argumento vai em parágrafo.

## Siglas

Em `1-pre-textuais/lista-de-abreviaturas-e-siglas.tex`:

```latex
\newacronym{COGERH}{COGERH}{Companhia de Gestão dos Recursos Hídricos}
\newacronym{JSON}{JSON}{\textit{JavaScript Object Notation}}
```

No texto, `\gls{COGERH}`: na primeira ocorrência sai o nome por extenso seguido da sigla, depois só a sigla. Termo estrangeiro por extenso em itálico. A lista sai na ordem do modelo; se uma sigla só aparece em figura, tabela ou título, acrescente `\glsadd{SIGLA}` na linha de `\glsadd` em `documento.tex`.

## Símbolos

Em `1-pre-textuais/lista-de-simbolos.tex`, na ordem em que aparecem no texto:

```latex
\begin{simbolos} \itemsep -1pt
    \item[$V_t$] Volume armazenado no início do mês $t$, em hm$^3$
    \item[$\eta$] Percentual de atendimento da transferência, em fração
\end{simbolos}
```

Todo símbolo de equação entra na lista com unidade; símbolo removido do texto sai da lista.

## Apêndices e anexos

Apêndice é material do autor; anexo é material de terceiros. Cada um em arquivo próprio, incluído depois de `\imprimirapendices` (ou `\imprimiranexos`) em `documento.tex`:

```latex
\apendice{DISPONIBILIDADE DO CÓDIGO-FONTE}
\label{ap:codigo-fonte}

Texto do apêndice.

\section{Identificação da versão}
```

Remissão: `Apêndice~\ref{ap:codigo-fonte}` (sai a letra). PDF de terceiros em anexo: `\includepdf[pages=-]{arquivo.pdf}` dentro do anexo.

## Posicionamento de flutuantes

- `[H]` (pacote float, já carregado) fixa o elemento no ponto do texto. É o mais previsível neste modelo e o usado no TCC.
- `[!htbp]` deixa o LaTeX escolher; pode deslocar a figura para depois de uma listagem e cortar a listagem (ver Código-fonte).
- Todo flutuante é citado no texto antes de aparecer ("conforme a Tabela~\ref{...}").
- Use `~` entre o nome e o número (`Figura~\ref{}`) para não quebrar a linha entre eles.
