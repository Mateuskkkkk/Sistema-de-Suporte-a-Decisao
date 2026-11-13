#!/usr/bin/env bash
# Compila um trabalho no modelo ufcTex, aponta problemas do log e limpa os auxiliares.
# Uso: compilar.sh [pasta-do-trabalho] [arquivo-principal-sem-.tex]
# Padrão: pasta atual e "documento".
set -u
DIR="${1:-.}"
BASE="${2:-documento}"
cd "$DIR" || { echo "Pasta não encontrada: $DIR"; exit 2; }
[ -f "$BASE.tex" ] || { echo "Arquivo $BASE.tex não encontrado em $(pwd)"; exit 2; }

latexmk -pdf -interaction=nonstopmode "$BASE.tex" > /dev/null 2>&1
RC=$?

echo "== Resultado da compilação (latexmk saiu com código $RC)"
if [ -f "$BASE.log" ]; then
  ERROS=$(grep -c "^! " "$BASE.log")
  INDEF=$(grep -cE "undefined|Undefined" "$BASE.log")
  VBOX=$(grep -c "Overfull \\\\vbox" "$BASE.log")
  HBOX=$(grep -cE "Overfull \\\\hbox \(([1-9][0-9]+|[5-9])\." "$BASE.log")
  echo "Erros: $ERROS | Referências ou citações indefinidas: $INDEF | Overfull vbox: $VBOX | Overfull hbox acima de 5pt: $HBOX"
  [ "$ERROS" -gt 0 ] && grep -A3 "^! " "$BASE.log" | head -20
  [ "$INDEF" -gt 0 ] && grep -E "undefined|Undefined" "$BASE.log" | sort -u | head -15
  if [ "$VBOX" -gt 0 ]; then
    echo "ATENÇÃO: Overfull vbox costuma significar conteúdo cortado no pé da página (listagem em minipage após figura). Confira as páginas:"
    grep -B12 "Overfull \\\\vbox" "$BASE.log" | grep -oE "\[[0-9]+" | tail -3
  fi
fi
[ -f "$BASE.blg" ] && grep -E "^Warning--|^I couldn.t|error message" "$BASE.blg" | head -10
[ -f "$BASE.pdf" ] && pdfinfo "$BASE.pdf" 2>/dev/null | grep -E "^Pages"

# Limpeza: latexmk -c e os arquivos que ele não remove neste modelo
latexmk -c > /dev/null 2>&1
rm -f "$BASE".{bbl,glo,ist,lol,loq,acn,acr,alg,glg,gls}
echo "Auxiliares removidos. O PDF fica em $(pwd)/$BASE.pdf"
exit $RC
