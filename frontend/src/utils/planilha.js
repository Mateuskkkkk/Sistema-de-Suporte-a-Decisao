// Exportação de planilhas .xlsx com ExcelJS.
// Mantém as mesmas chamadas usadas antes com a biblioteca xlsx (SheetJS):
//   const wb = utils.book_new()
//   utils.book_append_sheet(wb, utils.json_to_sheet(linhas), 'Nome')
//   writeFile(wb, 'arquivo.xlsx')
// O ExcelJS é carregado sob demanda, apenas quando o usuário exporta.

function nomeAbaSeguro(nome, usados) {
  let base = String(nome || 'Planilha').replace(/[\\/?*[\]:]/g, ' ').trim().slice(0, 31) || 'Planilha'
  let candidato = base
  let n = 2
  while (usados.has(candidato.toLowerCase())) {
    const sufixo = ` (${n++})`
    candidato = base.slice(0, 31 - sufixo.length) + sufixo
  }
  usados.add(candidato.toLowerCase())
  return candidato
}

export const utils = {
  book_new: () => ({ abas: [] }),
  json_to_sheet: (linhas) => ({ linhas: Array.isArray(linhas) ? linhas : [] }),
  book_append_sheet: (wb, ws, nome) => { wb.abas.push({ nome, linhas: ws.linhas }) },
}

export async function writeFile(wb, nomeArquivo) {
  const { default: ExcelJS } = await import('exceljs')
  const workbook = new ExcelJS.Workbook()
  const usados = new Set()
  for (const aba of wb.abas) {
    const sheet = workbook.addWorksheet(nomeAbaSeguro(aba.nome, usados))
    const colunas = []
    for (const linha of aba.linhas) {
      for (const chave of Object.keys(linha || {})) if (!colunas.includes(chave)) colunas.push(chave)
    }
    sheet.columns = colunas.map(c => ({ header: c, key: c, width: Math.min(40, Math.max(10, c.length + 2)) }))
    sheet.getRow(1).font = { bold: true }
    for (const linha of aba.linhas) sheet.addRow(linha)
  }
  const buffer = await workbook.xlsx.writeBuffer()
  const blob = new Blob([buffer], { type: 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet' })
  const url = URL.createObjectURL(blob)
  const a = document.createElement('a')
  a.href = url
  a.download = nomeArquivo
  document.body.appendChild(a)
  a.click()
  a.remove()
  setTimeout(() => URL.revokeObjectURL(url), 1000)
}
