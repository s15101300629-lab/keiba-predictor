// netkeiba ページの解析（アプリの API と scripts/ml の収集スクリプトで共用）

export function decodeEntities(value: string) {
  return value
    .replace(/&nbsp;/gi, " ")
    .replace(/&amp;/gi, "&")
    .replace(/&quot;/gi, '"')
    .replace(/&#39;/gi, "'")
    .replace(/&lt;/gi, "<")
    .replace(/&gt;/gi, ">");
}

export function cleanText(value: string) {
  return decodeEntities(
    value
      .replace(/<script\b[^>]*>[\s\S]*?<\/script>/gi, "")
      .replace(/<style\b[^>]*>[\s\S]*?<\/style>/gi, "")
      .replace(/<[^>]*>/g, "")
  )
    .replace(/\s+/g, " ")
    .trim();
}

export function extractTables(html: string) {
  const tables: string[][][] = [];
  const tableRegex = /<table\b[^>]*>([\s\S]*?)<\/table>/gi;
  let tableMatch: RegExpExecArray | null;

  while ((tableMatch = tableRegex.exec(html)) !== null) {
    const rows: string[][] = [];
    const rowRegex = /<tr\b[^>]*>([\s\S]*?)<\/tr>/gi;
    let rowMatch: RegExpExecArray | null;

    while ((rowMatch = rowRegex.exec(tableMatch[1])) !== null) {
      const cells: string[] = [];
      const cellRegex = /<t[dh]\b[^>]*>([\s\S]*?)<\/t[dh]>/gi;
      let cellMatch: RegExpExecArray | null;

      while ((cellMatch = cellRegex.exec(rowMatch[1])) !== null) {
        cells.push(cleanText(cellMatch[1]));
      }

      if (cells.length > 0) rows.push(cells);
    }

    if (rows.length > 0) tables.push(rows);
  }

  return tables;
}


export type PedigreeAncestor = {
  position: string; // F=父, M=母 を世代順に連ねたもの（例: MF = 母の父）
  generation: number;
  name: string;
  horseId: string | null;
  country: string | null;
  birthYear: number | null;
  color: string | null;
  line: string | null; // 系統（netkeiba が表示している場合のみ）
};
export type PedigreeCross = { name: string; percentage: number | null; generations: string };
export type Pedigree = {
  sire: string | null; sireId: string | null; sireLine: string | null;
  dam: string | null; damId: string | null;
  damsire: string | null; damsireId: string | null;
  damdam: string | null;
  ancestors: PedigreeAncestor[];
  crosses: PedigreeCross[];
};

const PEDIGREE_GENERATION_BY_ROWSPAN: Record<string, number> = { "16": 1, "8": 2, "4": 3, "2": 4 };

export function extractPedigree(html: string): Pedigree | null {
  const tableMatch = html.match(/<table\b[^>]*class=["'][^"']*\bblood_table\b[^"']*["'][^>]*>([\s\S]*?)<\/table>/i);
  if (!tableMatch) return null;

  const ancestors: PedigreeAncestor[] = [];
  const pathByGeneration: string[] = [];
  const childCount = new Map<string, number>();
  const cellRegex = /<td\b([^>]*)>([\s\S]*?)<\/td>/gi;
  let cellMatch: RegExpExecArray | null;
  while ((cellMatch = cellRegex.exec(tableMatch[1])) !== null) {
    const rowspan = cellMatch[1].match(/\browspan\s*=\s*["']?(\d+)/i)?.[1];
    const generation = rowspan ? PEDIGREE_GENERATION_BY_ROWSPAN[rowspan] ?? 5 : 5;
    // 血統表は「父→父の父→…」の前順でセルが並ぶので、親ごとの出現順で父(F)/母(M)を決める
    const parentPath = generation === 1 ? "" : pathByGeneration[generation - 1] ?? "";
    const seen = childCount.get(parentPath) ?? 0;
    childCount.set(parentPath, seen + 1);
    const position = parentPath + (seen === 0 ? "F" : "M");
    pathByGeneration[generation] = position;

    const body = cellMatch[2];
    const anchor = body.match(/<a\b[^>]*href=["'][^"']*\/horse\/(?!ped\/|sire\/|mare\/)([0-9a-z]+)\/?["'][^>]*>([\s\S]*?)<\/a>/i);
    const rawName = anchor ? cleanText(anchor[2]) : cleanText(body.split(/<br\s*\/?>/i)[0] ?? "");
    if (!rawName) continue;
    const countryMatch = rawName.match(/^(.*?)\s*[（(]([^）)]+)[）)]\s*$/);
    const lines = body.split(/<br\s*\/?>/i).slice(1).map(cleanText).filter(Boolean);
    const yearColor = lines.map((line) => line.match(/^(\d{4})\s*(\S+毛|芦|白|\S*)?/)).find(Boolean);
    ancestors.push({
      position,
      generation,
      name: countryMatch ? countryMatch[1].trim() : rawName,
      horseId: anchor?.[1] ?? null,
      country: countryMatch?.[2] ?? null,
      birthYear: yearColor ? Number(yearColor[1]) : null,
      color: yearColor?.[2] || null,
      line: lines.map((line) => line.replace(/\s*\d+(?:-[a-z])?$/, "")).find((line) => /系$/.test(line)) ?? null,
    });
  }
  if (!ancestors.length) return null;

  const crosses: PedigreeCross[] = [];
  const crossBlock = html.match(/<div\b[^>]*class=["']blood_cross["'][^>]*>([\s\S]*?)<\/div>/i)?.[1] ?? "";
  for (const row of crossBlock.matchAll(/<tr\b[^>]*>([\s\S]*?)<\/tr>/gi)) {
    const cells = [...row[1].matchAll(/<td\b[^>]*>([\s\S]*?)<\/td>/gi)].map((cell) => cleanText(cell[1]));
    if (cells.length < 3 || !cells[0]) continue;
    const percentage = Number(cells[1].replace(/%/, ""));
    crosses.push({ name: cells[0], percentage: Number.isFinite(percentage) ? percentage : null, generations: cells[2] });
  }

  const at = (position: string) => ancestors.find((ancestor) => ancestor.position === position);
  return {
    sire: at("F")?.name ?? null, sireId: at("F")?.horseId ?? null, sireLine: at("F")?.line ?? null,
    dam: at("M")?.name ?? null, damId: at("M")?.horseId ?? null,
    damsire: at("MF")?.name ?? null, damsireId: at("MF")?.horseId ?? null,
    damdam: at("MM")?.name ?? null,
    ancestors,
    crosses,
  };
}
