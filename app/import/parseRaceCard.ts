export type ParsedRow = {
  waku: number;
  umaban: number;
  name: string;
  sexAge: string;
  weight: string;
  jockey: string;
};

export type ParseResult = {
  rows: ParsedRow[];
  failed: string[];
};

const SEX_AGE = /^[\u7261\u725d\u30bb\u9a38]\d{1,2}$/;

function toHalfWidth(text: string): string {
  return text
    .replace(/[\uff10-\uff19]/g, (c) => String.fromCharCode(c.charCodeAt(0) - 0xfee0))
    .replace(/\uff0e/g, ".");
}

export function parseRaceCard(input: string): ParseResult {
  const tokens = toHalfWidth(input)
    .split(/[\s\u3000]+/)
    .filter((t) => t.length > 0);

  const rows: ParsedRow[] = [];
  const failed: string[] = [];
  const seen = new Set<number>();

  for (let i = 0; i < tokens.length; i++) {
    if (!SEX_AGE.test(tokens[i])) continue;

    const name = tokens[i - 1] ?? "";
    const umaban = Number(tokens[i - 2]);
    const waku = Number(tokens[i - 3]);
    const weight = tokens[i + 1] ?? "";
    const jockey = tokens[i + 2] ?? "";

    const ok =
      Number.isInteger(umaban) && umaban >= 1 && umaban <= 18 &&
      Number.isInteger(waku) && waku >= 1 && waku <= 8 &&
      /^\d{2}(\.\d)?$/.test(weight) &&
      name.length > 0 && jockey.length > 0;

    if (!ok) {
      failed.push(tokens.slice(Math.max(0, i - 3), i + 3).join(" "));
      continue;
    }
    if (seen.has(umaban)) continue;
    seen.add(umaban);
    rows.push({ waku, umaban, name, sexAge: tokens[i], weight, jockey });
  }

  rows.sort((a, b) => a.umaban - b.umaban);
  return { rows, failed };
}
