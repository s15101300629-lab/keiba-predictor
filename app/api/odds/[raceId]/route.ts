import { NextResponse } from "next/server";

export const runtime = "nodejs";

// netkeiba のオッズ（race.netkeiba.com/api/api_get_jra_odds.html）を券種ごとに取り、
// 「馬番を並べたキー → オッズ（幅があるものは下限・上限）」にそろえて返す。
// 券種の番号：1=単勝・複勝, 4=馬連, 5=ワイド, 7=3連複, 8=3連単
const TYPES: Record<string, number> = { win: 1, umaren: 4, wide: 5, trio: 7, trifecta: 8 };
const ODDS_URL = (raceId: string, type: number) =>
  `https://race.netkeiba.com/api/api_get_jra_odds.html?race_id=${raceId}&type=${type}&action=update`;

type Odds = { min: number; max: number | null };
export type OddsResponse = {
  ok: boolean; raceId: string; fetchedAt: string; officialTime: string | null;
  win: Record<string, Odds>; place: Record<string, Odds>; umaren: Record<string, Odds>;
  wide: Record<string, Odds>; trio: Record<string, Odds>; trifecta: Record<string, Odds>;
  errors: string[];
  // netkeiba の status。"yoso" は発売前の予想オッズ（netkeiba の予想で、実際のオッズではない）
  status: string | null;
};

const num = (v: unknown) => {
  const n = typeof v === "number" ? v : Number(String(v ?? "").replace(/,/g, ""));
  return Number.isFinite(n) && n > 0 ? n : null;
};

// 2桁ずつの馬番キー（"0102" など）→ "1-2"（そろえて扱いやすくする）
function normKey(key: string): string | null {
  if (!/^\d+$/.test(key) || key.length % 2 !== 0) return null;
  return (key.match(/\d\d/g) ?? []).map((x) => String(Number(x))).join("-");
}

function parseTable(table: unknown): Record<string, Odds> {
  const out: Record<string, Odds> = {};
  if (!table || typeof table !== "object") return out;
  for (const [rawKey, value] of Object.entries(table as Record<string, unknown>)) {
    const key = normKey(rawKey);
    if (!key) continue;
    const arr = Array.isArray(value) ? value : [value];
    const min = num(arr[0]);
    if (min === null) continue;
    out[key] = { min, max: num(arr[1]) };
  }
  return out;
}

async function fetchType(raceId: string, type: number) {
  const res = await fetch(ODDS_URL(raceId, type), {
    cache: "no-store", headers: { "User-Agent": "Mozilla/5.0", Accept: "application/json" },
    signal: AbortSignal.timeout(10000),
  });
  if (!res.ok) throw new Error(`HTTP ${res.status}`);
  const json = await res.json() as { status?: string; data?: { official_datetime?: string; odds?: Record<string, unknown> } };
  return json;
}

export async function GET(_request: Request, { params }: { params: Promise<{ raceId: string }> }) {
  const { raceId } = await params;
  if (!/^\d{12}$/.test(raceId)) return NextResponse.json({ ok: false, error: "invalid race id" }, { status: 400 });
  const result: OddsResponse = {
    ok: false, raceId, fetchedAt: new Date().toISOString(), officialTime: null,
    win: {}, place: {}, umaren: {}, wide: {}, trio: {}, trifecta: {}, errors: [], status: null,
  };
  await Promise.all(Object.entries(TYPES).map(async ([name, type]) => {
    try {
      const json = await fetchType(raceId, type);
      const odds = json.data?.odds ?? {};
      result.officialTime ??= json.data?.official_datetime ?? null;
      if (type === 1) {
        result.status = json.status ?? null;
        result.win = parseTable(odds["1"]);
        result.place = parseTable(odds["2"]);
      } else {
        (result as unknown as Record<string, Record<string, Odds>>)[name] = parseTable(odds[String(type)]);
      }
    } catch (error) {
      result.errors.push(`${name}: ${error instanceof Error ? error.message : "error"}`);
    }
  }));
  result.ok = Object.keys(result.win).length > 0;
  return NextResponse.json(result, { status: result.ok ? 200 : 502, headers: { "Cache-Control": "no-store" } });
}
