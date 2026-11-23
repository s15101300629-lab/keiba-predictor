// JRA の馬場情報（https://www.jra.go.jp/keiba/baba/ と同じデータ）。
// 馬場情報ページの HTML に書いてある天候・馬場は古いまま（2026-11-23 朝、東京は「11月21日（土曜）10時現在」のまま）で、
// ページは表示するときに /JRADB/accessJ.html（POST、CNAME=pw01iwtS3/CD）の JSON から最新の値に差し替えている。
// なので、その JSON を直接読む。開催中の競馬場ごとに、開催日・天候・芝とダートの馬場が入る（何時現在かは入っていない）。

export const JRA_BABA_URL = "https://www.jra.go.jp/JRADB/accessJ.html";

export type JraBaba = {
  venue: string; // 東京・京都 など
  date: string; // YYYY-MM-DD（開催日）
  asOf: string | null; // 取得した時刻（日本時間。「11月23日 9:20 取得」）
  weather: string | null;
  turf: string | null; // 良・稍重・重・不良
  dirt: string | null;
};

const GOINGS = ["稍重", "不良", "良", "重"];

/** accessJ の JSON（kaisai_info）→ 競馬場ごとの馬場. */
export function parseJraBabaJson(json: unknown, fetchedAt: Date): JraBaba[] {
  const list = (json as { kaisai_info?: unknown[] } | null)?.kaisai_info;
  if (!Array.isArray(list)) return [];
  const jst = new Date(fetchedAt.getTime() + 9 * 3600 * 1000);
  const asOf = `${jst.getUTCMonth() + 1}月${jst.getUTCDate()}日 ${jst.getUTCHours()}:${String(jst.getUTCMinutes()).padStart(2, "0")} 取得`;
  const out: JraBaba[] = [];
  for (const k of list as Record<string, unknown>[]) {
    const ymd = String(k.kaisai_ymd ?? "");
    const venue = String(k.jyoname ?? "");
    if (!/^\d{8}$/.test(ymd) || !venue) continue;
    const going = (v: unknown) => (typeof v === "string" && GOINGS.includes(v.trim()) ? v.trim() : null);
    const weather = typeof k.weather === "string" && k.weather.trim() ? k.weather.trim() : null;
    out.push({ venue, date: `${ymd.slice(0, 4)}-${ymd.slice(4, 6)}-${ymd.slice(6, 8)}`, asOf, weather, turf: going(k.ba_s), dirt: going(k.ba_d) });
  }
  return out;
}

let cache: { at: number; value: JraBaba[] } | null = null;
const CACHE_MS = 10 * 60 * 1000;

/** 開催中の競馬場の馬場（10分ためる。forceRefresh で取り直す）。取れなければ空. */
export async function fetchJraBaba(forceRefresh = false): Promise<JraBaba[]> {
  if (!forceRefresh && cache && Date.now() - cache.at < CACHE_MS) return cache.value;
  const response = await fetch(JRA_BABA_URL, {
    method: "POST",
    headers: { "Content-Type": "application/x-www-form-urlencoded" },
    body: "CNAME=" + encodeURIComponent("pw01iwtS3/CD"),
    cache: "no-store",
    signal: AbortSignal.timeout(10000),
  });
  if (!response.ok) return [];
  const text = new TextDecoder("shift_jis").decode(new Uint8Array(await response.arrayBuffer()));
  const value = parseJraBabaJson(JSON.parse(text), new Date());
  cache = { at: Date.now(), value };
  return value;
}
