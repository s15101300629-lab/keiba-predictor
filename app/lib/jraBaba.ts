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

// ===== クッション値・含水率・使用コース（モデルの入力。scripts/ml/features.py の BABA_FEATURES。2026-11-24） =====
// 学習は data/baba/baba.csv（過去の PDF と collect_baba_live.py）。アプリは馬場情報のページの部品を読む
// - _data_cushion.html・_data_moist.html：競馬場ごと（<div id="rcA" title="東京">）の測定の日時と値
// - index.html・index2.html・index3.html：競馬場ごとのページの「使用コース」（<h3>使用コース</h3> の後の <strong>C</strong>）
// その日の一番早い測定（学習と同じ朝の値）。その日の測定がまだなければ、その前の一番新しい測定
export type JraBabaDetail = { cushion: number | null; moistTurf: number | null; moistDirt: number | null; course: number | null; measuredOn: string | null };
type Unit = { date: string; time: string; cushion?: number; turf?: number; dirt?: number };
let detailCache: { at: number; units: Map<string, Unit[]>; course: Map<string, number> } | null = null;
const BABA_BASE = "https://www.jra.go.jp/keiba/baba/";

async function getSjis(name: string): Promise<string> {
  const r = await fetch(BABA_BASE + name, { cache: "no-store", signal: AbortSignal.timeout(10000) });
  if (!r.ok) return "";
  return new TextDecoder("shift_jis").decode(new Uint8Array(await r.arrayBuffer()));
}

function unitDate(text: string, ref: Date): { date: string; time: string } | null {
  const m = text.replace(/<[^>]+>/g, " ").match(/(\d+)月(\d+)日[\s\S]*?(\d+)時(\d+)分/);
  if (!m) return null;
  const jst = new Date(ref.getTime() + 9 * 3600 * 1000);
  const mo = Number(m[1]);
  const y = mo > jst.getUTCMonth() + 2 ? jst.getUTCFullYear() - 1 : jst.getUTCFullYear();
  return { date: `${y}-${String(mo).padStart(2, "0")}-${String(Number(m[2])).padStart(2, "0")}`, time: `${m[3].padStart(2, "0")}:${m[4]}` };
}

function blocks(page: string): [string, string[]][] {
  return page.split(/<div id="rc[A-Z]"/).slice(1).map((b) => [b.match(/title="([^"]+)"/)?.[1] ?? "", b.split('<div class="unit">').slice(1)]);
}

export function parseBabaDetail(cushionPage: string, moistPage: string, coursePages: string[], now: Date) {
  const units = new Map<string, Unit[]>();
  const put = (venue: string, d: { date: string; time: string }, v: Partial<Unit>) => {
    const list = units.get(venue) ?? [];
    let u = list.find((x) => x.date === d.date);
    if (!u) { u = { date: d.date, time: "99:99" }; list.push(u); }
    // 同じ日に何回か測ったときは一番早い測定（値の種類ごと）
    const key = "cushion" in v ? "c" : "m";
    const cur = (u as Record<string, unknown>)["_" + key] as string | undefined;
    if (!cur || d.time < cur) { Object.assign(u, v); (u as Record<string, unknown>)["_" + key] = d.time; }
    units.set(venue, list);
  };
  for (const [venue, us] of blocks(cushionPage)) {
    for (const u of us) {
      const d = unitDate(u, now);
      const v = u.match(/class="cushion">\s*([\d.]+)/);
      if (venue && d && v) put(venue, d, { cushion: Number(v[1]) });
    }
  }
  for (const [venue, us] of blocks(moistPage)) {
    for (const u of us) {
      const d = unitDate(u, now);
      const t = u.match(/class="turf">[\s\S]*?class="mg"[^>]*>\s*([\d.]+)[\s\S]*?class="m4c"[^>]*>\s*([\d.]+)/);
      const dt = u.match(/class="dirt">[\s\S]*?class="mg"[^>]*>\s*([\d.]+)[\s\S]*?class="m4c"[^>]*>\s*([\d.]+)/);
      if (venue && d && (t || dt)) put(venue, d, { turf: t ? (Number(t[1]) + Number(t[2])) / 2 : undefined, dirt: dt ? (Number(dt[1]) + Number(dt[2])) / 2 : undefined });
    }
  }
  const course = new Map<string, number>();
  for (const page of coursePages) {
    const venue = page.match(/第\d+回([^\s<（(]+?)競馬/)?.[1];
    const letter = page.match(/<h3>使用コース<\/h3>[\s\S]*?<strong>\s*([A-DＡ-Ｄ])\s*<\/strong>/)?.[1];
    if (venue && letter) course.set(venue, "AＡ".includes(letter) ? 0 : "BＢ".includes(letter) ? 1 : "CＣ".includes(letter) ? 2 : 3);
  }
  return { units, course };
}

/** その競馬場・その日の馬場の細かい値（10分ためる）。取れなければ全部 null. */
export async function fetchJraBabaDetail(venue: string, raceDate: string, forceRefresh = false): Promise<JraBabaDetail> {
  if (forceRefresh || !detailCache || Date.now() - detailCache.at >= CACHE_MS) {
    const [c, m, ...pages] = await Promise.all(["_data_cushion.html", "_data_moist.html", "index.html", "index2.html", "index3.html"]
      .map((n) => getSjis(n).catch(() => "")));
    detailCache = { at: Date.now(), ...parseBabaDetail(c, m, pages, new Date()) };
  }
  const list = (detailCache.units.get(venue) ?? []).filter((u) => u.date <= raceDate).sort((a, b) => (a.date < b.date ? 1 : -1));
  const pick = <K extends "cushion" | "turf" | "dirt">(k: K) => list.find((u) => u[k] != null) ?? null;
  const c = pick("cushion"), t = pick("turf"), d = pick("dirt");
  return { cushion: c?.cushion ?? null, moistTurf: t?.turf ?? null, moistDirt: d?.dirt ?? null,
    course: detailCache.course.get(venue) ?? null, measuredOn: c?.date ?? t?.date ?? null };
}
