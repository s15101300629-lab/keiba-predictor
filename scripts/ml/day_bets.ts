// 買い方の成績（2026-11-24、依頼者の指示）：答え合わせの日ごとに、画面と同じ計算（app/lib/raceBets.ts）で AI印と印から作る買い目を決め、
// 確定の払戻で当たり・払戻を数える。1点100円。
//   npx tsx scripts/ml/day_bets.ts 2026-11-29     # → data/ml/day_bets/2026-11-29.json
//   npx tsx scripts/ml/day_bets.ts --all          # 入力（data/ml/day_check/inputs）がある日を全部
// 入力：check_jra_day.py が書く data/ml/day_check/inputs/日付.json.gz（各馬のモデルの生の確率・本体の入力・確定オッズ・着順・払戻）と、
//       全部の組み合わせの確定オッズ data/jra-odds/日付.json.gz
// 画面との違い：オッズは確定オッズ（画面は取ったときのオッズ）。☆・消・▲はオッズで変わるので、レース前に見た印と違うことがある
import { existsSync, mkdirSync, readdirSync, readFileSync, writeFileSync } from "node:fs";
import path from "node:path";
import { gunzipSync } from "node:zlib";
import { valueLabels, type OddsTable, type ValueModel } from "@/app/lib/market";
import { longshotRatios, type LongshotModel } from "@/app/lib/longshot";
import { MARK_BET_POINTS, MARK_BET_TYPES } from "@/app/lib/markBets";
import { normalizeSum } from "@/app/lib/ml/model";
import { betLists, horseOddsOf, planBets, planMarks, starTickets, type PlanHorse, type PlanOdds } from "@/app/lib/raceBets";

const ROOT = path.resolve(__dirname, "../..");
const ML = path.join(ROOT, "data", "ml");
const readJson = <T>(p: string): T => JSON.parse(readFileSync(p, "utf8")) as T;
const readGz = <T>(p: string): T => JSON.parse(gunzipSync(readFileSync(p)).toString("utf8")) as T;

type InHorse = { n: number; horse: string; finish: number | null; odds: number | null; place: [number, number] | null;
  top3: number; top2: number | null; win: number | null; x: (number | null)[] };
type InRace = { race_id: string; venue: string; race_no: number; race_name: string; course: string;
  payouts: Record<string, Record<string, number>>; horses: InHorse[] };
type Inputs = { date: string; model: string; model_dir?: string; features: string[]; races: InRace[] };
type ComboOdds = Record<string, Record<string, number | [number, number | null]>>;

// 係数は、予測に使ったモデルと同じフォルダ（入力の model_dir。"" は本番、"prev" は 2025年までの一式 prev_models.py）から読む
function loadConfig(dir: string) {
  const d = path.join(ML, dir);
  return {
    calibration: readJson<{ top3?: number; top2?: number; win?: number }>(path.join(d, "calibration.json")),
    valueModel: readJson<ValueModel>(path.join(d, "value_combine.json")),
    longshotModel: existsSync(path.join(d, "model_longshot.json")) ? readJson<LongshotModel>(path.join(d, "model_longshot.json")) : null,
  };
}

// route.ts と同じ：モデルの log-odds に T を掛ける
const temper = (p: number, T: number | undefined) => {
  if (!T || T === 1) return p;
  const q = Math.min(Math.max(p, 1e-9), 1 - 1e-9);
  return 1 / (1 + Math.exp(-T * Math.log(q / (1 - q))));
};
const table = (src: Record<string, number | [number, number | null]> | undefined): OddsTable =>
  Object.fromEntries(Object.entries(src ?? {}).map(([k, v]) => [k, Array.isArray(v) ? { min: v[0], max: v[1] } : { min: v, max: null }]));

type Ticket = { key: string; marks: string[]; prob: number; odds: number | null; hit: boolean; pay: number };

function runDay(date: string) {
  const inp = readGz<Inputs>(path.join(ML, "day_check", "inputs", `${date}.json.gz`));
  const oddsPath = path.join(ROOT, "data", "jra-odds", `${date}.json.gz`);
  const combos = existsSync(oddsPath) ? readGz<{ races: Record<string, ComboOdds> }>(oddsPath).races : {};
  const { calibration, valueModel, longshotModel } = loadConfig(inp.model_dir ?? "");
  const longshotReady = !!longshotModel && inp.features.every((f, k) => longshotModel.features[k] === f);
  const races = inp.races.map((r) => {
    const hs = r.horses;
    const base3 = hs.map((h) => temper(h.top3, calibration.top3));
    const probs = normalizeSum(base3, 3);
    const baseWin = hs.map((h) => (h.win != null ? temper(h.win, calibration.win) : null));
    const plan: PlanHorse[] = hs.map((h, i) => ({ id: String(h.n), num: h.n, top3: probs[i], baseWin: baseWin[i] }));
    const c = combos[r.race_id] ?? {};
    const withOdds = hs.every((h) => h.odds != null && h.odds > 0);
    const odds: PlanOdds | null = withOdds ? {
      win: Object.fromEntries(hs.map((h) => [String(h.n), { min: h.odds!, max: null }])),
      place: Object.fromEntries(hs.filter((h) => h.place).map((h) => [String(h.n), { min: h.place![0], max: h.place![1] }])),
      umaren: table(c["馬連"]), wide: table(c["ワイド"]), trio: table(c["3連複"]), trifecta: table(c["3連単"]),
    } : null;
    const horseOdds = horseOddsOf(plan, odds);
    const values = odds ? valueLabels(new Map(hs.map((h, i) => [h.n, base3[i]])), odds.win, valueModel) : null;
    const longshot = odds && longshotReady
      ? new Map([...longshotRatios(longshotModel!, hs.filter((h, i) => baseWin[i] != null).map((h) => ({ n: h.n, x: h.x, top3: base3[hs.indexOf(h)], win: baseWin[hs.indexOf(h)]! })),
        new Map(hs.map((h) => [h.n, h.odds!])))].map(([n, v]) => [n, v.ratio])) : null;
    const marks = planMarks(plan, odds, values, longshot, horseOdds, valueModel.win ?? null);
    const plans = planBets(plan, marks, horseOdds, odds, values, valueModel.combined);
    const pay = (type: string, key: string) => r.payouts[type]?.[key] ?? 0;
    const grade = (mode: "prob" | "ev") => Object.fromEntries(betLists(plans, odds, mode).map((l) => [l.type, l.tickets.map((t): Ticket => ({
      key: t.key, marks: t.marks.map((m) => m ?? ""), prob: Math.round(t.prob * 10000) / 10000,
      odds: (l.type === "ワイド" ? c["ワイド"]?.[t.key] && (c["ワイド"][t.key] as [number, number])[0] : c[l.type]?.[t.key] as number | undefined) ?? null,
      hit: pay(l.type, t.key) > 0, pay: pay(l.type, t.key),
    }))]));
    const stars = starTickets(plan, marks, horseOdds, odds).map((t) => ({ kind: t.kind, key: t.key, hit: pay(t.type, t.key) > 0, pay: pay(t.type, t.key) }));
    return {
      race_id: r.race_id, venue: r.venue, race_no: r.race_no, race_name: r.race_name, course: r.course,
      shinba: r.race_name.includes("新馬"), payouts_ok: Object.keys(r.payouts).length > 0, odds_ok: !!odds,
      marks: hs.map((h) => ({ n: h.n, horse: h.horse, mark: marks.get(String(h.n))?.mark ?? null, finish: h.finish,
        pop: horseOdds.get(String(h.n))?.popularity ?? null, win_pay: pay("単勝", String(h.n)), place_pay: pay("複勝", String(h.n)) }))
        .filter((m) => m.mark).sort((a, b) => "◎○▲☆△消".indexOf(a.mark!) - "◎○▲☆△消".indexOf(b.mark!)),
      bets: { prob: grade("prob"), ev: grade("ev") }, stars,
    };
  });
  const out = { date, model: inp.model, model_dir: inp.model_dir ?? "", note: "1点100円・確定の払戻。印・買い目は確定オッズで決めた（画面は取ったときのオッズ）",
    points: MARK_BET_POINTS, types: MARK_BET_TYPES, races };
  mkdirSync(path.join(ML, "day_bets"), { recursive: true });
  writeFileSync(path.join(ML, "day_bets", `${date}.json`), JSON.stringify(out) + "\n");
  // 確かめ用の表示
  let stake = 0, ret = 0;
  for (const r of races) for (const ts of Object.values(r.bets.prob)) for (const t of ts) { stake += 100; ret += t.pay; }
  console.log(date, races.length, "races", "確率順の全券種 回収率", stake ? (ret / stake * 100).toFixed(1) + "%" : "-", "点数", stake / 100);
}

const args = process.argv.slice(2);
const dates = args.includes("--all")
  ? readdirSync(path.join(ML, "day_check", "inputs")).filter((f) => f.endsWith(".json.gz")).map((f) => f.slice(0, 10)).sort()
  : args.filter((a) => !a.startsWith("--"));
for (const d of dates) runDay(d);
