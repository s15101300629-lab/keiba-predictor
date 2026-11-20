// scripts/ml/features.py と同じ定義の特徴量計算。どちらかを変えたら両方合わせ、
// `npm run ml:parity` で Python と一致することを確認すること。

export type Store = {
  lastDate: string;
  // [date, venue, surface, distance, going(-1=不明), field, finish(-1=中止), timeDiff(-1=不明),
  //  pace(0=スロー 1=ミドル 2=ハイ -1=不明), posFirst(最初のコーナーの位置, -1=不明), posLast,
  //  prize(1着賞金・万円), weight(斤量), jockey(names の番号), trainer(names の番号), tidx(タイム指数),
  //  victim(成績表の〔制裁〕で被害馬=1), ab(能力指数), agDev, agRank, earlyGap, fade(推定上がりの指標。scripts/ml/features.py の add_ability)]
  horses: Record<string, [string, number, number, number | null, number, number, number, number, number, number, number,
    number | null, number | null, number, number, number | null, number?,
    (number | null)?, (number | null)?, (number | null)?, (number | null)?, (number | null)?][]>;
  names?: string[]; // 騎手・調教師名の一覧
  // [date, venue, surface, distance, top3]
  jockeys: Record<string, [string, number, number, number | null, number][]>;
  // [date, venue, top3]
  trainers?: Record<string, [string, number, number][]>;
  // "競馬場-芝ダ-距離" -> [[date, 枠番, 出走数, 3着以内数]]
  courseWaku?: Record<string, [string, number, number, number][]>;
};

export type RaceInput = {
  date: string; // YYYY-MM-DD
  venue: number; // 場コード 1〜10
  surface: number; // 0=芝 1=ダート
  distance: number | null;
  going: number | null; // 0=良 1=稍重 2=重 3=不良
  fieldSize: number;
  prize1stMan: number | null; // 1着本賞（万円）
};

export type RunnerInput = {
  horse: string;
  jockeyKey: string | null; // normJockey 済みで store.jockeys のキー
  trainerKey?: string | null; // normJockey 済みで store.trainers のキー
  waku: number | null; umaban: number | null; weight: number | null;
  age: number | null; sex: string | null;
  bodyWeight: number | null; bodyWeightDiff: number | null;
  // 成績表にない過去走（海外のレース）。store.horses と同じ形。アプリだけで使う（学習データにはない）
  extraRuns?: Store["horses"][string];
  trainRank?: number | null; // 調教評価 A=4〜D=1（netkeiba の調教ページ）。なければ null
  // 血統（app/lib/pedigree.ts の pedigreeModelValues。父・母父・母母父の産駒成績の 50:25:12.5 平均）
  pedigree?: Partial<Record<"ped_heavy" | "ped_wet" | "ped_top3" | "ped_dist", number | null>> | null;
};

export const TRAIN_RANK: Record<string, number> = { A: 4, B: 3, C: 2, D: 1 };

export type Features = Record<string, number>;

export const FORM_RUNS = 5;
export const NIGE_CAND_RATE = 0.4; // 近5走で最初のコーナー1番手が4割以上 → 逃げ候補
export const FRONT_RATIO = 1 / 3; // 最初のコーナーで前3分の1 → 先行
export const TIME_DIFF_SCALE = 2;
export const AB_RUNS = 5; // 能力指数・推定上がりは、今回と同じ芝ダの過去走（指数がある走）の新しい順に最大5走
export const AB_DECAY = 0.8; // 直近加重平均の重み 0.8^k
const AGARI_KEYS = ["ag_dev", "ag_rank", "early_gap", "fade"] as const;
const AGARI_LOWER_BETTER = new Set(["ag_rank", "early_gap", "fade"]);
const SEX: Record<string, number> = { "牡": 0, "牝": 1, "セ": 2 };

export function normJockey(name: string | null | undefined): string {
  if (!name) return "";
  const s = name.normalize("NFKC").split("(")[0];
  return [...s].filter((ch) => !" .・　".includes(ch)).join("");
}

/** netkeiba の略記（例: 岩田望, ルメール, Ｃデムーロ）を成績表の騎手名に対応付ける. */
export function resolveJockey(name: string | null | undefined, store: Store, raceDate: string): string | null {
  const n = normJockey(name);
  if (!n) return null;
  if (store.jockeys[n]) return n;
  const year = raceDate.slice(0, 4);
  const candidates = Object.keys(store.jockeys)
    .filter((key) => key.startsWith(n) || key.endsWith(n))
    .map((key) => ({ key, rides: store.jockeys[key].filter((r) => r[0].slice(0, 4) === year && r[0] < raceDate).length }))
    .sort((a, b) => b.rides - a.rides);
  if (!candidates.length) return null;
  if (candidates.length > 1 && candidates[0].rides === candidates[1].rides) return null;
  return candidates[0].key;
}

export function runComponents(finish: number | null, field: number, timeDiff: number | null): [number, number] {
  if (finish === null || field <= 1) return [0, 0];
  const finishC = 1 - (finish - 1) / (field - 1);
  if (timeDiff === null) return [finishC, 0];
  return [finishC, Math.min(1, Math.max(0, 1 - timeDiff / TIME_DIFF_SCALE))];
}

const rate = (top3: number, n: number) => (n ? top3 / n : NaN);
const daysBetween = (a: string, b: string) => Math.round((Date.parse(b) - Date.parse(a)) / 86400000);

/** store の過去走に、成績表にない過去走（海外）を日付順に足す. */
function horseRuns(store: Store, horse: string, extra?: Store["horses"][string]): Store["horses"][string] {
  const base = store.horses[horse] ?? [];
  return extra?.length ? [...base, ...extra].sort((a, b) => (a[0] < b[0] ? -1 : a[0] > b[0] ? 1 : 0)) : base;
}

function horseFeatures(store: Store, race: RaceInput, horse: string, extra?: Store["horses"][string]): Features {
  const runs = horseRuns(store, horse, extra)
    .filter((r) => r[0] < race.date)
    .map((r) => ({
      date: r[0], venue: r[1], surface: r[2], distance: r[3],
      going: r[4] < 0 ? null : r[4], field: r[5],
      finish: r[6] < 0 ? null : r[6], timeDiff: r[7] < 0 ? null : r[7],
      pace: r[8] < 0 ? null : r[8], posFirst: r[9] > 0 ? r[9] : null, posLast: r[10] > 0 ? r[10] : null,
      prize: r[11] ?? null, weight: r[12] ?? null, tidx: r[15] ?? null, victim: r[16] ?? 0,
      ab: r[17] ?? null, ag_dev: r[18] ?? null, ag_rank: r[19] ?? null, early_gap: r[20] ?? null, fade: r[21] ?? null,
    }));
  const f: Features = { n_prior: runs.length };
  if (runs.length) {
    f.days_since_last = daysBetween(runs[runs.length - 1].date, race.date);
    const comps = runs.slice(-FORM_RUNS).map((r) => runComponents(r.finish, r.field, r.timeDiff));
    let finishSum = 0;
    let timeSum = 0;
    for (const c of comps) { finishSum += c[0]; timeSum += c[1]; }
    f.form_finish_mean = finishSum / comps.length;
    f.form_time_mean = timeSum / comps.length;
    f.form_raw = (f.form_finish_mean + f.form_time_mean) / 2;
    [f.last1_finish_c, f.last1_time_c] = comps[comps.length - 1];
  } else {
    for (const k of ["days_since_last", "form_finish_mean", "form_time_mean", "form_raw", "last1_finish_c", "last1_time_c"]) f[k] = NaN;
  }
  const agg = (prefix: string, cond: (r: (typeof runs)[number]) => boolean) => {
    const sel = runs.filter(cond);
    const top3 = sel.filter((r) => r.finish !== null && r.finish <= 3).length;
    f[prefix + "_n"] = sel.length;
    f[prefix + "_top3"] = rate(top3, sel.length);
  };
  const s = race.surface;
  const d = race.distance;
  agg("surface", (r) => r.surface === s);
  agg("dist", (r) => r.surface === s && d !== null && r.distance !== null && Math.abs(r.distance - d) <= 200);
  agg("venue", (r) => r.surface === s && r.venue === race.venue);
  agg("heavy", (r) => r.surface === s && r.going !== null && r.going >= 2); // 道悪補正の判定用（モデルの入力ではない）
  agg("wet", (r) => r.surface === s && r.going !== null && r.going >= 1); // 稍重以上（モデルの入力）
  agg("hp", (r) => r.pace === 2);
  agg("sp", (r) => r.pace === 0);

  // クラスの上下（今回の1着賞金 ÷ 前走 / 近5走平均）
  const todayPrize = race.prize1stMan;
  const prizes = runs.slice(-FORM_RUNS).map((r) => r.prize).filter((p): p is number => !!p);
  const lastPrize = runs.length ? runs[runs.length - 1].prize : null;
  f.prize_ratio_last = todayPrize && lastPrize ? todayPrize / lastPrize : NaN;
  let prizeSum = 0;
  for (const p of prizes) prizeSum += p;
  f.prize_ratio_mean5 = todayPrize && prizes.length ? todayPrize / (prizeSum / prizes.length) : NaN;
  // 前走からの変化（騎手の乗り替わり・斤量は runnerFeatures で計算）
  if (runs.length) {
    const last = runs[runs.length - 1];
    f.dist_change = d !== null && last.distance !== null ? d - last.distance : NaN;
    f.surface_change = last.surface !== s ? 1 : 0;
  } else {
    f.dist_change = NaN;
    f.surface_change = NaN;
  }
  // 不利（前走で被害馬になったか・近5走で被害馬になった回数）
  f.last1_victim = runs.length ? runs[runs.length - 1].victim : NaN;
  f.victim_n5 = runs.length ? runs.slice(-FORM_RUNS).reduce((sum, r) => sum + r.victim, 0) : NaN;
  // タイム指数（近5走）
  const tx = runs.slice(-FORM_RUNS).map((r) => r.tidx).filter((t): t is number => t !== null);
  let txSum = 0;
  for (const t of tx) txSum += t;
  f.tidx_mean5 = tx.length ? txSum / tx.length : NaN;
  f.tidx_best5 = tx.length ? Math.max(...tx) : NaN;
  f.tidx_last1 = runs.length && runs[runs.length - 1].tidx !== null ? runs[runs.length - 1].tidx! : NaN;

  // 能力指数・推定上がり（今回と同じ芝ダで指数がある走、新しい順に最大5走）
  const prev = runs.filter((r) => r.surface === s && r.ab !== null).slice(-AB_RUNS).reverse();
  for (const [k, v] of Object.entries(summarizeAbility(prev.map((r) => r.ab!)))) f["ab_" + k] = v;
  const ag = prev.filter((r) => r.ag_dev !== null);
  for (const key of AGARI_KEYS) {
    const vals = ag.map((r) => r[key]!);
    f[key + "_wmean"] = summarizeAbility(vals).wmean;
    f[key + "_best"] = vals.length ? (AGARI_LOWER_BETTER.has(key) ? Math.min(...vals) : Math.max(...vals)) : NaN;
  }

  // 脚質・位置取り（近5走のうち通過順位がある走）
  const pos = runs.slice(-FORM_RUNS).filter((r) => r.posFirst && r.posLast && r.field);
  if (pos.length) {
    let first = 0;
    let last = 0;
    for (const r of pos) first += r.posFirst! / r.field;
    for (const r of pos) last += r.posLast! / r.field;
    f.pos_first_mean = first / pos.length;
    f.pos_last_mean = last / pos.length;
    f.nige_rate = pos.filter((r) => r.posFirst === 1).length / pos.length;
    f.senko_rate = pos.filter((r) => r.posFirst! / r.field <= FRONT_RATIO).length / pos.length;
  } else {
    for (const k of ["pos_first_mean", "pos_last_mean", "nige_rate", "senko_rate"]) f[k] = NaN;
  }
  return f;
}

export type JockeyStats = { n: number; top3: number; venueN: number; venueTop3: number; courseN: number; courseTop3: number };

/** 今年（レースと同じ暦年）のレース日より前の騎乗を集計. */
export function jockeyStats(store: Store, race: RaceInput, jockeyKey: string | null): JockeyStats {
  const year = race.date.slice(0, 4);
  const rides = (jockeyKey ? store.jockeys[jockeyKey] ?? [] : [])
    .filter((r) => r[0] < race.date && r[0].slice(0, 4) === year);
  const count = (cond: (r: (typeof rides)[number]) => boolean) => {
    const sel = rides.filter(cond);
    return [sel.length, sel.reduce((sum, r) => sum + r[4], 0)];
  };
  const [n, top3] = count(() => true);
  const [venueN, venueTop3] = count((r) => r[1] === race.venue);
  const [courseN, courseTop3] = count((r) => r[1] === race.venue && r[2] === race.surface && r[3] === race.distance);
  return { n, top3, venueN, venueTop3, courseN, courseTop3 };
}

/** netkeiba の厩舎表記（例: 栗東中村）を成績表の調教師名に対応付ける。同じ苗字の調教師がいるので、まずその馬の前走の調教師と照合する. */
export function resolveTrainer(name: string | null | undefined, store: Store, raceDate: string, horse: string): string | null {
  const n = normJockey((name ?? "").replace(/^(美浦|栗東|地方|海外)/, ""));
  if (!n) return null;
  const runs = (store.horses[horse] ?? []).filter((r) => r[0] < raceDate);
  for (let i = runs.length - 1; i >= 0; i--) {
    const t = store.names?.[runs[i][14]] ?? "";
    if (t && (t === n || t.startsWith(n))) return t;
  }
  const trainers = store.trainers ?? {};
  if (trainers[n]) return n;
  const year = raceDate.slice(0, 4);
  const candidates = Object.keys(trainers).filter((k) => k.startsWith(n))
    .map((key) => ({ key, n: trainers[key].filter((r) => r[0].slice(0, 4) === year && r[0] < raceDate).length }))
    .sort((a, b) => b.n - a.n);
  if (!candidates.length || (candidates.length > 1 && candidates[0].n === candidates[1].n)) return null;
  return candidates[0].key;
}

export type TrainerStats = { n: number; top3: number; venueN: number; venueTop3: number };

/** 今年（レースと同じ暦年）のレース日より前の出走を集計. */
export function trainerStats(store: Store, race: RaceInput, trainerKey: string | null | undefined): TrainerStats {
  const year = race.date.slice(0, 4);
  const entries = (trainerKey ? store.trainers?.[trainerKey] ?? [] : []).filter((r) => r[0] < race.date && r[0].slice(0, 4) === year);
  const venue = entries.filter((r) => r[1] === race.venue);
  return {
    n: entries.length, top3: entries.reduce((a, r) => a + r[2], 0),
    venueN: venue.length, venueTop3: venue.reduce((a, r) => a + r[2], 0),
  };
}

/** 同じコースでの今回の枠の成績: 出走数と、3着内率 − コース全体の3着内率. */
export function wakuCourse(store: Store, race: RaceInput, waku: number | null): { n: number; diff: number } {
  const entries = (store.courseWaku?.[`${race.venue}-${race.surface}-${race.distance}`] ?? []).filter((e) => e[0] < race.date);
  if (!waku || !entries.length) return { n: 0, diff: NaN };
  let n = 0, t = 0, totalN = 0, totalT = 0;
  for (const e of entries) {
    totalN += e[2]; totalT += e[3];
    if (e[1] === waku) { n += e[2]; t += e[3]; }
  }
  return { n, diff: n && totalN ? t / n - totalT / totalN : NaN };
}

/** 新しい順の値の列 → 直近加重平均・最高・前走・傾き（古い→新しい、3走以上）・ばらつき（標準偏差、3走以上）・走数. */
export function summarizeAbility(vals: number[]): Record<"wmean" | "best" | "last" | "trend" | "std" | "n", number> {
  const n = vals.length;
  if (!n) return { wmean: NaN, best: NaN, last: NaN, trend: NaN, std: NaN, n: 0 };
  let ws = 0;
  let wv = 0;
  for (let i = 0; i < n; i++) { const w = AB_DECAY ** i; ws += w; wv += w * vals[i]; }
  const out = { wmean: wv / ws, best: Math.max(...vals), last: vals[0], trend: NaN, std: NaN, n };
  if (n >= 3) {
    const xs = vals.map((_, i) => n - 1 - i);
    let mx = 0;
    let my = 0;
    for (let i = 0; i < n; i++) { mx += xs[i]; my += vals[i]; }
    mx /= n;
    my /= n;
    let sxy = 0;
    let sxx = 0;
    let syy = 0;
    for (let i = 0; i < n; i++) { sxy += (xs[i] - mx) * (vals[i] - my); sxx += (xs[i] - mx) ** 2; syy += (vals[i] - my) ** 2; }
    out.trend = sxy / sxx;
    out.std = Math.sqrt(syy / (n - 1));
  }
  return out;
}

/** 出走馬の中での 順位÷頭数・偏差（平均・母標準偏差で標準化）・平均との差・最高との差（・出走馬のばらつき）. */
function addRelative(rows: Features[], key: string, withFstd: boolean): void {
  const vals = rows.map((r) => r[key]).filter((x) => !Number.isNaN(x));
  let sum = 0;
  for (const x of vals) sum += x;
  const m = vals.length ? sum / vals.length : NaN;
  let sq = 0;
  for (const x of vals) sq += (x - m) ** 2;
  const sd = vals.length > 1 ? Math.sqrt(sq / vals.length) : NaN;
  const top = vals.length ? Math.max(...vals) : NaN;
  for (const r of rows) {
    const v = r[key];
    const ok = !Number.isNaN(v);
    r[key + "_rank"] = ok ? (1 + vals.filter((x) => x > v).length) / rows.length : NaN;
    r[key + "_z"] = ok && sd > 0 ? (v - m) / sd : NaN;
    r[key + "_dmean"] = ok ? v - m : NaN;
    r[key + "_dmax"] = ok ? v - top : NaN;
    if (withFstd) r[key + "_fstd"] = sd;
  }
}

export function runnerFeatures(store: Store, race: RaceInput, runner: RunnerInput): Features {
  const nz = (v: number | null) => (v === null || v === 0 ? NaN : v); // Python の `x or NAN` と同じ
  const f: Features = {
    surface: race.surface,
    distance: race.distance ?? NaN,
    going: race.going ?? NaN,
    field_size: race.fieldSize,
    venue: race.venue,
    prize_1st: race.prize1stMan ?? NaN,
    waku: nz(runner.waku), umaban: nz(runner.umaban), weight: nz(runner.weight), age: nz(runner.age),
    sex: runner.sex !== null && runner.sex in SEX ? SEX[runner.sex] : NaN,
    body_weight: nz(runner.bodyWeight),
    body_weight_diff: runner.bodyWeightDiff ?? NaN,
    train_rank: nz(runner.trainRank ?? null),
  };
  for (const k of ["ped_heavy", "ped_wet", "ped_top3", "ped_dist"] as const) f[k] = runner.pedigree?.[k] ?? NaN;
  f.umaban_ratio = Number.isNaN(f.umaban) ? NaN : f.umaban / race.fieldSize;
  Object.assign(f, horseFeatures(store, race, runner.horse, runner.extraRuns));
  const tr = trainerStats(store, race, runner.trainerKey);
  Object.assign(f, {
    trainer_n: tr.n, trainer_top3: rate(tr.top3, tr.n), trainer_venue_n: tr.venueN, trainer_venue_top3: rate(tr.venueTop3, tr.venueN),
  });
  const wc = wakuCourse(store, race, runner.waku);
  f.waku_course_n = wc.n;
  f.waku_course_diff = wc.diff;
  const prior = horseRuns(store, runner.horse, runner.extraRuns).filter((r) => r[0] < race.date);
  const last = prior.length ? prior[prior.length - 1] : null;
  f.weight_change = last && runner.weight && last[12] ? runner.weight - last[12] : NaN;
  const lastJockey = last ? store.names?.[last[13]] ?? "" : "";
  f.jockey_change = last && runner.jockeyKey && lastJockey ? (runner.jockeyKey === lastJockey ? 0 : 1) : NaN;
  const j = jockeyStats(store, race, runner.jockeyKey);
  Object.assign(f, {
    jockey_n: j.n, jockey_top3: rate(j.top3, j.n),
    jockey_venue_n: j.venueN, jockey_venue_top3: rate(j.venueTop3, j.venueN),
    jockey_course_n: j.courseN, jockey_course_top3: rate(j.courseTop3, j.courseN),
  });
  return f;
}

/** 出走馬全体の先行力（逃げ候補・先行馬の頭数）と、その中での位置取りの順位. */
export function addFieldFeatures(rows: Features[]): void {
  const nige = rows.filter((r) => !Number.isNaN(r.nige_rate) && r.nige_rate >= NIGE_CAND_RATE).length;
  const front = rows.filter((r) => !Number.isNaN(r.pos_first_mean) && r.pos_first_mean <= FRONT_RATIO).length;
  const known = rows.map((r) => r.pos_first_mean).filter((v) => !Number.isNaN(v));
  for (const r of rows) {
    r.field_nige_cands = nige;
    r.field_front = front;
    r.field_front_ratio = front / rows.length;
    const v = r.pos_first_mean;
    r.rel_pos_rank = Number.isNaN(v) ? NaN : (1 + known.filter((x) => x < v).length) / rows.length;
  }
  // 今回の出走馬の中での比較（タイム指数・クラスの、出走馬平均との差と順位÷頭数、出走馬全体の調子）
  for (const key of ["tidx_mean5", "tidx_best5", "prize_ratio_mean5"]) {
    const vals = rows.map((r) => r[key]).filter((x) => !Number.isNaN(x));
    let sum = 0;
    for (const x of vals) sum += x;
    const m = vals.length ? sum / vals.length : NaN;
    for (const r of rows) {
      const x = r[key];
      r["rel_" + key] = Number.isNaN(x) ? NaN : x - m;
      r["rank_" + key] = Number.isNaN(x) ? NaN : (1 + vals.filter((y) => y > x).length) / rows.length;
    }
  }
  const forms = rows.map((r) => r.form_raw).filter((x) => !Number.isNaN(x));
  let formSum = 0;
  for (const x of forms) formSum += x;
  for (const r of rows) r.field_form_mean = forms.length ? formSum / forms.length : NaN;
  // 能力指数・推定上がりの、出走馬の中での比較
  addRelative(rows, "ab_wmean", true);
  addRelative(rows, "ab_best", false);
  addRelative(rows, "ag_dev_wmean", true);
  // 調教評価の、出走馬の平均との差と、自分より評価が高い馬の数÷頭数
  const ranks = rows.map((r) => r.train_rank).filter((x) => !Number.isNaN(x));
  let rankSum = 0;
  for (const x of ranks) rankSum += x;
  const mr = ranks.length ? rankSum / ranks.length : NaN;
  for (const r of rows) {
    const v = r.train_rank;
    r.train_rank_rel = Number.isNaN(v) ? NaN : v - mr;
    r.train_rank_pos = Number.isNaN(v) ? NaN : ranks.filter((x) => x > v).length / rows.length;
  }
}

/** 想定ペース（表示用）。モデルには頭数・割合の数値を入れている. */
export function expectedPace(nigeCands: number, frontRatio: number): "ハイ" | "ミドル" | "スロー" {
  if (nigeCands >= 2 || frontRatio >= 0.4) return "ハイ";
  if (nigeCands === 0 && frontRatio < 0.25) return "スロー";
  return "ミドル";
}

/** 脚質（表示用）. */
export function runningStyle(f: Features): string | null {
  if (Number.isNaN(f.pos_first_mean)) return null;
  if (f.nige_rate >= NIGE_CAND_RATE) return "逃げ";
  if (f.pos_first_mean <= FRONT_RATIO) return "先行";
  if (f.pos_first_mean <= 2 / 3) return "差し";
  return "追込";
}

/** 同じレースの出走馬の中で form_raw を偏差値化（平均50・標準偏差10, 母標準偏差）. */
export function addFormDeviation(rows: Features[]): void {
  const vals = rows.map((r) => r.form_raw).filter((v) => !Number.isNaN(v));
  if (vals.length < 2) {
    for (const r of rows) r.form_dev = Number.isNaN(r.form_raw) ? NaN : 50;
    return;
  }
  let sum = 0;
  for (const v of vals) sum += v;
  const mean = sum / vals.length;
  let sq = 0;
  for (const v of vals) sq += (v - mean) ** 2;
  const sd = Math.sqrt(sq / vals.length);
  for (const r of rows) {
    const v = r.form_raw;
    r.form_dev = Number.isNaN(v) ? NaN : sd === 0 ? 50 : 50 + 10 * (v - mean) / sd;
  }
}
