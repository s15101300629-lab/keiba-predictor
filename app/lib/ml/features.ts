// scripts/ml/features.py と同じ定義の特徴量計算。どちらかを変えたら両方合わせ、
// `npm run ml:parity` で Python と一致することを確認すること。

export type Store = {
  lastDate: string;
  // [date, venue, surface, distance, going(-1=不明), field, finish(-1=中止), timeDiff(-1=不明)]
  horses: Record<string, [string, number, number, number | null, number, number, number, number][]>;
  // [date, venue, surface, distance, top3]
  jockeys: Record<string, [string, number, number, number | null, number][]>;
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
  waku: number | null; umaban: number | null; weight: number | null;
  age: number | null; sex: string | null;
  bodyWeight: number | null; bodyWeightDiff: number | null;
};

export type Features = Record<string, number>;

export const FORM_RUNS = 5;
export const TIME_DIFF_SCALE = 2;
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

function horseFeatures(store: Store, race: RaceInput, horse: string): Features {
  const runs = (store.horses[horse] ?? [])
    .filter((r) => r[0] < race.date)
    .map((r) => ({
      date: r[0], venue: r[1], surface: r[2], distance: r[3],
      going: r[4] < 0 ? null : r[4], field: r[5],
      finish: r[6] < 0 ? null : r[6], timeDiff: r[7] < 0 ? null : r[7],
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
  agg("heavy", (r) => r.surface === s && r.going !== null && r.going >= 2);
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
  };
  f.umaban_ratio = Number.isNaN(f.umaban) ? NaN : f.umaban / race.fieldSize;
  Object.assign(f, horseFeatures(store, race, runner.horse));
  const j = jockeyStats(store, race, runner.jockeyKey);
  Object.assign(f, {
    jockey_n: j.n, jockey_top3: rate(j.top3, j.n),
    jockey_venue_n: j.venueN, jockey_venue_top3: rate(j.venueTop3, j.venueN),
    jockey_course_n: j.courseN, jockey_course_top3: rate(j.courseTop3, j.courseN),
  });
  return f;
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
