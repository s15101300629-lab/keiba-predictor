// 表示用の調子偏差値（近5走）。netkeiba の戦績から、着順・勝ち馬とのタイム差・上がり3F で計算する。
// モデルの特徴量（app/lib/ml/features.ts の form_dev）は成績表ベースで上がり3F を含まない点が違う。
import { FORM_RUNS, runComponents } from "./ml/features";

export type FormRun = {
  date: string; race: string; finish: number | null; field: number | null;
  timeDiff: number | null; agari: number | null; raceLast3f: number | null;
  finishC: number; timeC: number; agariC: number | null; score: number;
};
export type FormDisplay = { runs: FormRun[]; raw: number | null; deviation: number | null };

const AGARI_SCALE = 2; // レースの上がりより1秒速い → 1, 同じ → 0.5, 1秒遅い → 0

function value(values: Record<string, string>, key: string) {
  const entry = Object.entries(values).find(([k]) => k.replace(/\s+/g, "") === key);
  return entry?.[1]?.trim() ?? "";
}
const toNum = (s: string) => (/^-?\d+(\.\d+)?$/.test(s) ? Number(s) : null);

export function formFromHistory(performances: { values: Record<string, string> }[], raceDate: string | null): FormDisplay {
  const runs: FormRun[] = [];
  for (const row of performances) {
    const v = row.values;
    const date = value(v, "日付").replace(/\//g, "-");
    if (!/^\d{4}-\d{2}-\d{2}$/.test(date) || (raceDate && date >= raceDate)) continue;
    const finishText = value(v, "着順");
    if (/取|除/.test(finishText) || finishText === "") continue; // 出走していない
    const finish = toNum(finishText);
    const field = toNum(value(v, "頭数"));
    const margin = toNum(value(v, "着差"));
    const agari = toNum(value(v, "上り"));
    const pace = value(v, "ペース").match(/-(\d+\.\d)$/);
    const raceLast3f = pace ? Number(pace[1]) : null;
    const timeDiff = margin === null ? null : Math.max(0, margin);
    const [finishC, timeC] = runComponents(finish, field ?? 0, timeDiff);
    const agariC = finish === null ? 0 : agari !== null && raceLast3f !== null
      ? Math.min(1, Math.max(0, 0.5 - (agari - raceLast3f) / AGARI_SCALE)) : null;
    const parts = [finishC, timeC, ...(agariC === null ? [] : [agariC])];
    runs.push({
      date, race: value(v, "レース名"), finish, field, timeDiff, agari, raceLast3f,
      finishC, timeC, agariC, score: parts.reduce((a, b) => a + b, 0) / parts.length,
    });
  }
  runs.sort((a, b) => (a.date < b.date ? 1 : -1));
  const recent = runs.slice(0, FORM_RUNS);
  return {
    runs: recent,
    raw: recent.length ? recent.reduce((a, r) => a + r.score, 0) / recent.length : null,
    deviation: null,
  };
}

export function addDisplayDeviation(forms: FormDisplay[]): void {
  const vals = forms.map((f) => f.raw).filter((v): v is number => v !== null);
  if (vals.length < 2) return;
  const mean = vals.reduce((a, b) => a + b, 0) / vals.length;
  const sd = Math.sqrt(vals.reduce((a, b) => a + (b - mean) ** 2, 0) / vals.length);
  for (const f of forms) if (f.raw !== null) f.deviation = sd === 0 ? 50 : 50 + 10 * (f.raw - mean) / sd;
}
