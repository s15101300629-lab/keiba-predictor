import { readdirSync, readFileSync } from "node:fs";
import path from "node:path";
import ResultsView, { type Day } from "./ResultsView";

// AI成績：data/ml/day_check/日付.json（scripts/ml/check_jra_day.py が毎日作る）を読み、AI 予想の順位と実際の着順を並べる
export const metadata = { title: "AI成績 | Racehorse Insight Analytics" };

function loadDays(): Day[] {
  const dir = path.join(process.cwd(), "data", "ml", "day_check");
  let files: string[] = [];
  try { files = readdirSync(dir).filter((f) => /^\d{4}-\d{2}-\d{2}\.json$/.test(f)); } catch { return []; }
  return files.sort().reverse().map((f) => {
    const d = JSON.parse(readFileSync(path.join(dir, f), "utf8")) as Day;
    return { date: d.date, detail: d.detail };
  });
}

export default function ResultsPage() {
  return <ResultsView days={loadDays()} />;
}
