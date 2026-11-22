import { readFile } from "node:fs/promises";
import path from "node:path";
import { NextResponse } from "next/server";
import { longshotRatios, type LongshotInput, type LongshotModel } from "@/app/lib/longshot";

export const runtime = "nodejs";

// 人気薄専用の2段目（AI印の☆）。画面がオッズを取ったあとに、各馬の本体モデルの入力・確率と単勝オッズを送ってくる
// 返すもの：6番人気以下の馬の 2段目の確率・補正した市場の確率・倍率（☆は倍率 ≥ threshold）
let modelPromise: Promise<LongshotModel | null> | null = null;
function loadModel() {
  modelPromise ??= readFile(path.join(process.cwd(), "data", "ml", "model_longshot.json"), "utf8")
    .then((text) => JSON.parse(text) as LongshotModel)
    .catch(() => { modelPromise = null; return null; });
  return modelPromise;
}

type Body = { rows?: LongshotInput[]; odds?: Record<string, number> };

export async function POST(request: Request) {
  const model = await loadModel();
  if (!model) return NextResponse.json({ ok: false, error: "model_longshot.json がない" }, { status: 503 });
  const body = (await request.json().catch(() => null)) as Body | null;
  const rows = (body?.rows ?? []).filter((r) => Number.isInteger(r?.n) && Array.isArray(r.x)
    && r.x.length === model.features.length - 6 && typeof r.top3 === "number" && typeof r.win === "number");
  const odds = new Map(Object.entries(body?.odds ?? {}).map(([k, v]) => [Number(k), Number(v)] as const)
    .filter(([n, v]) => Number.isInteger(n) && Number.isFinite(v) && v > 0));
  if (rows.length < 3 || odds.size < 3) return NextResponse.json({ ok: false, error: "入力が足りない" }, { status: 400 });
  const results = longshotRatios(model, rows, odds);
  return NextResponse.json({
    ok: true, threshold: model.threshold, minPop: model.min_pop,
    horses: Object.fromEntries([...results].map(([n, r]) => [n, { popularity: r.popularity, probability: r.probability, market: r.market, ratio: r.ratio }])),
  });
}
