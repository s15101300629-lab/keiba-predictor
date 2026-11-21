import type { NextConfig } from "next";
import { createHash } from "node:crypto";
import { readFileSync } from "node:fs";

// 学習データの最終日（フッターに出す）。ビルドのときに store.json の先頭から読む
let dataUntil = "";
try { dataUntil = readFileSync("./data/ml/store.json", "utf8").slice(0, 200).match(/"lastDate"\s*:\s*"([\d-]+)"/)?.[1] ?? ""; } catch { /* なければ空 */ }

// モデルの版（model.json の中身のハッシュ）。再学習でモデルが変わると、端末に保存したレース詳細（AI予想）を使わず取り直す
let modelVersion = "";
try { modelVersion = createHash("sha1").update(readFileSync("./data/ml/model.json")).digest("hex").slice(0, 10); } catch { /* なければ空 */ }

const nextConfig: NextConfig = {
  env: { NEXT_PUBLIC_DATA_UNTIL: dataUntil, NEXT_PUBLIC_MODEL_VERSION: modelVersion },
  // AI予測のモデル・特徴量データ（scripts/ml/train.py が生成）をサーバー側で読むため同梱する
  outputFileTracingIncludes: {
    "/api/races/[raceId]": ["./data/ml/model.json", "./data/ml/store.json", "./data/ml/metrics.json", "./data/ml/stack.json", "./data/ml/heavy_adjustment.json", "./data/ml/course_waku.json", "./data/ml/model_win.json", "./data/ml/model_top2.json"],
  },
};

export default nextConfig;
