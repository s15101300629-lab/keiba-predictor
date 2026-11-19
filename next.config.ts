import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  // AI予測のモデル・特徴量データ（scripts/ml/train.py が生成）をサーバー側で読むため同梱する
  outputFileTracingIncludes: {
    "/api/races/[raceId]": ["./data/ml/model.json", "./data/ml/store.json", "./data/ml/metrics.json", "./data/ml/stack.json", "./data/ml/heavy_adjustment.json", "./data/ml/course_waku.json"],
  },
};

export default nextConfig;
