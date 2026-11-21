// 全画面共通のフッター：データの出どころ・学習データの最終日（ビルド時に next.config.ts が入れる）・注意書き
export default function SiteFooter() {
  const until = process.env.NEXT_PUBLIC_DATA_UNTIL;
  return <footer className="site-footer">
    <div className="site-footer-inner">
      <p>データ：JRA 成績表（学習・過去走）、netkeiba（出馬表・戦績・血統・調教評価・オッズ）</p>
      <p>学習データ：{until ? `〜${until.replace(/-/g, "/")}` : "-"}</p>
      <p className="site-footer-warn">AIの確率は過去のデータから計算した目安で、的中を保証するものではない。馬券の購入は自己責任で。</p>
      <p className="site-footer-brand">RACEHORSE INSIGHT ANALYTICS</p>
    </div>
  </footer>;
}
