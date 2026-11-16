import Link from "next/link";
const races = [
  { time: "10:05", venue: "東京", number: "1R", title: "2歳未勝利", surface: "ダート 1400m", runners: 16, status: "出馬表" },
  { time: "10:35", venue: "東京", number: "2R", title: "2歳未勝利", surface: "芝 1600m", runners: 14, status: "出馬表" },
  { time: "11:05", venue: "京都", number: "3R", title: "2歳未勝利", surface: "芝 1800m", runners: 12, status: "出馬表" },
];

export default function Home() {
  return (
    <main className="shell">
      <header className="topbar">
        <div className="brand"><span className="brand-mark">K</span><div><strong>KEIBA PREDICTOR</strong><small>PERSONAL JRA ANALYSIS</small></div></div>
        <span className="private-pill">PRIVATE</span>
      </header>

      <section className="hero">
        <p className="eyebrow">YOUR RACING WORKSPACE</p>
        <h1>競馬予想を、<br /><span>データで組み立てる。</span></h1>
        <p className="hero-copy">レース情報、各馬の評価、予想と結果を一元管理する自分専用の分析環境。</p>
        <div className="hero-stats">
          <div><small>対象</small><b>JRA</b></div>
          <div><small>分析モード</small><b>準備中</b></div>
          <div><small>データ連携</small><b>未接続</b></div>
        </div>
      </section>

      <section className="section-head">
        <div><p className="eyebrow">RACE CENTER</p><h2>レース一覧</h2></div>
        <span className="date-chip">サンプル画面</span>
      </section>
      <div className="notice">現在は画面確認用のサンプルデータです。実際の開催情報・出馬表・オッズはまだ取得していません。</div>

      <section className="race-list">
        {races.map((race) => (
          <article className="race-card" key={`${race.venue}-${race.number}`}>
            <div className="race-time">{race.time}<span>{race.venue}</span></div>
            <div className="race-info"><div className="race-title"><b>{race.number}</b>{race.title}</div><p>{race.surface} ・ {race.runners}頭</p></div>
            <span className="race-status">{race.status}</span>
          </article>
        ))}
      </section>

      <section className="section-head lower"><div><p className="eyebrow">ANALYSIS TOOLS</p><h2>予想・分析</h2></div></section>
      <div className="tool-grid">
        <article className="tool-card"><span>01</span><b>馬ごとの評価</b><p>過去成績・適性・展開を評価</p></article>
        <article className="tool-card"><span>02</span><b>印と買い目</b><p>評価をもとに予想を整理</p></article>
        <article className="tool-card"><span>03</span><b>結果検証</b><p>的中率・回収率を記録</p></article>
        <article className="tool-card"><span>04</span><b>データ管理</b><p>予想履歴と分析結果を保存</p></article>
      </div>
<Link href="/import" className="notice" style={{ display: "block", marginTop: 16 }}>データ取り込み（出馬表・血統・過去戦績）を開く →</Link>
      <footer>KEIBA PREDICTOR <span>個人利用向け・開発中</span></footer>
    </main>
  );
}
