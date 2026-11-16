"use client";

import Link from "next/link";
import { useEffect, useState } from "react";
import styles from "./import.module.css";
import { parseRaceCard } from "./parseRaceCard";

type Horse = {
  waku: number;
  umaban: number;
  name: string;
  sexAge: string;
  weight: string;
  jockey: string;
  sire: string;
  dam: string;
  damSire: string;
  history: string;
};

type Card = { raceName: string; horses: Horse[]; savedAt: string };

const KEY = "keiba:cards:v1";

function readCards(): Card[] {
  try {
    const v = localStorage.getItem(KEY);
    return v ? (JSON.parse(v) as Card[]) : [];
  } catch {
    return [];
  }
}

function writeCards(cards: Card[]): boolean {
  try {
    localStorage.setItem(KEY, JSON.stringify(cards));
    return true;
  } catch {
    return false;
  }
}

export default function ImportPage() {
  const [raceName, setRaceName] = useState("");
  const [raw, setRaw] = useState("");
  const [horses, setHorses] = useState<Horse[]>([]);
  const [failed, setFailed] = useState<string[]>([]);
  const [cards, setCards] = useState<Card[]>([]);
  const [message, setMessage] = useState("");

  useEffect(() => {
    setCards(readCards());
  }, []);

  function handleParse() {
    const { rows, failed } = parseRaceCard(raw);
    setFailed(failed);
    if (rows.length === 0) {
      setMessage("馬を1頭も読み取れませんでした。貼り付けた文字に「牡3」「牝4」のような性齢が含まれているか確認してください。");
      return;
    }
    setHorses((prev) =>
      rows.map((r) => {
        const old = prev.find((h) => h.umaban === r.umaban && h.name === r.name);
        return {
          ...r,
          sire: old?.sire ?? "",
          dam: old?.dam ?? "",
          damSire: old?.damSire ?? "",
          history: old?.history ?? "",
        };
      })
    );
    setMessage(`${rows.length}頭を読み取りました。読み取れなかった候補: ${failed.length}件`);
  }

  function updateHorse(umaban: number, patch: Partial<Horse>) {
    setHorses((prev) => prev.map((h) => (h.umaban === umaban ? { ...h, ...patch } : h)));
  }

  function handleSave() {
    const name = raceName.trim();
    if (!name) {
      setMessage("レース名を入力してから保存してください。");
      return;
    }
    if (horses.length === 0) {
      setMessage("保存する馬がありません。先に「読み取る」を押してください。");
      return;
    }
    const next: Card[] = [
      { raceName: name, horses, savedAt: new Date().toISOString() },
      ...cards.filter((c) => c.raceName !== name),
    ];
    if (writeCards(next)) {
      setCards(next);
      setMessage(`「${name}」を保存しました（${horses.length}頭）。`);
    } else {
      setMessage("保存に失敗しました。Safariのプライベートブラウズを使っていないか確認してください。");
    }
  }

  function handleLoad(card: Card) {
    setRaceName(card.raceName);
    setHorses(card.horses);
    setFailed([]);
    setMessage(`「${card.raceName}」を読み込みました。`);
  }

  function handleDelete(name: string) {
    const next = cards.filter((c) => c.raceName !== name);
    if (writeCards(next)) {
      setCards(next);
      setMessage(`「${name}」を削除しました。`);
    }
  }

  return (
    <main className={styles.wrap}>
      <Link href="/" className={styles.back}>← トップへ戻る</Link>
      <h1 className={styles.h1}>データ取り込み</h1>
      <p className={styles.lead}>
        出馬表のページで文字をコピーし、下に貼り付けて「読み取る」を押します。保存先はこのiPhoneのSafari内です。
      </p>

      <label className={styles.label}>レース名（例: エリザベス女王杯）</label>
      <input className={styles.input} value={raceName} onChange={(e) => setRaceName(e.target.value)} />

      <label className={styles.label}>出馬表の貼り付け</label>
      <textarea className={styles.textarea} rows={8} value={raw} onChange={(e) => setRaw(e.target.value)} />
      <button className={styles.btn} onClick={handleParse}>読み取る</button>

      {message && <p className={styles.message}>{message}</p>}

      {failed.length > 0 && (
        <div className={styles.warn}>
          <b>読み取れなかった候補（捨てずに表示しています）</b>
          {failed.map((f, i) => (
            <p key={i}>{f}</p>
          ))}
        </div>
      )}

      {horses.map((h) => (
        <section key={h.umaban} className={styles.horse}>
          <div className={styles.horseHead}>
            <span className={styles.badge}>{h.waku}枠 {h.umaban}番</span>
            <b>{h.name}</b>
            <span className={styles.sub}>{h.sexAge} {h.weight}kg {h.jockey}</span>
          </div>
          <div className={styles.pedigree}>
            <input className={styles.input} placeholder="父" value={h.sire} onChange={(e) => updateHorse(h.umaban, { sire: e.target.value })} />
            <input className={styles.input} placeholder="母" value={h.dam} onChange={(e) => updateHorse(h.umaban, { dam: e.target.value })} />
            <input className={styles.input} placeholder="母父" value={h.damSire} onChange={(e) => updateHorse(h.umaban, { damSire: e.target.value })} />
          </div>
          <textarea
            className={styles.textarea}
            rows={4}
            placeholder="過去戦績を貼り付け"
            value={h.history}
            onChange={(e) => updateHorse(h.umaban, { history: e.target.value })}
          />
        </section>
      ))}

      {horses.length > 0 && <button className={styles.btn} onClick={handleSave}>保存する</button>}

      <h2 className={styles.h2}>保存済みレース</h2>
      {cards.length === 0 && <p className={styles.sub}>まだありません。</p>}
      {cards.map((c) => (
        <div key={c.raceName} className={styles.saved}>
          <span>{c.raceName}（{c.horses.length}頭）</span>
          <span>
            <button className={styles.small} onClick={() => handleLoad(c)}>開く</button>
            <button className={styles.small} onClick={() => handleDelete(c.raceName)}>削除</button>
          </span>
        </div>
      ))}
    </main>
  );
}
