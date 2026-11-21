
"use client";

import { useState } from "react";

export default function LoginPage() {
  const [password, setPassword] = useState("");
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);

  async function handleSubmit(
    event: React.FormEvent<HTMLFormElement>
  ) {
    event.preventDefault();
    setError("");
    setLoading(true);

    try {
      const response = await fetch("/api/login", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ password }),
      });

      if (!response.ok) {
        setError(
          response.status === 401
            ? "パスワードが違います。"
            : "認証に失敗しました。もう一度お試しください。"
        );
        return;
      }

      window.location.assign("/");
    } catch {
      setError("通信に失敗しました。もう一度お試しください。");
    } finally {
      setLoading(false);
    }
  }

  return (
    <main className="login-page">
      <div className="login-card">
        <h1 className="login-logo">
          <img src="/logo.png" alt="Racehorse Insight Analytics" width={224} height={224} />
        </h1>

        <p className="login-lead">
          利用するにはパスワードを入力してください。
        </p>

        <form onSubmit={handleSubmit}>
          <label
            htmlFor="password"
            className="login-label"
          >
            パスワード
          </label>

          <input
            id="password"
            type="password"
            autoComplete="current-password"
            value={password}
            onChange={(event) => setPassword(event.target.value)}
            required
            className="login-input"
          />

          {error && (
            <p className="login-error">
              {error}
            </p>
          )}

          <button
            type="submit"
            disabled={loading}
            className="login-button"
          >
            {loading ? "確認中..." : "ログイン"}
          </button>
        </form>
      </div>
    </main>
  );
}
