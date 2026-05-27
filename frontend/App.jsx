// ============================================================
// Smart Money Tracker
// Copyright (c) 2026 Dhruv Patel. All rights reserved.
//
// This software is proprietary and confidential.
// Unauthorized copying, distribution, or modification
// of this file, via any medium, is strictly prohibited.
//
// Author:  Dhruv Patel
// GitHub:  github.com/dhruvpatel29
// Email:   dhruvkumarp79@gmail.com
// ============================================================


import DayTradingDashboard from "./DayTradingDashboard";
// frontend/src/App.jsx
// Full React frontend with TradingView chart embed + live AI signals
//
// Setup:
//   npx create-react-app smart-money-frontend
//   cd smart-money-frontend
//   npm install
//   Replace src/App.jsx with this file
//   npm start

import { useState, useEffect, useRef, useCallback, lazy, Suspense } from "react";

const API_BASE = "http://localhost:8000";
const TICKERS  = ["AAPL", "NVDA", "TSLA", "BTC-USD", "SPY", "AMZN"];

// ── TradingView Chart Widget ────────────────────────────────────
// This embeds a full TradingView chart with all your indicators.
// FREE for everyone — no account needed for the widget.
// Your TradingView account's Pine Script runs in YOUR browser session.

function TradingViewChart({ ticker }) {
  const containerRef = useRef(null);
  const widgetRef    = useRef(null);

  useEffect(() => {
    // Clean up previous widget
    if (containerRef.current) {
      containerRef.current.innerHTML = "";
    }

    // The TradingView widget script
    const script = document.createElement("script");
    script.src   = "https://s3.tradingview.com/external-embedding/embed-widget-advanced-chart.js";
    script.async = true;
    script.innerHTML = JSON.stringify({
      // Chart configuration
      autosize:           true,
      symbol:             ticker === "BTC-USD" ? "BINANCE:BTCUSDT" : `NASDAQ:${ticker}`,
      interval:           "D",           // Daily candles
      timezone:           "America/New_York",
      theme:              "light",
      style:              "1",           // Candlestick
      locale:             "en",
      toolbar_bg:         "#f1f3f6",
      enable_publishing:  false,
      allow_symbol_change: false,
      save_image:         false,
      container_id:       "tv_chart",
      height:             420,

      // Studies (indicators) shown on the chart
      // These are TradingView's built-in indicators
      studies: [
        "STD;200_EMA",          // 200 EMA (use SMA in settings)
        "STD;MA%Ribbon",        // Moving Average Ribbon (shows 50+200 SMA)
        "STD;Bollinger_Bands",  // Bollinger Bands
        "STD;VWAP",             // VWAP
        "STD;RSI",              // RSI (appears in sub-chart below)
        "STD;MACD",             // MACD (appears in sub-chart below)
        "STD;Volume",           // Volume bars
      ],

      // Show the studies panel so user can toggle indicators
      studies_overrides: {
        "volume.volume.color.0":  "#ef4444",
        "volume.volume.color.1":  "#22c55e",
        "bollinger bands.upper.color":  "#6366f1",
        "bollinger bands.lower.color":  "#6366f1",
        "bollinger bands.basis.color":  "#6366f1",
      },
    });

    const container = containerRef.current;
    if (container) {
      container.appendChild(script);
      widgetRef.current = script;
    }

    return () => {
      if (container) container.innerHTML = "";
    };
  }, [ticker]);

  return (
    <div
      ref={containerRef}
      id="tv_chart"
      style={{ height: 420, width: "100%" }}
    />
  );
}


// ── Signal card components ──────────────────────────────────────

function PredictionGauge({ confidence, direction }) {
  const color = direction === "bullish" ? "#22c55e"
              : direction === "bearish" ? "#ef4444"
              : "#f59e0b";
  const pct  = Math.round(confidence * 100);

  return (
    <div style={{ textAlign: "center", padding: "12px 0" }}>
      <svg viewBox="0 0 120 70" width="120" height="70">
        {/* Background arc */}
        <path
          d="M10 65 A50 50 0 0 1 110 65"
          fill="none" stroke="#e5e7eb" strokeWidth="10" strokeLinecap="round"
        />
        {/* Filled arc */}
        <path
          d="M10 65 A50 50 0 0 1 110 65"
          fill="none"
          stroke={color}
          strokeWidth="10"
          strokeLinecap="round"
          strokeDasharray="157"
          strokeDashoffset={157 - 157 * confidence}
          style={{ transition: "stroke-dashoffset 0.8s ease" }}
        />
        <text x="60" y="60" textAnchor="middle" fontSize="18" fontWeight="500" fill="#111">
          {pct}%
        </text>
      </svg>
      <div style={{
        fontSize: 15, fontWeight: 500, color,
        marginTop: 4, textTransform: "capitalize"
      }}>
        {direction === "bullish" ? "▲" : direction === "bearish" ? "▼" : "—"} {direction}
      </div>
    </div>
  );
}

function SignalBar({ label, value, max = 1, color = "#6366f1" }) {
  const pct = Math.min(Math.abs(value) / max * 100, 100);
  return (
    <div style={{ marginBottom: 8 }}>
      <div style={{ display: "flex", justifyContent: "space-between",
                    fontSize: 11, color: "#6b7280", marginBottom: 3 }}>
        <span>{label}</span>
        <span style={{ fontWeight: 500, color: "#111" }}>
          {typeof value === "number" ? value.toFixed(2) : value}
        </span>
      </div>
      <div style={{ height: 5, background: "#f3f4f6", borderRadius: 99, overflow: "hidden" }}>
        <div style={{ height: 5, width: `${pct}%`, background: color,
                      borderRadius: 99, transition: "width 0.6s ease" }} />
      </div>
    </div>
  );
}

function Badge({ text, color, bg }) {
  return (
    <span style={{
      fontSize: 10, padding: "2px 8px", borderRadius: 99,
      background: bg, color, border: `0.5px solid ${color}`,
      fontWeight: 500, whiteSpace: "nowrap"
    }}>
      {text}
    </span>
  );
}


// ── Main App ────────────────────────────────────────────────────

export default function App() {
  const [mode,         setMode]         = useState("swing");
  const [activeTicker, setActiveTicker] = useState("AAPL");
  const [signal,       setSignal]       = useState(null);
  const [alerts,       setAlerts]       = useState([]);
  const [loading,      setLoading]      = useState(false);
  const [wsStatus,     setWsStatus]     = useState("connecting");
  const wsRef = useRef(null);

  // Fetch signal from REST API
  const fetchSignal = useCallback(async (ticker) => {
    setLoading(true);
    try {
      const res  = await fetch(`${API_BASE}/signal/${ticker}`);
      const data = await res.json();
      setSignal(data);
    } catch (e) {
      console.error("Signal fetch error:", e);
    } finally {
      setLoading(false);
    }
  }, []);

  // Connect WebSocket for real-time updates
  useEffect(() => {
    if (wsRef.current) wsRef.current.close();

    const ws = new WebSocket(`ws://localhost:8000/ws/${activeTicker}`);
    wsRef.current = ws;

    ws.onopen    = () => setWsStatus("live");
    ws.onclose   = () => setWsStatus("disconnected");
    ws.onerror   = () => setWsStatus("error");
    ws.onmessage = (e) => {
      const data = JSON.parse(e.data);
      // Check if this is a webhook-triggered unified signal
      if (data.all_agree !== undefined) {
        // It's a unified signal from TradingView webhook
        setSignal(prev => ({ ...prev, prediction: data }));
        if (data.confidence > 0.65 && data.direction !== "neutral") {
          setAlerts(prev => [{
            id:        Date.now(),
            ticker:    data.ticker,
            direction: data.direction,
            confidence:data.confidence,
            all_agree: data.all_agree,
            time:      new Date().toLocaleTimeString(),
          }, ...prev.slice(0, 9)]);
        }
      } else {
        // It's a regular scheduled update
        setSignal(data);
      }
    };

    fetchSignal(activeTicker);

    return () => ws.close();
  }, [activeTicker, fetchSignal]);

  const pred   = signal?.prediction ?? {};
  const inds   = signal?.indicators ?? {};
  const price  = signal?.price      ?? {};
  const sent   = signal?.sentiment  ?? {};

  const dirColor = pred.direction === "bullish" ? "#22c55e"
                 : pred.direction === "bearish" ? "#ef4444"
                 : "#f59e0b";

  if (mode === "daytrading") {
    return (
      <div>
        <div style={{ background: "#0d1117", borderBottom: "1px solid #21262d",
                      padding: "10px 20px", display: "flex", alignItems: "center", gap: 16 }}>
          <div style={{ display: "flex", gap: 4 }}>
            {TICKERS.map(t => (
              <button key={t} onClick={() => setActiveTicker(t)}
                style={{ fontSize: 11, padding: "3px 10px", borderRadius: 99, cursor: "pointer",
                  background: activeTicker === t ? "#1d4ed8" : "transparent",
                  color: activeTicker === t ? "#bfdbfe" : "#6b7280",
                  border: activeTicker === t ? "1px solid #1d4ed8" : "1px solid #374151" }}>
                {t}
              </button>
            ))}
          </div>
          <button onClick={() => setMode("swing")}
            style={{ fontSize: 11, padding: "4px 12px", borderRadius: 99, cursor: "pointer",
                     background: "#374151", color: "#d1d5db", border: "1px solid #4b5563",
                     marginLeft: "auto" }}>
            ← Swing Trading
          </button>
        </div>
        <DayTradingDashboard ticker={activeTicker} />
      </div>
    );
  }

  return (
    <div style={{ fontFamily: "system-ui, sans-serif", background: "#f9fafb",
                  minHeight: "100vh", padding: "0 0 40px" }}>

      {/* ── Nav bar ── */}
      <div style={{ background: "#fff", borderBottom: "1px solid #e5e7eb",
                    padding: "12px 20px", display: "flex",
                    alignItems: "center", justifyContent: "space-between" }}>
        <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
          <div style={{ width: 30, height: 30, background: "#1a1a2e",
                        borderRadius: 8, display: "flex", alignItems: "center",
                        justifyContent: "center", fontSize: 16 }}>📈</div>
          <div>
            <div style={{ fontWeight: 600, fontSize: 14 }}>Smart Money Tracker</div>
            <div style={{ fontSize: 11, color: "#9ca3af" }}>AI market signal dashboard</div>
          </div>
        </div>

        {/* Ticker selector */}
        <div style={{ display: "flex", gap: 6 }}>
          {TICKERS.map(t => (
            <button key={t} onClick={() => setActiveTicker(t)}
              style={{
                fontSize: 12, padding: "5px 12px", borderRadius: 99, cursor: "pointer",
                background: activeTicker === t ? "#1a1a2e" : "transparent",
                color:      activeTicker === t ? "#e2e8f0" : "#374151",
                border:     activeTicker === t ? "1px solid #1a1a2e" : "1px solid #d1d5db",
              }}>
              {t}
            </button>
          ))}
        </div>

        {/* WebSocket status + Day Trading toggle */}
        <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
          <button onClick={() => setMode("daytrading")}
            style={{ fontSize: 11, padding: "5px 14px", borderRadius: 99, cursor: "pointer",
                     background: "#1a1a2e", color: "#fde68a",
                     border: "1px solid #f59e0b", fontWeight: 600 }}>
            ⚡ Day Trading Mode
          </button>
          <div style={{ display: "flex", alignItems: "center", gap: 6 }}>
            <div style={{
              width: 8, height: 8, borderRadius: "50%",
              background: wsStatus === "live" ? "#22c55e" : "#ef4444",
              animation: wsStatus === "live" ? "pulse 2s infinite" : "none",
            }} />
            <span style={{ fontSize: 11, color: "#6b7280" }}>{wsStatus}</span>
          </div>
        </div>
      </div>

      <div style={{ maxWidth: 1200, margin: "0 auto", padding: "16px 20px" }}>

        {/* ── Top row: price + prediction + backtest ── */}
        <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr 1fr",
                      gap: 12, marginBottom: 12 }}>

          {/* Price card */}
          <div style={{ background: "#fff", border: "1px solid #e5e7eb",
                        borderRadius: 12, padding: "14px 16px" }}>
            <div style={{ fontSize: 11, color: "#9ca3af", marginBottom: 4,
                          fontWeight: 500, textTransform: "uppercase", letterSpacing: "0.05em" }}>
              {activeTicker}
            </div>
            <div style={{ fontSize: 26, fontWeight: 600, color: "#111", marginBottom: 4 }}>
              {price.close ? `$${price.close.toLocaleString()}` : "—"}
            </div>
            <div style={{ fontSize: 13, color: price.change_pct >= 0 ? "#22c55e" : "#ef4444" }}>
              {price.change_pct >= 0 ? "▲" : "▼"} {Math.abs(price.change_pct ?? 0).toFixed(2)}% today
            </div>
            <div style={{ display: "flex", gap: 12, marginTop: 10 }}>
              {[["Open", price.open], ["High", price.high], ["Low", price.low]].map(([k, v]) => (
                <div key={k} style={{ fontSize: 10 }}>
                  <div style={{ color: "#9ca3af" }}>{k}</div>
                  <div style={{ color: "#374151", fontWeight: 500 }}>
                    {v ? `$${v.toLocaleString()}` : "—"}
                  </div>
                </div>
              ))}
            </div>
          </div>

          {/* AI Prediction card */}
          <div style={{ background: "#fff", border: `2px solid ${dirColor}33`,
                        borderRadius: 12, padding: "14px 16px" }}>
            <div style={{ fontSize: 10, color: "#6b7280", fontWeight: 500,
                          textTransform: "uppercase", letterSpacing: "0.05em", marginBottom: 6 }}>
              🧠 AI prediction · LSTM + TV + sentiment
            </div>
            {loading ? (
              <div style={{ color: "#9ca3af", fontSize: 13 }}>Computing...</div>
            ) : (
              <>
                <PredictionGauge
                  confidence={pred.confidence ?? 0.5}
                  direction={pred.direction ?? "neutral"}
                />
                <div style={{ fontSize: 11, color: "#9ca3af", textAlign: "center" }}>
                  24h horizon · updated live
                </div>
                {pred.all_agree && (
                  <div style={{ textAlign: "center", marginTop: 6 }}>
                    <Badge text="All signals agree ✓" color="#085041" bg="#E1F5EE" />
                  </div>
                )}
              </>
            )}
          </div>

          {/* Sentiment card */}
          <div style={{ background: "#fff", border: "1px solid #e5e7eb",
                        borderRadius: 12, padding: "14px 16px" }}>
            <div style={{ fontSize: 10, color: "#6b7280", fontWeight: 500,
                          textTransform: "uppercase", letterSpacing: "0.05em", marginBottom: 10 }}>
              📰 FinBERT sentiment
            </div>
            <div style={{ fontSize: 20, fontWeight: 600, color: "#111", marginBottom: 4 }}>
              {sent.score > 0 ? "+" : ""}{(sent.score ?? 0).toFixed(2)}
            </div>
            <Badge
              text={sent.label ?? "neutral"}
              color={sent.label === "bullish" ? "#0F6E56" : sent.label === "bearish" ? "#A32D2D" : "#633806"}
              bg={sent.label === "bullish" ? "#E1F5EE" : sent.label === "bearish" ? "#FCEBEB" : "#FAEEDA"}
            />
            <div style={{ fontSize: 11, color: "#9ca3af", marginTop: 8 }}>
              {sent.article_count ?? 0} articles · {sent.post_count ?? 0} Reddit posts
            </div>
          </div>
        </div>

        {/* ── Main chart (TradingView embed) + Alerts ── */}
        <div style={{ display: "grid", gridTemplateColumns: "1fr 260px",
                      gap: 12, marginBottom: 12 }}>

          {/* TradingView chart */}
          <div style={{ background: "#fff", border: "1px solid #e5e7eb",
                        borderRadius: 12, overflow: "hidden" }}>
            <div style={{ padding: "10px 14px", borderBottom: "1px solid #e5e7eb",
                          display: "flex", justifyContent: "space-between", alignItems: "center" }}>
              <div style={{ fontSize: 13, fontWeight: 500 }}>
                {activeTicker} — full chart with your indicators
              </div>
              <div style={{ fontSize: 11, color: "#9ca3af" }}>
                powered by TradingView · your account's indicators apply
              </div>
            </div>
            <TradingViewChart ticker={activeTicker} />
          </div>

          {/* Alerts panel */}
          <div style={{ background: "#fff", border: "1px solid #e5e7eb",
                        borderRadius: 12, padding: 14 }}>
            <div style={{ fontSize: 11, fontWeight: 500, color: "#6b7280",
                          textTransform: "uppercase", letterSpacing: "0.05em", marginBottom: 10 }}>
              🔔 Smart alerts
            </div>
            {alerts.length === 0 ? (
              <div style={{ color: "#9ca3af", fontSize: 12 }}>
                Alerts appear when TradingView fires a webhook or confidence exceeds 70%
              </div>
            ) : (
              alerts.map(a => (
                <div key={a.id} style={{ padding: "8px 0",
                                         borderBottom: "1px solid #f3f4f6",
                                         display: "flex", gap: 8 }}>
                  <div style={{
                    width: 8, height: 8, borderRadius: "50%", marginTop: 4, flexShrink: 0,
                    background: a.direction === "bullish" ? "#22c55e" : "#ef4444"
                  }} />
                  <div>
                    <div style={{ fontSize: 12, color: "#111", marginBottom: 2 }}>
                      {a.ticker} — {a.direction}
                    </div>
                    <div style={{ fontSize: 10, color: "#9ca3af" }}>
                      {Math.round(a.confidence * 100)}% confidence · {a.time}
                    </div>
                    {a.all_agree && (
                      <Badge text="All agree" color="#085041" bg="#E1F5EE" />
                    )}
                  </div>
                </div>
              ))
            )}
          </div>
        </div>

        {/* ── Indicator bars (from TradingView webhook data) ── */}
        <div style={{ background: "#fff", border: "1px solid #e5e7eb",
                      borderRadius: 12, padding: 16 }}>
          <div style={{ fontSize: 12, fontWeight: 500, color: "#111", marginBottom: 14 }}>
            Signal breakdown — from TradingView indicators
          </div>
          <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr 1fr 1fr", gap: 20 }}>
            <div>
              <div style={{ fontSize: 11, color: "#6b7280", marginBottom: 8, fontWeight: 500 }}>Trend</div>
              <SignalBar label="200 SMA dist" value={inds.sma200_dist ?? 0} max={20}
                color={inds.sma200_dist > 0 ? "#22c55e" : "#ef4444"} />
              <SignalBar label="Golden cross" value={inds.golden_cross ? 1 : 0} max={1}
                color="#22c55e" />
              <SignalBar label="VWAP dist" value={inds.vwap_dist ?? 0} max={5}
                color={inds.vwap_dist > 0 ? "#22c55e" : "#ef4444"} />
            </div>
            <div>
              <div style={{ fontSize: 11, color: "#6b7280", marginBottom: 8, fontWeight: 500 }}>Momentum</div>
              <SignalBar label="RSI" value={(inds.rsi_14 ?? 50) / 100} max={1}
                color={inds.rsi_14 > 70 ? "#ef4444" : inds.rsi_14 < 30 ? "#22c55e" : "#6366f1"} />
              <SignalBar label="MACD" value={Math.abs(inds.macd ?? 0)} max={1}
                color={inds.macd > 0 ? "#22c55e" : "#ef4444"} />
            </div>
            <div>
              <div style={{ fontSize: 11, color: "#6b7280", marginBottom: 8, fontWeight: 500 }}>Volatility</div>
              <SignalBar label="BB width %" value={inds.bb_width ?? 0} max={20}
                color="#6366f1" />
              <SignalBar label="ATR norm" value={inds.atr_norm ?? 0} max={0.05}
                color="#f59e0b" />
            </div>
            <div>
              <div style={{ fontSize: 11, color: "#6b7280", marginBottom: 8, fontWeight: 500 }}>Volume</div>
              <SignalBar label="Volume ratio" value={(inds.volume_ratio ?? 1) - 1} max={3}
                color={inds.volume_ratio > 1.5 ? "#22c55e" : "#9ca3af"} />
              <SignalBar label="OBV trend" value={Math.abs(inds.obv_norm ?? 0)} max={3}
                color="#6366f1" />
            </div>
          </div>
        </div>

      </div>
    </div>
  );
}
