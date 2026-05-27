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

// DayTradingDashboard.jsx
// Add this to your smart-money-frontend/src/ folder
// Then import and use it in App.jsx

import { useState, useEffect, useRef, useCallback } from "react";

const API_BASE = "http://localhost:8000";

// ── Time display ───────────────────────────────────────────────
function Clock() {
  const [time, setTime] = useState(new Date());
  useEffect(() => {
    const t = setInterval(() => setTime(new Date()), 1000);
    return () => clearInterval(t);
  }, []);

  const et = new Date(time.toLocaleString("en-US", { timeZone: "America/New_York" }));
  const h  = String(et.getHours()).padStart(2, "0");
  const m  = String(et.getMinutes()).padStart(2, "0");
  const s  = String(et.getSeconds()).padStart(2, "0");

  const isMarketOpen = et.getHours() >= 9 && (et.getHours() < 16);
  const isPremarket  = et.getHours() >= 4 && (et.getHours() < 9 || (et.getHours() === 9 && et.getMinutes() < 30));

  return (
    <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
      <div style={{
        width: 8, height: 8, borderRadius: "50%",
        background: isMarketOpen ? "#22c55e" : isPremarket ? "#f59e0b" : "#6b7280",
        animation: (isMarketOpen || isPremarket) ? "pulse 2s infinite" : "none",
      }} />
      <span style={{ fontFamily: "monospace", fontSize: 13, color: "#e2e8f0" }}>
        {h}:{m}:{s} ET
      </span>
      <span style={{
        fontSize: 10, padding: "1px 7px", borderRadius: 99,
        background: isMarketOpen ? "#14532d" : isPremarket ? "#78350f" : "#1f2937",
        color: isMarketOpen ? "#86efac" : isPremarket ? "#fde68a" : "#9ca3af",
      }}>
        {isMarketOpen ? "MARKET OPEN" : isPremarket ? "PRE-MARKET" : "CLOSED"}
      </span>
    </div>
  );
}

// ── Signal badge ───────────────────────────────────────────────
function SignalBadge({ signal, confidence }) {
  const configs = {
    CALLS: { bg: "#14532d", color: "#86efac", border: "#22c55e", icon: "▲" },
    PUTS:  { bg: "#7f1d1d", color: "#fca5a5", border: "#ef4444", icon: "▼" },
    WAIT:  { bg: "#78350f", color: "#fde68a", border: "#f59e0b", icon: "⏸" },
  };
  const c = configs[signal] || configs.WAIT;
  return (
    <div style={{
      background: c.bg, border: `2px solid ${c.border}`,
      borderRadius: 12, padding: "16px 20px", textAlign: "center",
    }}>
      <div style={{ fontSize: 32, marginBottom: 4 }}>{c.icon}</div>
      <div style={{ fontSize: 22, fontWeight: 700, color: c.color, letterSpacing: 2 }}>
        {signal}
      </div>
      {confidence > 0 && (
        <div style={{ fontSize: 13, color: c.color, opacity: 0.8, marginTop: 4 }}>
          {Math.round(confidence * 100)}% confidence
        </div>
      )}
    </div>
  );
}

// ── Mini candlestick chart ─────────────────────────────────────
function MiniChart({ candles = [], pmHigh, pmLow, vwap, prevHigh, prevLow, resistance = [], support = [], height = 140 }) {
  if (!candles.length) return (
    <div style={{ height, display: "flex", alignItems: "center", justifyContent: "center",
                  color: "#4b5563", fontSize: 12 }}>
      No intraday data yet
    </div>
  );

  const w = 600, h = height;
  const pad = { l: 8, r: 40, t: 8, b: 20 };
  const chartW = w - pad.l - pad.r;
  const chartH = h - pad.t - pad.b;

  const allPrices = candles.flatMap(c => [c.high, c.low]);
  const minP = Math.min(...allPrices) * 0.9995;
  const maxP = Math.max(...allPrices) * 1.0005;
  const range = maxP - minP || 1;

  const scaleY = (p) => pad.t + chartH - ((p - minP) / range) * chartH;
  const candleW = Math.max(2, Math.floor(chartW / candles.length) - 1);
  const step = chartW / candles.length;

  return (
    <svg viewBox={`0 0 ${w} ${h}`} width="100%" preserveAspectRatio="none"
         style={{ borderRadius: 6 }}>
      <rect width={w} height={h} fill="#0f1117" rx="6" />

      {/* Pre-market levels */}
      {pmHigh && (
        <>
          <line x1={pad.l} y1={scaleY(pmHigh)} x2={w - pad.r} y2={scaleY(pmHigh)}
                stroke="#f59e0b" strokeWidth="1" strokeDasharray="4 3" opacity="0.8" />
          <text x={w - pad.r + 2} y={scaleY(pmHigh) + 4} fontSize="8" fill="#f59e0b">PM H</text>
        </>
      )}
      {pmLow && (
        <>
          <line x1={pad.l} y1={scaleY(pmLow)} x2={w - pad.r} y2={scaleY(pmLow)}
                stroke="#f59e0b" strokeWidth="1" strokeDasharray="4 3" opacity="0.8" />
          <text x={w - pad.r + 2} y={scaleY(pmLow) + 4} fontSize="8" fill="#f59e0b">PM L</text>
        </>
      )}

      {/* VWAP line */}
      {vwap && (
        <>
          <line x1={pad.l} y1={scaleY(vwap)} x2={w - pad.r} y2={scaleY(vwap)}
                stroke="#818cf8" strokeWidth="1.5" strokeDasharray="6 2" opacity="0.9" />
          <text x={w - pad.r + 2} y={scaleY(vwap) + 4} fontSize="8" fill="#818cf8">VWAP</text>
        </>
      )}

      {/* Previous day high/low */}
      {prevHigh && scaleY(prevHigh) > pad.t && scaleY(prevHigh) < h && (
        <>
          <line x1={pad.l} y1={scaleY(prevHigh)} x2={w - pad.r} y2={scaleY(prevHigh)}
                stroke="#f97316" strokeWidth="1" strokeDasharray="3 2" opacity="0.8" />
          <text x={w - pad.r + 2} y={scaleY(prevHigh) + 4} fontSize="8" fill="#f97316">PD H</text>
        </>
      )}
      {prevLow && scaleY(prevLow) > pad.t && scaleY(prevLow) < h && (
        <>
          <line x1={pad.l} y1={scaleY(prevLow)} x2={w - pad.r} y2={scaleY(prevLow)}
                stroke="#f97316" strokeWidth="1" strokeDasharray="3 2" opacity="0.8" />
          <text x={w - pad.r + 2} y={scaleY(prevLow) + 4} fontSize="8" fill="#f97316">PD L</text>
        </>
      )}

      {/* Key resistance levels */}
      {resistance.map((level, i) => scaleY(level) > pad.t && scaleY(level) < h && (
        <g key={"r"+i}>
          <line x1={pad.l} y1={scaleY(level)} x2={w - pad.r} y2={scaleY(level)}
                stroke="#ef4444" strokeWidth="0.75" strokeDasharray="2 3" opacity="0.7" />
          <text x={w - pad.r + 2} y={scaleY(level) + 4} fontSize="7" fill="#ef4444">R{i+1}</text>
        </g>
      ))}

      {/* Key support levels */}
      {support.map((level, i) => scaleY(level) > pad.t && scaleY(level) < h && (
        <g key={"s"+i}>
          <line x1={pad.l} y1={scaleY(level)} x2={w - pad.r} y2={scaleY(level)}
                stroke="#22c55e" strokeWidth="0.75" strokeDasharray="2 3" opacity="0.7" />
          <text x={w - pad.r + 2} y={scaleY(level) + 4} fontSize="7" fill="#22c55e">S{i+1}</text>
        </g>
      ))}

      {/* Candles */}
      {candles.map((c, i) => {
        const x     = pad.l + i * step + step / 2;
        const isUp  = c.close >= c.open;
        const color = isUp ? "#22c55e" : "#ef4444";
        const bodyT = scaleY(Math.max(c.open, c.close));
        const bodyB = scaleY(Math.min(c.open, c.close));
        const bodyH = Math.max(1, bodyB - bodyT);

        return (
          <g key={i}>
            <line x1={x} y1={scaleY(c.high)} x2={x} y2={scaleY(c.low)}
                  stroke={color} strokeWidth="1" />
            <rect x={x - candleW / 2} y={bodyT} width={candleW} height={bodyH}
                  fill={color} opacity="0.9" />
          </g>
        );
      })}
    </svg>
  );
}

// ── Level card ─────────────────────────────────────────────────
function LevelCard({ label, value, color, note }) {
  return (
    <div style={{ background: "#1a1a2e", border: `0.5px solid ${color}33`,
                  borderRadius: 8, padding: "8px 12px", flex: 1 }}>
      <div style={{ fontSize: 10, color: color, fontWeight: 600,
                    letterSpacing: "0.05em", textTransform: "uppercase", marginBottom: 3 }}>
        {label}
      </div>
      <div style={{ fontSize: 18, fontWeight: 600, color: "#e2e8f0", marginBottom: 2 }}>
        {value ? `$${value}` : "—"}
      </div>
      {note && <div style={{ fontSize: 10, color: "#6b7280" }}>{note}</div>}
    </div>
  );
}

// ── Reason list ────────────────────────────────────────────────
function ReasonList({ reasons = [] }) {
  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 4 }}>
      {reasons.map((r, i) => (
        <div key={i} style={{ fontSize: 11, color: "#9ca3af", lineHeight: 1.5 }}>
          {r}
        </div>
      ))}
    </div>
  );
}

// ── Phase indicator ────────────────────────────────────────────
function PhaseIndicator({ phase }) {
  if (!phase) return null;
  return (
    <div style={{
      background: "#1a1a2e", border: `.5px solid ${phase.color}`,
      borderRadius: 10, padding: "12px 16px",
    }}>
      <div style={{ display: "flex", alignItems: "center", gap: 10, marginBottom: 8 }}>
        <div style={{
          background: phase.color + "22", border: `1px solid ${phase.color}`,
          borderRadius: 99, padding: "2px 12px", fontSize: 11, fontWeight: 600,
          color: phase.color, letterSpacing: "0.05em",
        }}>
          {phase.primary} CHART
          {phase.confluence && ` + ${phase.confluence} confluence`}
        </div>
        <div style={{ fontSize: 11, color: "#6b7280", textTransform: "capitalize" }}>
          {phase.phase?.replace("-", " ")}
        </div>
      </div>
      <div style={{ fontSize: 12, color: "#d1d5db", lineHeight: 1.6 }}>
        {phase.action}
      </div>
    </div>
  );
}

// ── Main DayTrading Dashboard ──────────────────────────────────
export default function DayTradingDashboard({ ticker = "AAPL" }) {
  const [data,    setData]    = useState(null);
  const [loading, setLoading] = useState(true);
  const [chart,   setChart]   = useState("5m");
  const [error,   setError]   = useState(null);
  const intervalRef = useRef(null);

  const fetchData = useCallback(async () => {
    setLoading(true);
    setData(null);
    try {
      const res = await fetch(`${API_BASE}/daytrading/${ticker}`, {
        headers: { "X-API-Key": "dev" },
      });
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const json = await res.json();
      setData(json);
      setError(null);
    } catch (e) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  }, [ticker]);

  useEffect(() => {
    fetchData();
    clearInterval(intervalRef.current);
    intervalRef.current = setInterval(fetchData, 60_000);
    return () => clearInterval(intervalRef.current);
  }, [fetchData]);

  const s = {
    container: {
      background: "#0d1117", minHeight: "100vh",
      fontFamily: "'DM Mono', 'Fira Code', monospace",
      color: "#e2e8f0", padding: "0 0 40px",
    },
    nav: {
      background: "#0d1117", borderBottom: "1px solid #1f2937",
      padding: "10px 20px", display: "flex",
      alignItems: "center", justifyContent: "space-between",
    },
    logo: { fontSize: 14, fontWeight: 700, color: "#e2e8f0", letterSpacing: 1 },
    body: { maxWidth: 1100, margin: "0 auto", padding: "16px 20px" },
    card: {
      background: "#161b22", border: "0.5px solid #21262d",
      borderRadius: 12, padding: "14px 16px", marginBottom: 12,
    },
    label: {
      fontSize: 10, fontWeight: 600, color: "#6b7280",
      letterSpacing: "0.08em", textTransform: "uppercase", marginBottom: 8,
    },
    grid3: { display: "grid", gridTemplateColumns: "1fr 1fr 1fr", gap: 10, marginBottom: 12 },
    grid2: { display: "grid", gridTemplateColumns: "1fr 1fr", gap: 10, marginBottom: 12 },
    tabRow: { display: "flex", gap: 4, marginBottom: 10 },
    tab: (active) => ({
      fontSize: 11, padding: "4px 12px", borderRadius: 6,
      cursor: "pointer", border: "0.5px solid",
      background:   active ? "#1d4ed8" : "#161b22",
      color:        active ? "#bfdbfe" : "#6b7280",
      borderColor:  active ? "#1d4ed8" : "#21262d",
    }),
  };

  if (loading) return (
    <div style={{ ...s.container, display: "flex", alignItems: "center",
                  justifyContent: "center", height: "100vh" }}>
      <div style={{ textAlign: "center" }}>
        <div style={{ fontSize: 32, marginBottom: 12 }}>📊</div>
        <div style={{ color: "#6b7280" }}>Loading day trading signals...</div>
      </div>
    </div>
  );

  if (error) return (
    <div style={{ ...s.container, display: "flex", alignItems: "center",
                  justifyContent: "center", height: "100vh" }}>
      <div style={{ textAlign: "center" }}>
        <div style={{ color: "#ef4444", marginBottom: 8 }}>Failed to load: {error}</div>
        <button onClick={fetchData} style={{ fontSize: 12, padding: "6px 14px",
          borderRadius: 6, cursor: "pointer" }}>Retry</button>
      </div>
    </div>
  );

  const pm      = data?.premarket   || {};
  const struct  = data?.structure   || {};
  const vwapD   = data?.vwap        || {};
  const candle  = data?.candle      || {};
  const options = data?.options     || {};
  const phase   = data?.market_phase || {};
  const charts  = data?.charts      || {};
  const price   = data?.current_price || 0;

  const chartData = charts[chart] || [];

  return (
    <div style={s.container}>

      {/* Nav */}
      <div style={s.nav}>
        <div style={s.logo}>⚡ DAY TRADER — {ticker}</div>
        <Clock />
        <div style={{ fontSize: 11, color: "#6b7280" }}>
          Auto-refreshes every 60s
        </div>
      </div>

      <div style={s.body}>

        {/* Timeframe phase */}
        <div style={{ marginBottom: 12 }}>
          <PhaseIndicator phase={phase} />
        </div>

        {/* Top row: Signal + Pre-market levels + Structure */}
        <div style={s.grid3}>

          {/* Options signal */}
          <div style={s.card}>
            <div style={s.label}>Options signal</div>
            <SignalBadge signal={options.signal || "WAIT"}
                         confidence={options.confidence || 0} />
            <div style={{ fontSize: 11, color: "#9ca3af", marginTop: 10,
                          lineHeight: 1.6, padding: "8px", background: "#0d1117",
                          borderRadius: 6 }}>
              {options.action}
            </div>
          </div>

          {/* Pre-market levels */}
          <div style={s.card}>
            <div style={s.label}>Pre-market levels</div>
            <div style={{ display: "flex", gap: 8, marginBottom: 10 }}>
              <LevelCard label="PM High" value={pm.high} color="#ef4444"
                         note="Resistance — puts target" />
              <LevelCard label="PM Low" value={pm.low} color="#22c55e"
                         note="Support — calls target" />
            </div>
            <div style={{ fontSize: 10, color: "#6b7280" }}>
              {pm.note || "Pre-market levels auto-detected"}
            </div>
            <div style={{ fontSize: 12, color: "#e2e8f0", marginTop: 6 }}>
              Current: <strong>${price.toFixed(2)}</strong>
              {pm.high && pm.low && (
                <span style={{ color: "#6b7280", fontSize: 10 }}>
                  {" "}({((price - pm.low) / (pm.high - pm.low) * 100).toFixed(0)}% of PM range)
                </span>
              )}
            </div>
          </div>

          {/* Structure + VWAP */}
          <div style={s.card}>
            <div style={s.label}>Market structure</div>
            <div style={{
              fontSize: 16, fontWeight: 600, marginBottom: 6,
              color: struct.bias === "bullish" ? "#22c55e"
                   : struct.bias === "bearish" ? "#ef4444" : "#f59e0b",
            }}>
              {struct.emoji} {struct.structure || "Detecting..."}
            </div>
            <div style={{ fontSize: 11, color: "#9ca3af", marginBottom: 10 }}>
              {struct.action}
            </div>
            <div style={{ borderTop: "0.5px solid #21262d", paddingTop: 8 }}>
              <div style={s.label}>VWAP</div>
              <div style={{ fontSize: 12, color: vwapD.bias === "bullish" ? "#22c55e" : "#ef4444" }}>
                ${vwapD.vwap} — {vwapD.signal}
              </div>
            </div>
          </div>

        </div>

        {/* Chart */}
        <div style={s.card}>
          <div style={{ display: "flex", justifyContent: "space-between",
                        alignItems: "center", marginBottom: 10 }}>
            <div style={s.label}>Intraday chart</div>
            <div style={s.tabRow}>
              {["2m", "5m", "10m", "1h"].map(tf => (
                <button key={tf} style={s.tab(chart === tf)}
                        onClick={() => setChart(tf)}>
                  {tf}
                </button>
              ))}
            </div>
          </div>
          <MiniChart
            candles={chartData}
            pmHigh={pm.high}
            pmLow={pm.low}
            vwap={vwapD.vwap}
            prevHigh={data?.prev_day?.high}
            prevLow={data?.prev_day?.low}
            resistance={data?.key_levels?.resistance || []}
            support={data?.key_levels?.support || []}
            height={180}
          />
          <div style={{ display: "flex", gap: 12, marginTop: 8, flexWrap: "wrap" }}>
            <span style={{ fontSize: 10, color: "#f59e0b" }}>── PM High/Low</span>
            <span style={{ fontSize: 10, color: "#818cf8" }}>── VWAP</span>
            <span style={{ fontSize: 10, color: "#f97316" }}>── Prev Day H/L</span>
            <span style={{ fontSize: 10, color: "#ef4444" }}>── R levels</span>
            <span style={{ fontSize: 10, color: "#22c55e" }}>── S levels</span>
            <span style={{ fontSize: 10, color: "#22c55e" }}>▮ Bullish</span>
            <span style={{ fontSize: 10, color: "#ef4444" }}>▮ Bearish</span>
          </div>
        </div>

        {/* Key levels panel */}
        <div style={s.card}>
          <div style={s.label}>Key levels — auto-detected</div>
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr 1fr', gap: 8 }}>
            <div>
              <div style={{ fontSize: 10, color: '#f97316', fontWeight: 600, marginBottom: 4 }}>PREV DAY</div>
              <div style={{ fontSize: 11, color: '#e2e8f0' }}>H: ${data?.prev_day?.high || '—'}</div>
              <div style={{ fontSize: 11, color: '#e2e8f0' }}>L: ${data?.prev_day?.low  || '—'}</div>
            </div>
            <div>
              <div style={{ fontSize: 10, color: '#ef4444', fontWeight: 600, marginBottom: 4 }}>RESISTANCE</div>
              {(data?.key_levels?.resistance || []).slice(0,3).map((l,i) => (
                <div key={i} style={{ fontSize: 11, color: '#fca5a5' }}>${l}</div>
              ))}
              {!(data?.key_levels?.resistance?.length) && <div style={{ fontSize: 11, color: '#6b7280' }}>None nearby</div>}
            </div>
            <div>
              <div style={{ fontSize: 10, color: '#22c55e', fontWeight: 600, marginBottom: 4 }}>SUPPORT</div>
              {(data?.key_levels?.support || []).slice(0,3).map((l,i) => (
                <div key={i} style={{ fontSize: 11, color: '#86efac' }}>${l}</div>
              ))}
              {!(data?.key_levels?.support?.length) && <div style={{ fontSize: 11, color: '#6b7280' }}>None nearby</div>}
            </div>
          </div>
          {(data?.key_levels?.trendlines || []).length > 0 && (
            <div style={{ marginTop: 8, borderTop: '0.5px solid #21262d', paddingTop: 8 }}>
              <div style={{ fontSize: 10, color: '#6b7280', fontWeight: 600, marginBottom: 4 }}>AUTO TRENDLINES</div>
              <div style={{ display: 'flex', gap: 12 }}>
                {(data?.key_levels?.trendlines || []).map((t,i) => (
                  <div key={i} style={{ fontSize: 11, color: t.color }}>
                    {t.label}: ${t.p1} → ${t.p2}
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>

        {/* Bottom row: Candle pattern + Signal reasons */}
        <div style={s.grid2}>

          {/* Candle pattern */}
          <div style={s.card}>
            <div style={s.label}>Latest candle pattern</div>
            {candle.detected ? (
              <>
                <div style={{
                  fontSize: 14, fontWeight: 600, marginBottom: 6,
                  color: candle.type === "hammer" ? "#22c55e"
                       : candle.type === "inverted_hammer" ? "#ef4444" : "#f59e0b",
                }}>
                  {candle.type === "hammer"          ? "🔨 Hammer" :
                   candle.type === "inverted_hammer" ? "🔨 Inverted Hammer" :
                   candle.type === "doji"            ? "➕ Doji" : candle.type}
                  {" "}
                  <span style={{ fontSize: 10, color: "#6b7280" }}>({candle.strength})</span>
                </div>
                <div style={{ fontSize: 11, color: "#9ca3af", lineHeight: 1.6 }}>
                  {candle.signal}
                </div>
              </>
            ) : (
              <div style={{ fontSize: 12, color: "#6b7280" }}>
                No reversal candle on current bar. Normal candle — follow the trend.
              </div>
            )}
          </div>

          {/* Signal reasons */}
          <div style={s.card}>
            <div style={s.label}>Why this signal?</div>
            <ReasonList reasons={options.reasons || []} />
            {(!options.reasons || !options.reasons.length) && (
              <div style={{ fontSize: 12, color: "#6b7280" }}>
                Gathering signals...
              </div>
            )}
          </div>

        </div>

        {/* Trading rules reminder */}
        <div style={{
          background: "#0d1117", border: "0.5px solid #21262d",
          borderRadius: 10, padding: "12px 16px",
        }}>
          <div style={s.label}>Your rules (from your system)</div>
          <div style={{
            display: "grid", gridTemplateColumns: "1fr 1fr 1fr", gap: 8
          }}>
            {[
              { time: "9:30–10am", chart: "2m + 5m", rule: "Wait for HH/HL structure. Enter at pullback." },
              { time: "10–11am",   chart: "5m",      rule: "Trend continuation. No forced entries." },
              { time: "11am+",     chart: "10m + 5m", rule: "Bigger moves only. Wait for trend line breaks." },
            ].map((r, i) => (
              <div key={i} style={{
                background: "#161b22", borderRadius: 8, padding: "8px 10px",
                border: "0.5px solid #21262d",
              }}>
                <div style={{ fontSize: 10, color: "#f59e0b", fontWeight: 600,
                              marginBottom: 3 }}>{r.time} — {r.chart}</div>
                <div style={{ fontSize: 11, color: "#9ca3af", lineHeight: 1.5 }}>{r.rule}</div>
              </div>
            ))}
          </div>
        </div>

      </div>
    </div>
  );
}
