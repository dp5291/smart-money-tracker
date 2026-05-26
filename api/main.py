"""
api/main.py — FastAPI backend with full security integrated.
"""
import os, sys, json, asyncio
from datetime import datetime
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, Depends, Request
from fastapi.responses import JSONResponse

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from config import TICKERS, FEATURE_COLUMNS, LOOKBACK_DAYS, MODEL_DIR
from data.fetcher import fetch_ohlcv
from data.indicators import compute_all_indicators, get_feature_matrix
from data.sentiment import get_combined_sentiment, TICKER_NAMES
from models.lstm import LSTMPredictor, predict_latest
from models.backtest import run_backtest
from api.security import setup_security, verify_api_key, validate_ticker, limiter, RATE_LIMITS, logger
import torch, joblib, numpy as np

app = FastAPI(title="Smart Money Tracker API", version="1.0.0",
    docs_url=None if os.getenv("HIDE_DOCS")=="true" else "/docs",
    redoc_url=None if os.getenv("HIDE_DOCS")=="true" else "/redoc")
setup_security(app)

from api.webhook import router as webhook_router
from api.daytrading import router as daytrading_router
app.include_router(webhook_router)
app.include_router(daytrading_router)

_models, _scalers = {}, {}
def load_model(ticker):
    if ticker not in _models:
        mp = os.path.join(MODEL_DIR, f"{ticker}.pt")
        sp = os.path.join(MODEL_DIR, f"{ticker}_scaler.pkl")
        if not os.path.exists(mp): return None, None
        m = LSTMPredictor(input_size=len(FEATURE_COLUMNS))
        m.load_state_dict(torch.load(mp, map_location="cpu")); m.eval()
        _models[ticker] = m; _scalers[ticker] = joblib.load(sp)
    return _models.get(ticker), _scalers.get(ticker)

async def compute_signal(ticker):
    try:
        df = fetch_ohlcv(ticker, period="2y"); df = compute_all_indicators(df)
        feat_df = get_feature_matrix(df, FEATURE_COLUMNS)
        model, scaler = load_model(ticker)
        prediction = predict_latest(model, scaler.transform(feat_df.values), LOOKBACK_DAYS) if model else {"probability":0.5,"direction":"neutral","confidence":0.5}
        latest = df.iloc[-1]; prev = df.iloc[-2] if len(df)>1 else latest
        sentiment = get_combined_sentiment(ticker, TICKER_NAMES.get(ticker,ticker), hours_back=6)
        return {"ticker":ticker,"timestamp":datetime.utcnow().isoformat(),
            "price":{"close":round(float(latest["close"]),2),"open":round(float(latest["open"]),2),
                "high":round(float(latest["high"]),2),"low":round(float(latest["low"]),2),
                "volume":int(latest["volume"]),"change_pct":round(float((latest["close"]-prev["close"])/prev["close"]*100),2)},
            "prediction":prediction,
            "indicators":{"sma_200":round(float(latest.get("sma_200",0) or 0),2),"sma_50":round(float(latest.get("sma_50",0) or 0),2),
                "golden_cross":bool(latest.get("golden_cross",0)),"rsi_14":round(float(latest.get("rsi_14",0.5) or 0.5)*100,1),
                "macd":round(float(latest.get("macd",0) or 0),4),"bb_width":round(float(latest.get("bb_width",0) or 0),2),
                "vwap":round(float(latest.get("vwap",0) or 0),2),"sma_200_dist":round(float(latest.get("sma_200_dist",0) or 0),2)},
            "sentiment":{"score":sentiment["combined_score"],"label":sentiment["label"],
                "article_count":sentiment["article_count"],"post_count":sentiment["post_count"]},
            "chart":[{"date":str(r.Index.date()),"open":round(float(r.open),2),"high":round(float(r.high),2),
                "low":round(float(r.low),2),"close":round(float(r.close),2),"volume":int(r.volume),
                "sma_200":round(float(r.sma_200 or 0),2),"sma_50":round(float(r.sma_50 or 0),2),
                "bb_upper":round(float(r.bb_upper or 0),2),"bb_lower":round(float(r.bb_lower or 0),2),
                "vwap":round(float(r.vwap or 0),2)} for r in df.iloc[-90:].itertuples()]}
    except Exception as e:
        logger.error(f"compute_signal error {ticker}: {e}"); return {"error":str(e),"ticker":ticker}

@app.get("/health")
async def health(): return {"status":"ok","timestamp":datetime.utcnow().isoformat()}

@app.get("/signal/{ticker}")
@limiter.limit(RATE_LIMITS["signal"])
async def get_signal(request:Request, ticker:str=Depends(validate_ticker), _key:str=Depends(verify_api_key)):
    logger.info(f"Signal: {ticker}"); return await compute_signal(ticker)

@app.get("/backtest/{ticker}")
@limiter.limit(RATE_LIMITS["backtest"])
async def get_backtest(request:Request, ticker:str=Depends(validate_ticker), _key:str=Depends(verify_api_key)):
    try: return run_backtest(ticker)
    except FileNotFoundError: raise HTTPException(404,f"No model for {ticker}. Run: python run.py --train {ticker}")
    except Exception as e: logger.error(f"Backtest {ticker}: {e}"); raise HTTPException(500,"Backtest failed")

@app.get("/historical/{ticker}")
@limiter.limit(RATE_LIMITS["general"])
async def get_historical(request:Request, ticker:str=Depends(validate_ticker), period:str="1y", _key:str=Depends(verify_api_key)):
    if period not in {"1d","5d","1mo","3mo","6mo","1y","2y","5y"}: raise HTTPException(400,"Invalid period")
    df = compute_all_indicators(fetch_ohlcv(ticker, period=period))
    return {"ticker":ticker,"data":[{"date":str(r.Index.date()),"open":round(float(r.open),2),"high":round(float(r.high),2),
        "low":round(float(r.low),2),"close":round(float(r.close),2),"volume":int(r.volume),
        "sma_200":round(float(r.sma_200 or 0),2),"sma_50":round(float(r.sma_50 or 0),2),
        "rsi":round(float(r.rsi_14 or 0.5)*100,1),"bb_upper":round(float(r.bb_upper or 0),2),
        "bb_lower":round(float(r.bb_lower or 0),2),"macd":round(float(r.macd or 0),4),
        "vwap":round(float(r.vwap or 0),2)} for r in df.itertuples()]}

class ConnectionManager:
    def __init__(self): self.connections = {}
    async def connect(self, ticker, ws):
        await ws.accept(); self.connections.setdefault(ticker,[]).append(ws)
    def disconnect(self, ticker, ws):
        if ticker in self.connections:
            try: self.connections[ticker].remove(ws)
            except ValueError: pass
    async def broadcast(self, ticker, data):
        dead=[]
        for ws in self.connections.get(ticker,[]):
            try: await ws.send_json(data)
            except: dead.append(ws)
        for ws in dead: self.disconnect(ticker,ws)

manager = ConnectionManager()

@app.websocket("/ws/{ticker}")
async def websocket_endpoint(websocket:WebSocket, ticker:str):
    ticker = ticker.upper()
    if ticker not in TICKERS: await websocket.close(code=4004,reason=f"{ticker} not supported"); return
    token = websocket.query_params.get("token","")
    valid_keys = set(os.getenv("API_KEYS","").split(","))
    if valid_keys and valid_keys!={""} and os.getenv("DEV_MODE")!="true":
        if token not in valid_keys: await websocket.close(code=4001,reason="Invalid API key"); return
    await manager.connect(ticker, websocket)
    try:
        await websocket.send_json(await compute_signal(ticker))
        while True:
            await asyncio.sleep(300)
            await websocket.send_json(await compute_signal(ticker))
    except WebSocketDisconnect: manager.disconnect(ticker, websocket)
