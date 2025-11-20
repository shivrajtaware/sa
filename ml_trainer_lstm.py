import os
import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

from data_fetcher import get_stock_data
from indicators import apply_all

MODEL_DIR = "models"
MODEL_FILE = os.path.join(MODEL_DIR, "lstm_next_dir.h5")
SCALER_FILE = os.path.join(MODEL_DIR, "lstm_scaler.pkl")

SEQ = 30

FEATURES = [
    'Open','High','Low','Close','Volume',
    'EMA20','EMA50','RSI','MACD','Signal',
    'BB_UPPER','BB_LOWER','AvgVol'
]

NIFTY50 = [
    "RELIANCE.NS","TCS.NS","HDFCBANK.NS","INFY.NS","ICICIBANK.NS","SBIN.NS",
    "HINDUNILVR.NS","HCLTECH.NS","KOTAKBANK.NS","ITC.NS","LT.NS","ASIANPAINT.NS",
    "ULTRACEMCO.NS","TITAN.NS","BAJFINANCE.NS","BAJAJFINSV.NS","ADANIENT.NS",
    "ADANIPORTS.NS","NESTLEIND.NS","SUNPHARMA.NS","WIPRO.NS","AXISBANK.NS",
    "POWERGRID.NS","ONGC.NS","COALINDIA.NS","TATAMOTORS.NS","TATASTEEL.NS",
    "BPCL.NS","BHARTIARTL.NS","TECHM.NS","JSWSTEEL.NS","GRASIM.NS","HEROMOTOCO.NS",
    "DRREDDY.NS","BRITANNIA.NS","HDFCLIFE.NS","DIVISLAB.NS","CIPLA.NS",
    "EICHERMOT.NS","MARUTI.NS","M&M.NS","BAJAJ-AUTO.NS","UPL.NS","SBILIFE.NS",
    "HINDALCO.NS","APOLLOHOSP.NS","INDUSINDBK.NS","TATACONSUM.NS","ICICIPRULI.NS"
]

def build_dataset():
    X_all = []
    Y_all = []

    for sym in NIFTY50:
        print(f"Fetching {sym} …")
        df = get_stock_data(sym, interval="1d", limit=3000)
        if df.empty:
            print("Skipping:", sym)
            continue

        df = apply_all(df)

        for c in FEATURES:
            if c not in df.columns:
                df[c] = 0

        df = df[FEATURES].fillna(0)

        closes = df['Close'].values
        arr = df.values

        y = [1 if closes[i] > closes[i-1] else 0 for i in range(1, len(closes))]

        arr = arr[1:]
        y = np.array(y)

        if len(arr) < SEQ:
            continue

        scaler = StandardScaler()
        arr_scaled = scaler.fit_transform(arr)

        for i in range(SEQ, len(arr_scaled)):
            X_all.append(arr_scaled[i-SEQ:i])
            Y_all.append(y[i])

    X_all = np.array(X_all)
    Y_all = np.array(Y_all)

    print("Final dataset:", X_all.shape, Y_all.shape)
    return X_all, Y_all


def build_model(input_shape):
    model = Sequential()
    model.add(LSTM(96, return_sequences=True, input_shape=input_shape))
    model.add(Dropout(0.25))
    model.add(LSTM(48))
    model.add(Dropout(0.25))
    model.add(Dense(1, activation="sigmoid"))
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return model


def train_model():
    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)

    print("Building multi-stock dataset…")
    X, Y = build_dataset()

    print("Training model…")
    model = build_model((SEQ, X.shape[2]))

    es = EarlyStopping(monitor='loss', patience=8, restore_best_weights=True)

    model.fit(
        X, Y,
        epochs=30,
        batch_size=64,
        callbacks=[es],
        verbose=1
    )

    scaler = StandardScaler().fit(X.reshape(-1, X.shape[2]))

    print("Saving…")
    model.save(MODEL_FILE)
    joblib.dump(scaler, SCALER_FILE)

    print("✓ LSTM trained on Nifty 50!")
    print("✓ Model saved:", MODEL_FILE)


if __name__ == "__main__":
    train_model()
