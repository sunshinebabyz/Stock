import os
import joblib
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
import glob
from datetime import timedelta
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error

from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, GRU, Dense, Dropout, Bidirectional
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

import yfinance as yf



#LSTM

def train_lstm_model_from_csv(file_path, window_size=60, epochs=20, batch_size=32):
    PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
    selected_symbol = os.path.splitext(os.path.basename(file_path))[0]
    model_dir = os.path.join(PROJECT_ROOT, "models")
    os.makedirs(model_dir, exist_ok=True)

    model_save_path = os.path.join(
        model_dir, f"{selected_symbol}_lstm_ws{window_size}_ep{epochs}.h5"
    )
    scaler_save_path = os.path.join(
        model_dir, f"{selected_symbol}_lstm_scaler_ws{window_size}_ep{epochs}.pkl"
    )

    # Load data
    df = pd.read_csv(file_path)
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date')
    data = df['Close'].values.reshape(-1, 1)

    # Scale
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data)

    # 👇 Split BEFORE creating windows
    split_index = int(len(scaled_data) * 0.8)
    train_data = scaled_data[:split_index]
    test_data = scaled_data[split_index - window_size:]  # 👈 cần giữ lại `window_size` trước test để tạo chuỗi đầy đủ

    def create_dataset(series, window):
        X, y = [], []
        for i in range(len(series) - window):
            X.append(series[i:i + window])
            y.append(series[i + window])
        return np.array(X), np.array(y)

    X_train, y_train = create_dataset(train_data, window_size)
    X_test, y_test = create_dataset(test_data, window_size)

    X_train = X_train.reshape(-1, window_size, 1)
    X_test = X_test.reshape(-1, window_size, 1)

    # Build model
    model = Sequential([
        Bidirectional(LSTM(128, return_sequences=True), input_shape=(window_size, 1)),
        Dropout(0.2),
        Bidirectional(LSTM(64)),
        Dropout(0.2),
        Dense(1)
    ])

    model.compile(optimizer='adam', loss='mean_squared_error')
    early_stop = EarlyStopping(monitor='loss', patience=5, restore_best_weights=True)

    print(f"🔧 Đang huấn luyện mô hình Bidirectional LSTM cho {selected_symbol}...")
    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=1, callbacks=[early_stop])

    model.save(model_save_path)
    joblib.dump(scaler, scaler_save_path)
    model_type = "lstm"

    # Dự báo và tính lỗi
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    y_train_inv = scaler.inverse_transform(y_train.reshape(-1, 1))
    y_test_inv = scaler.inverse_transform(y_test.reshape(-1, 1))
    y_train_pred_inv = scaler.inverse_transform(y_train_pred)
    y_test_pred_inv = scaler.inverse_transform(y_test_pred)

    mae = mean_absolute_error(y_test_inv, y_test_pred_inv)
    rmse = np.sqrt(mean_squared_error(y_test_inv, y_test_pred_inv))
    mape = np.mean(np.abs((y_test_inv - y_test_pred_inv) / y_test_inv)) * 100

    error_dir = os.path.join(PROJECT_ROOT, "errors")
    os.makedirs(error_dir, exist_ok=True)
    error_path = os.path.join(
        error_dir, f"{selected_symbol}_{model_type}_ws{window_size}_ep{epochs}_bs{batch_size}.csv"
    )
    pd.DataFrame([{
        "Symbol": selected_symbol,
        "Model": "LSTM",
        "Window Size": window_size,
        "Epochs": epochs,
        "Batch Size": batch_size,
        "MAE": mae,
        "RMSE": rmse,
        "MAPE": mape
    }]).to_csv(error_path, index=False)

    print(f"📁 Đã lưu lỗi dự báo tại: {error_path}")
    print(f"✅ Đã lưu model tại: {model_save_path}")
    print(f"✅ Đã lưu scaler tại: {scaler_save_path}")




#GRU

def train_gru_model_from_csv(file_path, window_size=60, epochs=20, batch_size=32):
    PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
    selected_symbol = os.path.splitext(os.path.basename(file_path))[0]
    model_dir = os.path.join(PROJECT_ROOT, "models")
    os.makedirs(model_dir, exist_ok=True)

    model_save_path = os.path.join(
        model_dir, f"{selected_symbol}_gru_ws{window_size}_ep{epochs}.h5"
    )
    scaler_save_path = os.path.join(
        model_dir, f"{selected_symbol}_gru_scaler_ws{window_size}_ep{epochs}.pkl"
    )

    # Load data
    df = pd.read_csv(file_path)
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date')
    data = df['Close'].values.reshape(-1, 1)

    # Scale
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data)

    def create_dataset(series, window_size):
        X, y = [], []
        for i in range(len(series) - window_size):
            X.append(series[i:i + window_size])
            y.append(series[i + window_size])
        return np.array(X), np.array(y)

    X, y = create_dataset(scaled_data, window_size)
    X = X.reshape(-1, window_size, 1)

    # Build model
    model = Sequential()
    model.add(GRU(64, input_shape=(window_size, 1)))
    model.add(Dropout(0.2))
    model.add(Dense(1))

    model.compile(optimizer='adam', loss='mean_squared_error')
    early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

    print(f"🔧 Đang huấn luyện mô hình GRU cho {selected_symbol}...")
    model.fit(X, y, validation_split=0.1,
              epochs=epochs, batch_size=batch_size, verbose=1, callbacks=[early_stop])

    model.save(model_save_path)
    joblib.dump(scaler, scaler_save_path)
    model_type = "gru"  # hoặc "lstm" tùy vào hàm
    split_index = int(len(X) * 0.8)
    X_train, y_train = X[:split_index], y[:split_index]
    X_test, y_test = X[split_index:], y[split_index:]

    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    y_train_inv = scaler.inverse_transform(y_train.reshape(-1, 1))
    y_test_inv = scaler.inverse_transform(y_test.reshape(-1, 1))
    y_train_pred_inv = scaler.inverse_transform(y_train_pred)
    y_test_pred_inv = scaler.inverse_transform(y_test_pred)

    mae = mean_absolute_error(y_test_inv, y_test_pred_inv)
    rmse = np.sqrt(mean_squared_error(y_test_inv, y_test_pred_inv))
    mape = np.mean(np.abs((y_test_inv - y_test_pred_inv) / y_test_inv)) * 100

    # ✅ Tạo thư mục errors nếu chưa có
    error_dir = os.path.join(PROJECT_ROOT, "errors")
    os.makedirs(error_dir, exist_ok=True)

    # ✅ Tên file đầy đủ theo format chuẩn
    error_path = os.path.join(
        error_dir, f"{selected_symbol}_{model_type}_ws{window_size}_ep{epochs}_bs{batch_size}.csv"
    )

    # ✅ Lưu file lỗi
    pd.DataFrame([{
        "Symbol": selected_symbol,
        "Model": model_type.upper(),  # ⬅ dùng biến
        "Window Size": window_size,
        "Epochs": epochs,
        "Batch Size": batch_size,
        "MAE": mae,
        "RMSE": rmse,
        "MAPE": mape
    }]).to_csv(error_path, index=False)

    print(f"📁 Đã lưu lỗi dự báo tại: {error_path}")
    print(f"📁 Đã lưu lỗi dự báo tại: {error_path}")
    print(f"✅ Đã lưu GRU model tại: {model_save_path}")
    print(f"✅ Đã lưu GRU scaler tại: {scaler_save_path}")





#Biểu đồ và chỉ số lỗi
def analyze_trained_model(symbol, model_type='lstm', window_size=60, epochs=20):
    PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
    model_type = model_type.lower()
    symbol = symbol.upper()

    model_file = f"{symbol}_{model_type}_ws{window_size}_ep{epochs}.h5"
    scaler_file = f"{symbol}_{model_type}_scaler_ws{window_size}_ep{epochs}.pkl"

    model_path = os.path.join(PROJECT_ROOT, 'models', model_file)
    scaler_path = os.path.join(PROJECT_ROOT, 'models', scaler_file)
    csv_path = os.path.join(PROJECT_ROOT, 'dataset', f"{symbol}.csv")

    if not (os.path.exists(model_path) and os.path.exists(scaler_path) and os.path.exists(csv_path)):
        raise FileNotFoundError(f"❌ Không tìm thấy model, scaler hoặc dataset.\n{model_path}\n{scaler_path}\n{csv_path}")
    # Load dữ liệu
    df = pd.read_csv(csv_path)
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date')
    data = df['Close'].values.reshape(-1, 1)

    # Load model + scaler
    model = load_model(model_path)
    scaler = joblib.load(scaler_path)

    # Scale lại
    scaled_data = scaler.transform(data)

    # Tạo tập dữ liệu
    def create_dataset(series, window):
        X, y = [], []
        for i in range(len(series) - window):
            X.append(series[i:i + window])
            y.append(series[i + window])
        return np.array(X), np.array(y)

    X, y = create_dataset(scaled_data, window_size)
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    X_train = X_train.reshape(-1, window_size, 1)
    X_test = X_test.reshape(-1, window_size, 1)

    # Dự đoán
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    y_train_inv = scaler.inverse_transform(y_train.reshape(-1, 1))
    y_test_inv = scaler.inverse_transform(y_test.reshape(-1, 1))
    y_train_pred_inv = scaler.inverse_transform(y_train_pred)
    y_test_pred_inv = scaler.inverse_transform(y_test_pred)

    # Các mốc thời gian
    train_dates = df['Date'].iloc[window_size:split_idx + window_size].dropna()
    test_dates = df['Date'].iloc[split_idx + window_size:].dropna()

    # Tạo biểu đồ Plotly
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=train_dates, y=y_train_inv.flatten(), mode='lines', name='Train Real', line=dict(color='#38bdf8')))
    fig.add_trace(go.Scatter(x=train_dates, y=y_train_pred_inv.flatten(), mode='lines', name='Train Predict', line=dict(color='#facc15')))
    fig.add_trace(go.Scatter(x=test_dates, y=y_test_inv.flatten(), mode='lines', name='Test Real', line=dict(color='#22c55e')))
    fig.add_trace(go.Scatter(x=test_dates, y=y_test_pred_inv.flatten(), mode='lines', name='Test Predict', line=dict(color='#ef4444')))
    fig.update_layout(
        title=f"{model_type.upper()} Model Analysis - {symbol}",
        xaxis_title="Date",
        yaxis_title="Price",
        template="plotly_dark",
        height=480,
        margin=dict(l=20, r=20, t=40, b=20)
    )
    chart_html = pio.to_html(fig, full_html=False, include_plotlyjs='cdn')

    # Chỉ số lỗi TRAIN & TEST
    metrics_train = {
        "MAE": float(mean_absolute_error(y_train_inv, y_train_pred_inv)),
        "RMSE": float(np.sqrt(mean_squared_error(y_train_inv, y_train_pred_inv))),
        "MAPE": float(np.mean(np.abs((y_train_inv - y_train_pred_inv) / y_train_inv)) * 100)
    }
    metrics_test = {
        "MAE": float(mean_absolute_error(y_test_inv, y_test_pred_inv)),
        "RMSE": float(np.sqrt(mean_squared_error(y_test_inv, y_test_pred_inv))),
        "MAPE": float(np.mean(np.abs((y_test_inv - y_test_pred_inv) / y_test_inv)) * 100)
    }

    # Trả về dict đúng chuẩn Jinja2 template
    return {
        'model_type': model_type.upper(),
        'symbol': symbol,
        'metrics_train': metrics_train,
        'metrics_test': metrics_test,
        'chart_html': chart_html,
        'summary': "Đánh giá mô hình dựa trên tập Train và Test."
    }




def predict_future(df, scaler, window_size=60, steps=10, symbol="AAPL", model_type="lstm", model_path=None):
    print("DEBUG: df.shape =", df.shape)
    print("DEBUG: df.columns =", df.columns)
    print("DEBUG: df.head() =", df.head())
    model_type = model_type.lower()
    if model_type not in ["lstm", "gru"]:
        raise ValueError("❌ model_type phải là 'lstm' hoặc 'gru'")

    # ✅ Tìm model mới nhất nếu không truyền vào
    if model_path is None:
        model_files = glob.glob(f"models/{symbol}_{model_type}_ws*.h5")
        if not model_files:
            raise FileNotFoundError(f"❌ Không tìm thấy mô hình: models/{symbol}_{model_type}_ws*.h5")
        model_path = max(model_files, key=os.path.getmtime)

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"❌ Không tìm thấy mô hình tại {model_path}")
    model = load_model(model_path)

    if 'Close' not in df.columns or len(df) < window_size:
        print(f"❌ Dữ liệu không đủ. df có {len(df)} dòng, cần ít nhất {window_size} với cột 'Close'.")
        return pd.DataFrame([])

    last_sequence = df['Close'].values[-window_size:]
    last_scaled = scaler.transform(last_sequence.reshape(-1, 1))
    input_seq = last_scaled.reshape(1, window_size, 1)
    last_date = pd.to_datetime(df['Date']).max()
    forecast = []

    for _ in range(steps):
        pred_scaled = model.predict(input_seq, verbose=0)
        pred_price = scaler.inverse_transform(pred_scaled)[0][0]

        next_date = last_date + timedelta(days=1)
        while next_date.weekday() >= 5:
            next_date += timedelta(days=1)

        forecast.append({"Date": next_date, "Close": round(pred_price, 2)})
        last_date = next_date
        input_seq = np.append(input_seq[:, 1:, :], pred_scaled.reshape(1, 1, 1), axis=1)

    forecast_df = pd.DataFrame(forecast)[["Date", "Close"]]
    print("DEBUG: forecast_df =", forecast_df)
    return forecast_df



def plot_forecast_chart(df, forecast_df):
    # df: DataFrame gốc (có Date, Close)
    # forecast_df: DataFrame dự báo (có Date, Close)

    df = df.copy()
    df['Date'] = pd.to_datetime(df['Date'])

    # Dữ liệu thực tế
    trace_actual = go.Scatter(
        x=df['Date'], y=df['Close'],
        mode='lines', name='Actual Price', line=dict(color='royalblue')
    )
    # Dự báo
    forecast_dates = forecast_df['Date']
    forecast_prices = forecast_df['Close']
    trace_pred = go.Scatter(
        x=forecast_dates, y=forecast_prices,
        mode='lines+markers', name='Forecast', line=dict(color='orangered', dash='dash')
    )

    layout = go.Layout(
        title='Price Forecast',
        xaxis_title='Date',
        yaxis_title='Price',
        template='plotly_dark'
    )
    fig = go.Figure([trace_actual, trace_pred], layout)
    chart_html = fig.to_html(full_html=False, include_plotlyjs='cdn')
    return chart_html


def log(message):
    print(f"[LOG] {message}")

def evaluate_with_yahoo(symbol, model_type='lstm', window_size=60, steps=30):
    try:
        log(f"🚀 Đang đánh giá mô hình {model_type.upper()} cho {symbol}...")

        model_path = os.path.join("models", f"{symbol}_{model_type}_model.h5")
        scaler_path = os.path.join("models", f"{symbol}_{model_type}_scaler.pkl")
        if not os.path.exists(model_path) or not os.path.exists(scaler_path):
            raise FileNotFoundError("Không tìm thấy model hoặc scaler đã huấn luyện.")

        log("✅ Tải model và scaler...")
        model = load_model(model_path)
        scaler = joblib.load(scaler_path)

        log("📥 Đang tải dữ liệu từ Yahoo Finance...")
        df_yahoo = yf.download(symbol, start="2022-07-12", end="2022-09-30", auto_adjust=True)
        df_yahoo.reset_index(inplace=True)
        df_yahoo = df_yahoo[['Date', 'Close']].dropna()
        df_yahoo['Date'] = pd.to_datetime(df_yahoo['Date'])

        log("📊 Tạo dữ liệu dự báo tương lai...")
        last_sequence = df_yahoo['Close'].values[-window_size:]
        last_scaled = scaler.transform(last_sequence.reshape(-1, 1))
        input_seq = last_scaled.reshape(1, window_size, 1)
        last_date = df_yahoo['Date'].max()

        forecast = []
        for i in range(steps):
            pred = model.predict(input_seq, verbose=0)
            pred_price = scaler.inverse_transform(pred)[0][0]
            next_date = last_date + timedelta(days=1)
            while next_date.weekday() >= 5:
                next_date += timedelta(days=1)
            forecast.append((next_date, pred_price))
            last_date = next_date
            input_seq = np.append(input_seq[:, 1:, :], pred.reshape(1, 1, 1), axis=1)

        log(f"📅 Đã tạo {len(forecast)} ngày dự báo.")

        # Ghép tay với thực tế
        actual_dict = df_yahoo.set_index('Date')['Close'].to_dict()
        matched = []
        for date, pred_price in forecast:
            if date in actual_dict:
                matched.append({
                    'Date': date,
                    'Predicted': pred_price,
                    'Actual': actual_dict[date]
                })

        if not matched:
            log("⚠️ Không có ngày nào trùng giữa dự báo và dữ liệu thực tế.")
            return {"error": "Không có dữ liệu thực tế trùng với dự báo."}

        result_df = pd.DataFrame(matched).sort_values('Date')

        mae = mean_absolute_error(result_df['Actual'], result_df['Predicted'])
        rmse = np.sqrt(mean_squared_error(result_df['Actual'], result_df['Predicted']))
        mape = np.mean(np.abs((result_df['Actual'] - result_df['Predicted']) / result_df['Actual'])) * 100

        log(f"📈 MAE: {mae:.4f}, RMSE: {rmse:.4f}, MAPE: {mape:.2f}%")

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=result_df['Date'], y=result_df['Actual'], name='Actual', mode='lines+markers'))
        fig.add_trace(go.Scatter(x=result_df['Date'], y=result_df['Predicted'], name='Forecast', mode='lines+markers', line=dict(dash='dot')))
        fig.update_layout(title=f"Forecast vs Actual - {symbol}",
                          xaxis_title="Date", yaxis_title="Price",
                          template="plotly_white")
        chart_html = fig.to_html(full_html=False, include_plotlyjs='cdn')

        return {
            "symbol": symbol,
            "model_type": model_type.upper(),
            "steps": steps,
            "mae": round(mae, 4),
            "rmse": round(rmse, 4),
            "mape": round(mape, 2),
            "chart_html": chart_html
        }

    except Exception as e:
        log(f"❌ Lỗi khi đánh giá mô hình: {str(e)}")
        return {"error": str(e)}
