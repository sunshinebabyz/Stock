import pandas as pd
import numpy as np
import os
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from scipy import stats
from sklearn.metrics import mean_squared_error, mean_absolute_error
from datetime import datetime, timedelta
import joblib
import glob
from statsmodels.tsa.arima.model import ARIMA
import plotly.io as pio
from pmdarima import auto_arima
from statsmodels.tsa.holtwinters import Holt



def predict_stock(df, model_type, forecast_days):
    if 'Close' not in df.columns:
        raise ValueError("Dataset must contain 'Close' column for prediction.")

    last_price = df['Close'].iloc[-1]
    prediction = [last_price * (1 + np.random.normal(0, 0.01)) for _ in range(forecast_days)]
    future_dates = pd.date_range(start=pd.to_datetime(df['Date'].iloc[-1]) + timedelta(days=1), periods=forecast_days)

    pred_df = pd.DataFrame({'Date': future_dates, 'Predicted Close': prediction})

    fig = px.line(pred_df, x='Date', y='Predicted Close', title=f"Predicted Prices - {model_type}")
    output_dir = 'static'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    chart_path = os.path.join(output_dir, f"{model_type}_forecast_chart.html")
    fig.write_html(chart_path)

    return prediction, chart_path

# 1️⃣ Load Dataset
def load_dataset(symbol):
    file_path = os.path.join('dataset', f"{symbol}.csv")
    if os.path.exists(file_path):
        return pd.read_csv(file_path)
    else:
        return None

# 2️⃣ Calculate Statistics
def calculate_statistics(df, start_date=None, end_date=None):
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date']).dt.tz_localize(None)
        if start_date and end_date:
            start_date = pd.to_datetime(start_date)
            end_date = pd.to_datetime(end_date)
            df = df[(df['Date'] >= start_date) & (df['Date'] <= end_date)]
    columns = ['Close', 'Open', 'High', 'Low', 'Volume']
    available_cols = [col for col in columns if col in df.columns]
    df_numeric = df[available_cols]
    statistics = df_numeric.describe().to_dict()
    for col in available_cols:
        statistics[col]['Mode'] = df_numeric[col].mode()[0] if not df_numeric[col].mode().empty else np.nan
        statistics[col]['Sample Variance'] = df_numeric[col].var()
        statistics[col]['Kurtosis'] = df_numeric[col].kurt()
        statistics[col]['Skewness'] = df_numeric[col].skew()
        statistics[col]['Range'] = df_numeric[col].max() - df_numeric[col].min()
        statistics[col]['Sum'] = df_numeric[col].sum()
        try:
            ci = stats.t.interval(0.95, len(df_numeric[col])-1, loc=np.mean(df_numeric[col]), scale=stats.sem(df_numeric[col]))
            statistics[col]['Confidence Interval (95%)'] = f"{ci[0]:.2f} - {ci[1]:.2f}"
        except:
            statistics[col]['Confidence Interval (95%)'] = "N/A"
    return df[available_cols].describe().transpose()



# 3️⃣ Plot Price Chart
def plot_price_chart(df, symbol):
    if 'Date' not in df.columns:
        df = df.reset_index()
    df['Date'] = pd.to_datetime(df['Date']).dt.tz_localize(None)

    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Scatter(x=df['Date'], y=df['Close'], mode='lines', name='Close', line=dict(color='green')), secondary_y=False)
    fig.add_trace(go.Scatter(x=df['Date'], y=df['Open'], mode='lines', name='Open', line=dict(color='blue')), secondary_y=False)
    fig.add_trace(go.Bar(x=df['Date'], y=df['Volume'], name='Volume', marker_color='rgba(0, 0, 0, 0.6)'), secondary_y=True)

    fig.update_layout(
        title=f"Stock Price & Volume - {symbol}",
        xaxis_title="Date", 
        yaxis_title="Price", 
        yaxis2_title="Volume", 
        height=600,
        barmode='overlay', 
        xaxis_rangeslider_visible=True,
        template='plotly_white'
    )

    output_dir = 'static'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    chart_path = os.path.join(output_dir, f"{symbol}_price_chart.html")
    fig.write_html(chart_path)

    return chart_path  # Trả về path để hiển thị trên giao diện HTML

# 4️⃣ Evaluate Forecast
def evaluate_forecast(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    return {"MAE": mae, "RMSE": rmse, "MAPE (%)": mape}

# 5️⃣ Load Trained Model (dành cho các model khác, không phải LSTM)
def load_trained_model(model_path):
    if os.path.exists(model_path):
        return joblib.load(model_path)
    else:
        return None

#Biểu đồ hộp
def plot_box_chart(df, symbol):
    price_columns = ['Close', 'Open', 'High', 'Low']
    available_cols = [col for col in price_columns if col in df.columns]

    if not available_cols:
        raise ValueError("Không có cột giá chứng khoán hợp lệ trong dữ liệu.")

    fig = px.box(
        df,
        y=available_cols,
        points=False
    )

    fig.update_layout(
        template='plotly_dark',
        yaxis_title="Giá Trị",
        xaxis_title="Chỉ Số",
        height=600
    )

    # ✅ Thay vì lưu file, trả về trực tiếp HTML code để nhúng vào template
    return fig.to_html(full_html=False, include_plotlyjs='cdn')


def plot_correlation_heatmap(df):
    import plotly.express as px

    corr_df = df.select_dtypes(include=['float64', 'int64']).corr()

    fig = px.imshow(
        corr_df,
        text_auto=".2f",
        color_continuous_scale="RdBu_r",
        aspect="auto"
    )
    fig.update_layout(
        template='plotly_dark',
        height=600,
        autosize=True,  # ✅ Tự động điều chỉnh kích thước phù hợp container
        margin=dict(l=60, r=60, t=80, b=60),
        font=dict(color="#FFFFFF", size=14),
        xaxis=dict(side="bottom")
    )

    return fig.to_html(full_html=False, include_plotlyjs='cdn')

# Histogram Giá Đóng Cửa 
def plot_close_price_histogram(df, symbol):
    close_prices = df['Close'].dropna()

    fig = go.Figure()

    fig.add_trace(go.Histogram(
        x=close_prices,
        nbinsx=50,
        marker_color='#00BFFF',
        opacity=0.75,
        name="Phân phối giá đóng cửa"
    ))

    mean_price = close_prices.mean()
    fig.add_vline(
        x=mean_price,
        line_width=3,
        line_dash="dash",
        line_color="red",
        annotation_text=f"Trung Bình: {mean_price:.2f}",
        annotation_position="top right",
        annotation_font_color="white"
    )

    fig.update_layout(
        title_x=0.5,
        plot_bgcolor="#0d1117",
        paper_bgcolor="#0d1117",
        font_color="#FFFFFF",
        bargap=0.05,
        margin=dict(l=20, r=20, t=50, b=20),
        height=500,
        xaxis_title="Giá đóng cửa",
        yaxis_title="Tần suất xuất hiện",
        xaxis=dict(gridcolor="#333333", zerolinecolor="#555555"),
        yaxis=dict(gridcolor="#333333", zerolinecolor="#555555")
    )

    # ✅ Trả về trực tiếp HTML code thay vì đường dẫn file
    return fig.to_html(full_html=False, include_plotlyjs='cdn')

def plot_volume_histogram(df, symbol, bins=50):
    """
    Vẽ biểu đồ histogram cho số lượng giao dịch (Volume).
    Trả về nội dung HTML của biểu đồ, không lưu ra file.
    """
    if 'Volume' not in df.columns:
        raise ValueError("Dataset không có cột 'Volume'.")

    fig = px.histogram(
        df,
        x='Volume',
        nbins=bins,
        color_discrete_sequence=['#1f77b4']
    )

    fig.update_layout(
        template='plotly_dark',
        xaxis_title="Volume",
        yaxis_title="Tần suất",
        height=600
    )

    # ✅ Không lưu file, trả về HTML để hiển thị trực tiếp
    return fig.to_html(full_html=False, include_plotlyjs='cdn')

# Candlestick
def plot_candlestick_chart(df, symbol):
    """
    Vẽ biểu đồ nến và trả về HTML code thay vì lưu file.
    """
    if not {'Date', 'Open', 'High', 'Low', 'Close'}.issubset(df.columns):
        raise ValueError("❌ Dữ liệu thiếu các cột cần thiết để vẽ biểu đồ nến!")

    fig = go.Figure(data=[go.Candlestick(
        x=df['Date'],
        open=df['Open'],
        high=df['High'],
        low=df['Low'],
        close=df['Close'],
        increasing_line_color='green',
        decreasing_line_color='red',
        showlegend=False
    )])

    fig.update_layout(
        
        xaxis_title="Ngày",
        yaxis_title="Giá",
        template="plotly_dark",
        xaxis_rangeslider_visible=False,
        height=500
    )

    # ✅ Trả về HTML để render trực tiếp trên giao diện
    return fig.to_html(full_html=False, include_plotlyjs='cdn')

# RSI tính
def calculate_rsi(df, period=14, price_column='Close'):
    """
    Tính chỉ số RSI (Relative Strength Index).
    - df: DataFrame chứa dữ liệu giá.
    - period: Số chu kỳ tính RSI, mặc định 14.
    - price_column: Cột giá đóng cửa.
    """
    delta = df[price_column].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.rolling(window=period, min_periods=period).mean()
    avg_loss = loss.rolling(window=period, min_periods=period).mean()

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))

    df['RSI'] = rsi
    return df

# RSI
def plot_rsi_chart(df, symbol):
    if 'RSI' not in df.columns:
        df = calculate_rsi(df)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df['Date'],
        y=df['RSI'],
        mode='lines',
        name='RSI',
        line=dict(color='orange')
    ))

    fig.add_hline(y=70, line=dict(color='red', dash='dash'),
                  annotation_text='Quá mua (70)', annotation_position='top right')
    fig.add_hline(y=30, line=dict(color='green', dash='dash'),
                  annotation_text='Quá bán (30)', annotation_position='bottom right')

    fig.update_layout(
        
        xaxis_title="Ngày",
        yaxis_title="RSI",
        yaxis_range=[0, 100],
        template="plotly_dark",
        height=500
    )

    # Trả về HTML inline (không lưu file)
    return fig.to_html(full_html=False, include_plotlyjs='cdn')

# Tổng trung bình giá đóng cửa theo tháng
def plot_close_volume_chart(df, symbol):
    if 'Date' not in df.columns:
        raise ValueError("Dataset cần có cột 'Date'.")
    if 'Close' not in df.columns or 'Volume' not in df.columns:
        raise ValueError("Dataset cần có cột 'Close' và 'Volume'.")

    df['Date'] = pd.to_datetime(df['Date']).dt.tz_localize(None)

    fig = go.Figure()

    # Giá đóng cửa - Line Chart
    fig.add_trace(go.Scatter(
        x=df['Date'], y=df['Close'], name="Giá Đóng Cửa",
        mode='lines', line=dict(color='#00BFFF', width=2)
    ))

    # Volume - Bar Chart (Màu đậm và rõ nét hơn)
    fig.add_trace(go.Bar(
        x=df['Date'], y=df['Volume'], name="Khối lượng giao dịch",
        marker_color='#FF4C4C',  
        opacity=0.85,  
        yaxis='y2'
    ))

    fig.update_layout(
        
        xaxis=dict(title='Ngày'),
        yaxis=dict(title='Giá đóng cửa'),
        yaxis2=dict(title='Khối lượng', overlaying='y', side='right'),
        template='plotly_dark',
        legend=dict(x=0, y=1.1, orientation='h'),
        height=600
    )

    # ✅ Không lưu file, trả về HTML inline
    return fig.to_html(full_html=False, include_plotlyjs='cdn')


def plot_monthly_total_close(df, symbol):
    import plotly.graph_objects as go
    import pandas as pd

    if 'Date' not in df.columns or 'Close' not in df.columns:
        raise ValueError("Dataset cần có cột 'Date' và 'Close'.")

    # Chuẩn hóa cột Date và trích xuất Tháng
    df['Date'] = pd.to_datetime(df['Date']).dt.tz_localize(None)
    df['Month'] = df['Date'].dt.to_period('M').astype(str)

    # Tính Tổng Giá Đóng Cửa theo Tháng
    monthly_total = df.groupby('Month')['Close'].sum().reset_index()

    # Vẽ biểu đồ
    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=monthly_total['Month'],
        y=monthly_total['Close'],
        name='Tổng giá đóng cửa',
        marker_color='#1f77b4',
        opacity=0.8
    ))

    fig.update_layout(
        
        xaxis_title="Tháng",
        yaxis_title="Tổng giá đóng cửa",
        template='plotly_dark',
        height=600,
        margin=dict(l=60, r=60, t=80, b=60),
        title_x=0.5
    )

    return fig.to_html(full_html=False, include_plotlyjs='cdn')



def plot_monthly_average_close(df, symbol):
    import plotly.graph_objects as go
    import pandas as pd

    if 'Date' not in df.columns or 'Close' not in df.columns:
        raise ValueError("Dataset cần có cột 'Date' và 'Close'.")

    df['Date'] = pd.to_datetime(df['Date']).dt.tz_localize(None)
    df['Month'] = df['Date'].dt.to_period('M').astype(str)

    monthly_avg = df.groupby('Month')['Close'].mean().reset_index()

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=monthly_avg['Month'],
        y=monthly_avg['Close'],
        mode='lines+markers',
        name='Trung bình giá đóng cửa',
        line=dict(color='#FF4C4C', width=3),
        marker=dict(size=6, color='#FF4C4C')
    ))

    fig.update_layout(
        
        xaxis_title="Tháng",
        yaxis_title="Giá trung bình",
        template='plotly_dark',
        height=600,
        margin=dict(l=60, r=60, t=80, b=60),
        title_x=0.5
    )

    return fig.to_html(full_html=False, include_plotlyjs='cdn')



def plot_monthly_traded_value(df, symbol):
    import plotly.graph_objects as go
    import pandas as pd

    if 'Date' not in df.columns or 'Close' not in df.columns or 'Volume' not in df.columns:
        raise ValueError("Dataset cần có các cột 'Date', 'Close' và 'Volume'.")

    # Chuẩn hóa dữ liệu
    df['Date'] = pd.to_datetime(df['Date']).dt.tz_localize(None)
    df['Month'] = df['Date'].dt.to_period('M').astype(str)

    # Tính giá trị giao dịch theo tháng
    df['Traded_Value'] = df['Close'] * df['Volume']
    monthly_traded = df.groupby('Month')['Traded_Value'].sum().reset_index()

    # Vẽ biểu đồ vùng (Area Chart)
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=monthly_traded['Month'],
        y=monthly_traded['Traded_Value'],
        mode='lines+markers',
        fill='tozeroy',
        name='Giá trị giao dịch',
        line=dict(color='#00BFFF', width=3),
        marker=dict(size=6, color='#00BFFF'),
        opacity=0.8
    ))

    fig.update_layout(
        
        xaxis_title="Tháng",
        yaxis_title="Giá trị giao dịch",
        template='plotly_dark',
        legend=dict(x=0, y=1.1, orientation='h'),
        height=600
    )

    return fig.to_html(full_html=False, include_plotlyjs='cdn')


def plot_seasonality(df, symbol, freq='M'):
    import pandas as pd
    import plotly.express as px

    if 'Date' not in df.columns or 'Close' not in df.columns:
        raise ValueError("Dataset cần có cột 'Date' và 'Close'.")

    df['Date'] = pd.to_datetime(df['Date']).dt.tz_localize(None)

    if freq == 'M':
        df['Month'] = df['Date'].dt.month
        seasonality = df.groupby('Month')['Close'].mean().reset_index()
        x_label = "Tháng"
    elif freq == 'Q':
        df['Quarter'] = df['Date'].dt.quarter
        seasonality = df.groupby('Quarter')['Close'].mean().reset_index()
        x_label = "Quý"
    else:
        raise ValueError("Tần suất chỉ chấp nhận 'M' (Tháng) hoặc 'Q' (Quý).")

    fig = px.line(
        seasonality,
        x=seasonality.columns[0],
        y='Close',
        markers=True,
        
    )

    fig.update_layout(
        template='plotly_dark',
        xaxis_title=x_label,
        yaxis_title="Trung bình giá đóng cửa",
        height=500
    )

    return fig.to_html(full_html=False, include_plotlyjs='cdn')

def plot_decomposition(df, symbol, period=30):
    import pandas as pd
    import plotly.graph_objects as go
    from statsmodels.tsa.seasonal import seasonal_decompose

    if 'Date' not in df.columns or 'Close' not in df.columns:
        raise ValueError("Dataset cần có cột 'Date' và 'Close'.")

    df['Date'] = pd.to_datetime(df['Date']).dt.tz_localize(None)
    df = df.sort_values('Date')
    df.set_index('Date', inplace=True)

    # Decompose
    result = seasonal_decompose(df['Close'], model='additive', period=period)

    # Vẽ biểu đồ
    fig = make_subplots(rows=4, cols=1, shared_xaxes=True,
                        subplot_titles=("Original Series", "Trend", "Seasonality", "Residual"))

    fig.add_trace(go.Scatter(x=result.observed.index, y=result.observed, name='Original', line=dict(color='#1f77b4')), row=1, col=1)
    fig.add_trace(go.Scatter(x=result.trend.index, y=result.trend, name='Trend', line=dict(color='#FF4C4C')), row=2, col=1)
    fig.add_trace(go.Scatter(x=result.seasonal.index, y=result.seasonal, name='Seasonality', line=dict(color='#00BFFF')), row=3, col=1)
    fig.add_trace(go.Scatter(x=result.resid.index, y=result.resid, name='Residual', line=dict(color='#AAAAAA')), row=4, col=1)

    fig.update_layout(
        height=800,
        template='plotly_dark',
        
        showlegend=False
    )

    return fig.to_html(full_html=False, include_plotlyjs='cdn')


def plot_moving_averages_with_multi_forecast(df, symbol, ma_windows=[5, 20, 50, 100], forecast_days=5):
    df['Date'] = pd.to_datetime(df['Date']).dt.tz_localize(None)
    df = df.sort_values('Date').copy()
    error_metrics = {}

    # Chia dữ liệu
    split_idx = int(len(df) * 0.8)
    train_df = df.iloc[:split_idx].copy()
    test_df = df.iloc[split_idx:].copy()

    # Tính MA và đánh giá sai số trên tập test
    for window in ma_windows:
        col = f"MA{window}"

        # Tính MA trên toàn bộ (cho biểu đồ)
        df[col] = df['Close'].rolling(window=window).mean()

            # Tính MA trên train để dùng làm dự báo test
        train_df[col] = df[col].iloc[:split_idx]
        train_clean = train_df.dropna(subset=[col])
        if not train_clean.empty:
            y_train = train_clean['Close']
            y_train_pred = train_clean[col]
        else:
            y_train = y_train_pred = []

        test_df[col] = df[col].iloc[split_idx:]
        test_clean = test_df.dropna(subset=[col])
        if not test_clean.empty:
            y_test = test_clean['Close']
            y_test_pred = test_clean[col]
        else:
            y_test = y_test_pred = []

        error_metrics[col] = {
            "Train": {
                "MAE": round(mean_absolute_error(y_train, y_train_pred), 4) if len(y_train) else None,
                "RMSE": round(np.sqrt(mean_squared_error(y_train, y_train_pred)), 4) if len(y_train) else None,
                "MAPE": round(np.mean(np.abs((y_train - y_train_pred) / y_train)) * 100, 2) if len(y_train) else None
            },
            "Test": {
                "MAE": round(mean_absolute_error(y_test, y_test_pred), 4) if len(y_test) else None,
                "RMSE": round(np.sqrt(mean_squared_error(y_test, y_test_pred)), 4) if len(y_test) else None,
                "MAPE": round(np.mean(np.abs((y_test - y_test_pred) / y_test)) * 100, 2) if len(y_test) else None
            }
        }

        # ✅ Thêm sau khi error_metrics[col] đã xong
        save_error_to_csv(
            symbol=symbol,
            model_type=f"ma{window}",
            window_size=window,
            epochs=1,
            batch_size=1,
            mae=error_metrics[col]["Test"]["MAE"],
            rmse=error_metrics[col]["Test"]["RMSE"],
            mape=error_metrics[col]["Test"]["MAPE"]
        )
        



    # Dự báo tương lai
    forecast_df = pd.DataFrame()
    forecast_df['Date'] = pd.date_range(start=df['Date'].iloc[-1] + pd.Timedelta(days=1), periods=forecast_days)
    for window in ma_windows:
        temp = df['Close'].tolist()
        forecast = []
        for _ in range(forecast_days):
            avg = np.mean(temp[-window:])
            forecast.append(avg)
            temp.append(avg)
        forecast_df[f"Forecast_MA{window}"] = forecast

    # Vẽ actual train
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=train_df['Date'],
        y=train_df['Close'],
        mode='lines',
        name='Actual Train',
        line=dict(color='white', width=2, dash='solid')
    ))
    # Vẽ actual test
    fig.add_trace(go.Scatter(
        x=test_df['Date'],
        y=test_df['Close'],
        mode='lines',
        name='Actual Test',
        line=dict(color='gray')
    ))


    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    dashes = ['dashdot', 'dot', 'dash', 'longdash']

    for i, window in enumerate(ma_windows):
        fig.add_trace(go.Scatter(
            x=df['Date'], y=df[f"MA{window}"], mode='lines',
            name=f"MA{window}", line=dict(color=colors[i], dash='dot')))
        fig.add_trace(go.Scatter(
            x=forecast_df['Date'], y=forecast_df[f"Forecast_MA{window}"],
            mode='lines+markers',
            name=f"Forecast MA{window}", line=dict(color=colors[i], dash=dashes[i])))

    fig.update_layout(
        xaxis_title="Date",
        yaxis_title="Close Price",
        template='plotly_dark',
        height=600,
        legend=dict(x=0, y=1.1, orientation='h'),
        title_x=0.5
    )

    return fig.to_html(full_html=False, include_plotlyjs='cdn'), error_metrics


def get_latest_close_price(df):
    """
    Trả về giá đóng cửa gần nhất từ DataFrame.
    """
    return round(df['Close'].iloc[-1], 2)

def get_forecast_price_after_n_days(df, window=5, days_ahead=5):
    """
    Dự báo giá sau N ngày sử dụng phương pháp trung bình trượt (Moving Average).
    Chỉ trả về giá trị dự báo ở ngày thứ N.
    """
    values = df['Close'].tolist()
    for _ in range(days_ahead):
        avg = sum(values[-window:]) / window
        values.append(avg)
    return round(values[-1], 2)



def exponential_smoothing_forecast(df, symbol='AAPL', alpha=0.3, forecast_days=5):
    """
    Dự báo bằng Exponential Smoothing:
    - Dự báo trên tập test (80%-20%) để đánh giá lỗi.
    - Dự báo tương lai (forward forecast) với forecast_days bước.
    """
    if 'Date' not in df.columns or 'Close' not in df.columns:
        raise ValueError("Dataset must contain 'Date' and 'Close' columns.")

    df = df.copy()
    df['Date'] = pd.to_datetime(df['Date']).dt.tz_localize(None)
    df = df.sort_values('Date')

    # Chia dữ liệu
    split_idx = int(len(df) * 0.8)
    train_df = df.iloc[:split_idx].copy()
    test_df = df.iloc[split_idx:].copy()

    # Dự báo tập test
    forecasts = []
    history = train_df['Close'].tolist()
    for _ in range(len(test_df)):
        forecast = pd.Series(history).ewm(alpha=alpha, adjust=False).mean().iloc[-1]
        forecasts.append(forecast)
        history.append(forecast)  # append chính dự báo, không dùng giá trị thật
    test_df['Forecast_ES'] = forecasts


    y_true = test_df['Close']
    y_pred = test_df['Forecast_ES']
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100

    # Dự báo tương lai
    future_history = df['Close'].tolist()
    for _ in range(forecast_days):
        next_forecast = pd.Series(future_history).ewm(alpha=alpha, adjust=False).mean().iloc[-1]
        future_history.append(next_forecast)
    future_forecast = future_history[-forecast_days:]
    future_dates = pd.date_range(start=df['Date'].iloc[-1] + pd.Timedelta(days=1), periods=forecast_days)
    forecast_df = pd.DataFrame({'Date': future_dates, 'Forecast': future_forecast})
    forecast_day_value = forecast_df['Forecast'].iloc[-1] 

    fig = go.Figure()

    # 1. Actual Price (full)
    fig.add_trace(go.Scatter(
        x=df['Date'], y=df['Close'],
        name='Actual Price',
        line=dict(color='white')
    ))

    # 2. Actual Test (riêng phần test_df)
    if not test_df.empty:
        fig.add_trace(go.Scatter(
            x=test_df['Date'], y=test_df['Close'],
            name='Actual Test',
            line=dict(color='gray')
        ))

    # 3. Forecast (Train) - đường ES trên tập train (giữ nguyên α)
    es_train_series = train_df['Close'].ewm(alpha=alpha, adjust=False).mean()
    fig.add_trace(go.Scatter(
        x=train_df['Date'],
        y=es_train_series,
        name=f'ES Train (α={alpha})',
        line=dict(color='red', dash='dot')
    ))

    # 4. Forecast (Test)
    fig.add_trace(go.Scatter(
        x=test_df['Date'], y=test_df['Forecast_ES'],
        name='Forecast (Test)',
        mode='lines+markers',
        line=dict(color='cyan'),
        marker=dict(size=5, color='cyan')
    ))

    # 5. Forecast tương lai (multi-step)
    if not forecast_df.empty:
        if len(forecast_df) > 1:
            fig.add_trace(go.Scatter(
                x=forecast_df['Date'], y=forecast_df['Forecast'],
                mode='lines+markers',
                name=f'Forecast Next {forecast_days}D',
                line=dict(color='orange', dash='dash'),
                marker=dict(size=6, color='orange')
            ))
        else:
            fig.add_trace(go.Scatter(
                x=forecast_df['Date'], y=forecast_df['Forecast'],
                mode='markers+text',
                name=f'Forecast Next {forecast_days}D',
                marker=dict(size=8, color='orange'),
                text=[f"{forecast_day_value:.2f}"],
                textposition="top center"
            ))

    # 6. Forecast End Point
    if forecast_day_value:
        fig.add_trace(go.Scatter(
            x=[forecast_df['Date'].iloc[-1]], y=[forecast_day_value],
            mode='markers+text',
            name='Forecast End',
            marker=dict(size=10, color='orange'),
            text=[f"{forecast_day_value:.2f}"],
            textposition="top center"
        ))

    # 7. Layout
    fig.update_layout(
        xaxis_title="Date",
        yaxis_title="Close Price",
        template="plotly_dark",
        height=600,
        legend=dict(x=0, y=1.1, orientation='h'),
        title_x=0.5
    )

    # 🔸 Tính lỗi train (giả lập bằng smoothing)
    es_train_series = train_df['Close'].ewm(alpha=alpha, adjust=False).mean()
    y_train_true = train_df['Close']
    y_train_pred = es_train_series

    mae_train = mean_absolute_error(y_train_true, y_train_pred)
    rmse_train = np.sqrt(mean_squared_error(y_train_true, y_train_pred))
    mape_train = np.mean(np.abs((y_train_true - y_train_pred) / y_train_true)) * 100

    # 🔸 Tính lỗi test như cũ
    mae_test = mean_absolute_error(y_true, y_pred)
    rmse_test = np.sqrt(mean_squared_error(y_true, y_pred))
    mape_test = np.mean(np.abs((y_true - y_pred) / y_true)) * 100

    # 🔸 Gộp lại dictionary
    es_errors = {
        "Train": {
            "MAE": mae_train,
            "RMSE": rmse_train,
            "MAPE": mape_train
        },
        "Test": {
            "MAE": mae_test,
            "RMSE": rmse_test,
            "MAPE": mape_test,
            "Forecast_NDay": round(future_forecast[-1], 2)
        }
    }

    # 🔸 Ghi file lỗi ES
    save_error_to_csv(
        symbol=symbol,
        model_type="es",
        window_size=int(alpha * 100),
        epochs=1,
        batch_size=1,
        mae=mae_test,
        rmse=rmse_test,
        mape=mape_test
    )

    return pio.to_html(fig, full_html=False, include_plotlyjs='cdn'), es_errors



def get_forecast_price_es_1_day(df, alpha=0.3):
    """
    Dự báo giá đóng cửa cho ngày tiếp theo (1 day ahead)
    sử dụng Exponential Smoothing (trên toàn bộ tập dữ liệu hiện tại).
    """
    if 'Close' not in df.columns:
        return None

    df = df.copy()
    df = df.sort_values('Date')
    df['Date'] = pd.to_datetime(df['Date']).dt.tz_localize(None)

    if df.empty or len(df) < 2:
        return None

    # Tính ES trên toàn bộ dữ liệu
    es_series = df['Close'].ewm(alpha=alpha, adjust=False).mean()
    last_smoothed = es_series.iloc[-1]
    last_actual = df['Close'].iloc[-1]

    # Công thức ES để dự báo ngày kế tiếp
    forecast_1day = alpha * last_actual + (1 - alpha) * last_smoothed


    return round(forecast_1day, 2)


def arima_forecast(df, symbol, save_dir='static/charts', forecast_days=5):
    def find_growth_start_date(series, min_window=30):
        for i in range(len(series) - min_window):
            window = series[i:i + min_window]
            if window.is_monotonic_increasing or window.corr(pd.Series(range(len(window)))) > 0.8:
                return i
        return 0

    # 1. Chuẩn hóa dữ liệu
    df = df[['Date', 'Close']].copy()
    df = df.dropna(subset=['Close'])  # loại bỏ NaN trong cột Close
    df['Date'] = pd.to_datetime(df['Date'])
    df_original = df.copy()

    # 🔍 Tìm điểm bắt đầu tăng trưởng
    growth_start_idx = find_growth_start_date(df['Close'])

    if isinstance(growth_start_idx, int):
        growth_start = df.iloc[growth_start_idx]['Date']
    else:
        growth_start = pd.to_datetime(growth_start_idx)

    df = df[df['Date'] >= growth_start]

    # 📉 Nếu quá ít dữ liệu thì fallback dùng toàn bộ
    if len(df) < 50:
        df = df_original.copy()
        growth_start = df['Date'].iloc[0]

    df = df[df['Close'] > 0]
    df = df.sort_values('Date')
    df.set_index('Date', inplace=True)

    if df.empty or len(df) < 30:
        return None, {
            "MAE": None, "RMSE": None, "MAPE": "NaN",
            "Forecast_ARIMA": None,
            "Growth_Start": "N/A"
        }

    log_series = np.log(df['Close'])

    # 2. Chia dữ liệu train/test
    split_idx = int(len(log_series) * 0.8)
    train, test = log_series[:split_idx], log_series[split_idx:]

    # 3. Train ARIMA
    auto_model = auto_arima(train, seasonal=False, stepwise=True,
                            error_action='ignore', suppress_warnings=True)
    model_fit = auto_model.fit(train)
    # 👉 Fitted values trên tập train (log)
    fitted_log_train = model_fit.predict_in_sample()
    fitted_train = pd.Series(np.exp(fitted_log_train))
    fitted_train.index = train.index[:len(fitted_train)]  # 🔁 ép lại index khớp chính xác

    # ✅ Tính lỗi Train (sau khi ép index)
    train_actual = pd.Series(np.exp(train.values[:len(fitted_train)]), index=fitted_train.index)
    train_mae = mean_absolute_error(train_actual, fitted_train)
    train_rmse = np.sqrt(mean_squared_error(train_actual, fitted_train))
    train_mape = np.mean(np.abs((train_actual - fitted_train) / train_actual)) * 100

    # 4. Dự báo và đánh giá trên tập test
    # ✅ Khởi tạo mặc định để tránh lỗi nếu vào except hoặc skip nhánh if
    test_mae, test_rmse, test_mape = None, None, None
    forecast_test = pd.Series([], dtype=float)
    test_actual = pd.Series([], dtype=float)
    if len(test) >= max(5, forecast_days):
        try:
            forecast_log_test = model_fit.predict(n_periods=len(test))
            forecast_test = pd.Series(np.exp(forecast_log_test), name='predicted')
            test_actual = pd.Series(np.exp(test.values[:len(forecast_log_test)]), name='actual')

            # Gán index thủ công: dùng chính index gốc slice từ test
            forecast_test.index = test.index[:len(forecast_log_test)]
            test_actual.index = test.index[:len(forecast_log_test)]

            df_compare = pd.concat([test_actual, forecast_test], axis=1).dropna()
            df_compare.columns = ['actual', 'predicted']

            if not df_compare.empty:
                mae = mean_absolute_error(df_compare['actual'], df_compare['predicted'])
                rmse = np.sqrt(mean_squared_error(df_compare['actual'], df_compare['predicted']))
                mape = np.mean(np.abs((df_compare['actual'] - df_compare['predicted']) / df_compare['actual'])) * 100

                # 🔁 Gán lại để export ra ngoài
                test_mae = mae
                test_rmse = rmse
                test_mape = mape
            else:
                test_mae, test_rmse, test_mape = None, None, None


        except Exception as e:
            forecast_test = pd.Series([], dtype=float)
            test_actual = pd.Series([], dtype=float)
            mae, rmse, mape = None, None, None

    # 5. Dự báo tương lai
    full_model = auto_model.fit(log_series)
    forecast_log_future = full_model.predict(n_periods=forecast_days)
    forecast_log_future = np.asarray(forecast_log_future)  # 👈 ép thành ndarray để tránh lỗi chỉ số

    if np.isnan(forecast_log_future).any():
        forecast_day_value = None
        forecast_future_series = pd.Series(dtype=float)
    else:
        forecast_future = np.exp(forecast_log_future).astype(float)
        forecast_dates = pd.date_range(start=log_series.index[-1] + pd.Timedelta(days=1), periods=forecast_days, freq='B')
        forecast_future_series = pd.Series(forecast_future, index=forecast_dates)

        last_value = forecast_future_series.iloc[-1] if not forecast_future_series.empty else None
        forecast_day_value = round(last_value, 2) if last_value is not None and not np.isnan(last_value) else None


    # Tạo biểu đồ chỉ 1 lần
    fig = go.Figure()

    # 1. Actual Price
    fig.add_trace(go.Scatter(
        x=df.index,
        y=df['Close'],
        name='Actual Price',
        line=dict(color='white', width=2)
    ))

    # 2. Actual Test
    if not test_actual.empty:
        fig.add_trace(go.Scatter(
            x=test_actual.index,
            y=test_actual,
            name='Actual Test',
            line=dict(color='gray')
        ))

    # 3. Forecast (Train)
    if not fitted_train.empty:
        fig.add_trace(go.Scatter(
            x=fitted_train.index,
            y=fitted_train.values,
            mode='lines+markers',
            name='Forecast (Train)',
            line=dict(color='red', dash='dot'),
            marker=dict(size=4)
        ))

    # 4. Forecast (Test)
    if not forecast_test.empty:
        fig.add_trace(go.Scatter(
            x=forecast_test.index,
            y=forecast_test,
            mode='lines+markers',
            name='Forecast (Test)',
            line=dict(color='cyan'),
            marker=dict(size=5)
        ))

    # 5. Forecast Next N Days
    if not forecast_future_series.empty:
        fig.add_trace(go.Scatter(
            x=forecast_future_series.index,
            y=forecast_future_series.values,
            mode='lines+markers',
            name=f'Forecast Next {forecast_days}D',
            line=dict(color='orange', dash='dash'),
            marker=dict(size=6)
        ))

    # 6. Forecast End
    if forecast_day_value is not None:
        fig.add_trace(go.Scatter(
            x=[forecast_future_series.index[-1]],
            y=[forecast_day_value],
            mode='markers+text',
            name='Forecast End',
            marker=dict(size=10, color='orange'),
            text=[f"{forecast_day_value:.2f}"],
            textposition="top center"
        ))
        fig.update_layout(
        xaxis_title="Date",
        yaxis_title="Close Price",
        template="plotly_dark",
        height=600,
        font=dict(color="white"),
        legend=dict(x=0, y=1.1, orientation='h', traceorder='normal'),
        margin=dict(l=20, r=20, t=40, b=20)
    )

    # mở rộng trục X
    if not forecast_future_series.empty:
        fig.update_xaxes(range=[df.index[0], forecast_future_series.index[-1] + pd.Timedelta(days=1)])

    os.makedirs(save_dir, exist_ok=True)
    chart_file = os.path.join(save_dir, f"{symbol}_arima.html")
    fig.write_html(chart_file)

    error_metrics = {
        "Train": {
            "MAE": round(train_mae, 4),
            "RMSE": round(train_rmse, 4),
            "MAPE": round(train_mape, 2)
        },
        "Test": {
            "MAE": round(test_mae, 4) if test_mae is not None else None,
            "RMSE": round(test_rmse, 4) if test_rmse is not None else None,
            "MAPE": round(test_mape, 2) if test_mape is not None else None,
            "Forecast_NDay": forecast_day_value
            },
    }
    
    save_error_to_csv(
        symbol=symbol,
        model_type="arima",
        window_size=1,
        epochs=1,
        batch_size=1,
        mae=error_metrics["Test"]["MAE"],
        rmse=error_metrics["Test"]["RMSE"],
        mape=error_metrics["Test"]["MAPE"]
    )


    return os.path.basename(chart_file), {
        **error_metrics,
        "Growth_Start": growth_start.strftime('%Y-%m-%d')
    }


def holt_forecast(df, symbol='AAPL', forecast_days=5, alpha=0.3, beta=0.1):

    if 'Date' not in df.columns or 'Close' not in df.columns:
        raise ValueError("Dataset must contain 'Date' and 'Close' columns.")

    df = df.copy()
    df['Date'] = pd.to_datetime(df['Date']).dt.tz_localize(None)
    df = df.sort_values('Date')

    # Split train/test
    split_idx = int(len(df) * 0.8)
    train_df = df.iloc[:split_idx].copy()
    test_df = df.iloc[split_idx:].copy()

    # Train model
    model = Holt(train_df['Close']).fit(smoothing_level=alpha, smoothing_slope=beta, optimized=False)

    fitted = pd.Series(model.fittedvalues.values, index=train_df['Date'])

    # Forecast on test set
    test_forecast = model.forecast(len(test_df))
    test_df['Forecast_Holt'] = test_forecast

    # Đánh giá tập TRAIN
    y_train_true = train_df['Close'].reset_index(drop=True)
    y_train_pred = fitted.reset_index(drop=True)
    mask_train = y_train_true != 0
    mae_train = mean_absolute_error(y_train_true, y_train_pred)
    rmse_train = np.sqrt(mean_squared_error(y_train_true, y_train_pred))
    mape_train = (
        np.mean(np.abs((y_train_true[mask_train] - y_train_pred[mask_train]) / y_train_true[mask_train])) * 100
        if mask_train.any() else None
    )

    # Đánh giá tập TEST
    y_test_true = test_df['Close'].reset_index(drop=True)
    y_test_pred = test_df['Forecast_Holt'].reset_index(drop=True)
    mask_test = y_test_true != 0
    mae_test = mean_absolute_error(y_test_true, y_test_pred)
    rmse_test = np.sqrt(mean_squared_error(y_test_true, y_test_pred))
    mape_test = (
        np.mean(np.abs((y_test_true[mask_test] - y_test_pred[mask_test]) / y_test_true[mask_test])) * 100
        if mask_test.any() else None
    )

    # Forecast tương lai
    future_forecast = model.forecast(forecast_days)
    future_dates = pd.date_range(start=df['Date'].iloc[-1] + pd.Timedelta(days=1), periods=forecast_days)
    forecast_df = pd.DataFrame({'Date': future_dates, 'Forecast': future_forecast})
    forecast_day_value = forecast_df['Forecast'].iloc[-1] if not forecast_df.empty else None


    # Vẽ biểu đồ
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df['Date'], y=df['Close'], name='Actual Price', line=dict(color='white')))
    if not test_df.empty:
        fig.add_trace(go.Scatter(x=test_df['Date'], y=test_df['Close'], name='Actual Test', line=dict(color='gray')))
    if not fitted.isnull().all():
        fig.add_trace(go.Scatter(x=fitted.index, y=fitted.values, mode='lines+markers',
                                 name='Forecast (Train)', line=dict(color='red', dash='dot'), marker=dict(size=4)))
    if not test_df['Forecast_Holt'].isnull().all():
        fig.add_trace(go.Scatter(x=test_df['Date'], y=test_df['Forecast_Holt'], mode='lines+markers',
                                 name='Forecast (Test)', line=dict(color='cyan'), marker=dict(size=5)))
    if not forecast_df.empty:
        fig.add_trace(go.Scatter(x=forecast_df['Date'], y=forecast_df['Forecast'], mode='lines+markers',
                                 name=f'Forecast Next {forecast_days}D',
                                 line=dict(color='orange', dash='dash'), marker=dict(size=6)))
    if forecast_day_value is not None:
        fig.add_trace(go.Scatter(
            x=[forecast_df['Date'].iloc[-1]],
            y=[forecast_day_value],
            mode='markers+text',
            name='Forecast End',
            marker=dict(size=10, color='orange'),
            text=[f"{forecast_day_value:.2f}"],
            textposition="top center"
        ))


    fig.update_layout(
        xaxis_title="Date",
        yaxis_title="Close Price",
        template="plotly_dark",
        height=600,
        legend=dict(x=0, y=1.15, orientation='h', traceorder='normal', itemwidth=70),
        title_x=0.5
    )

    error_metrics = {
        "Train": {
            "MAE": round(mae_train, 4),
            "RMSE": round(rmse_train, 4),
            "MAPE": round(mape_train, 2) if mape_train is not None else None
        },
        "Test": {
            "MAE": round(mae_test, 4),
            "RMSE": round(rmse_test, 4),
            "MAPE": round(mape_test, 2) if mape_test is not None else None,
            "N_Days": len(test_df),
            "Forecast_NDay": round(forecast_day_value, 2) if forecast_day_value is not None else None

        }
    }
    save_error_to_csv(
        symbol=symbol,
        model_type="holt",
        window_size=forecast_days,
        epochs=int(alpha * 100),
        batch_size=int(beta * 100),
        mae=error_metrics["Test"]["MAE"],
        rmse=error_metrics["Test"]["RMSE"],
        mape=error_metrics["Test"]["MAPE"]
    )


    return pio.to_html(fig, full_html=False, include_plotlyjs='cdn'), error_metrics

def save_error_to_csv(symbol, model_type, window_size, epochs, batch_size, mae, rmse, mape):
    os.makedirs("errors", exist_ok=True)
    file_name = f"{symbol.upper()}_{model_type.lower()}_ws{window_size}_ep{epochs}_bs{batch_size}.csv"
    error_path = os.path.join("errors", file_name)

    df = pd.DataFrame([{
        "Symbol": symbol.upper(),
        "Model": model_type.upper(),
        "Window Size": window_size,
        "Epochs": epochs,
        "Batch Size": batch_size,
        "MAE": round(mae, 4),
        "RMSE": round(rmse, 4),
        "MAPE": round(mape, 2)
    }])
    df.to_csv(error_path, index=False)




