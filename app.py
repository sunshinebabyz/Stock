import os
import joblib
import pandas as pd
import numpy as np
import subprocess
import sys
import glob
import re
import plotly.graph_objs as go
from plotly.subplots import make_subplots



from flask import Flask, render_template, request, session, url_for, redirect, jsonify
from functions import (
    load_dataset, calculate_statistics, plot_box_chart, plot_correlation_heatmap, 
    plot_close_price_histogram, plot_volume_histogram, plot_candlestick_chart, 
    plot_rsi_chart, plot_close_volume_chart, plot_monthly_total_close,
    plot_monthly_average_close, plot_monthly_traded_value, plot_seasonality,
    plot_decomposition, exponential_smoothing_forecast,
    arima_forecast, plot_moving_averages_with_multi_forecast,
    get_latest_close_price, get_forecast_price_after_n_days,
    get_forecast_price_es_1_day,holt_forecast
)


from ml import train_lstm_model_from_csv, train_gru_model_from_csv, predict_future, analyze_trained_model, plot_forecast_chart, evaluate_with_yahoo


app = Flask(__name__)
app.secret_key = 'your_secret_key'  # Bắt buộc để dùng session

# Trang chủ
@app.route('/')
def index():
    return render_template('index.html')


# Trang Ticker - Chỉ dùng để nhập mã chứng khoán và xem dự đoán
@app.route('/ticker', methods=['GET'])
def ticker():
    symbol = request.args.get('code', '').upper()
    if not symbol:
        return render_template('ticker.html', error="Please enter a stock symbol.")

    return render_template('ticker.html', symbol=symbol)


@app.route('/ticker/statistics', methods=['GET', 'POST'])
def descriptive_statistics():
    symbol = request.args.get('code', '').upper()
    if not symbol:
        return render_template('ticker.html', error="Please enter a stock symbol.")

    df = load_dataset(symbol)
    if df is None:
        return render_template('statistics.html', error=f"Không tìm thấy dữ liệu cho mã: {symbol}", symbol=symbol)

    stats_df = calculate_statistics(df)
    stats = stats_df if isinstance(stats_df, dict) else stats_df.to_dict(orient='index')

    # Chỉ giữ lại các biểu đồ thống kê mô tả
    box_chart_html = plot_box_chart(df, symbol)
    chart_html = plot_correlation_heatmap(df)
    hist_chart_html = plot_close_price_histogram(df, symbol)
    volume_histogram_html = plot_volume_histogram(df, symbol)

    return render_template(
        'statistics.html',
        symbol=symbol,
        stats=stats,
        box_chart_html=box_chart_html,
        chart_html=chart_html,
        hist_chart_html=hist_chart_html,
        volume_histogram_html=volume_histogram_html,

    )



# Trang Time Series
@app.route('/ticker/timeseries', methods=['GET', 'POST'])
def timeseries():
    symbol = request.args.get('code', '').upper()
    if not symbol:
        return render_template('ticker.html', error="Please enter a stock symbol.")

    df = load_dataset(symbol)
    if df is None:
        return render_template('timeseries.html', error=f"Không tìm thấy dữ liệu cho mã: {symbol}", symbol=symbol)


    # Chỉ các biểu đồ liên quan đến chuỗi thời gian
    close_volume_chart_html = plot_close_volume_chart(df, symbol)
    candlestick_chart_html = plot_candlestick_chart(df, symbol)
    monthly_total_close_html = plot_monthly_total_close(df, symbol)
    monthly_avg_html = plot_monthly_average_close(df, symbol)
    monthly_traded_value_html = plot_monthly_traded_value(df, symbol)
    rsi_chart_html = plot_rsi_chart(df, symbol)
    seasonality_html = plot_seasonality(df, symbol, freq='M')  
    decomposition_html = plot_decomposition(df, symbol)


    return render_template(
        'timeseries.html',
        symbol=symbol,
        close_volume_chart_html=close_volume_chart_html,
        candlestick_chart_html=candlestick_chart_html,
        monthly_total_close_html =monthly_total_close_html,
        monthly_avg_close_html=monthly_avg_html,
        rsi_chart_html=rsi_chart_html,
        monthly_traded_value_html=monthly_traded_value_html,
        seasonality_html=seasonality_html,
        decomposition_html=decomposition_html
    )




@app.route('/statistical_model', methods=['GET', 'POST'])
def statistical_model():
    symbol = request.args.get('code', '').upper()
    model = request.args.get('model')
    alpha = float(request.args.get('alpha', 0.3))
    beta = float(request.args.get('beta', 0.1)) 
    steps = int(request.args.get('steps', 5))

    # ✅ Load dữ liệu trước
    df = load_dataset(symbol)
    if df is None:
        return render_template('statistical_model.html', error=f"Không tìm thấy dữ liệu cho mã: {symbol}", symbol=symbol)

 # ✅ Khởi tạo tất cả các biến
    moving_averages_html, ma_errors = None, None
    arima_chart_file, arima_errors = None, None
    holt_html, holt_errors = None, None
    es_html, es_errors = None, None
    latest_price = None
    forecast_day5 = None
    forecast_title = "Predicted Price"

    # Gọi hàm phù hợp theo model
    if model == 'MA':
        forecast_title = "Predicted Price 5 Days"
        moving_averages_html, ma_errors = plot_moving_averages_with_multi_forecast(df, symbol)
        latest_price = get_latest_close_price(df)
        forecast_day5 = get_forecast_price_after_n_days(df, window=5, days_ahead=5)

    elif model == 'ES':
        forecast_title = "Predicted Price 5 Days"
        es_html, es_errors = exponential_smoothing_forecast(df, symbol, alpha, forecast_days=steps)
        latest_price = get_latest_close_price(df)

        forecast_day5 = None
        if es_errors and isinstance(es_errors.get("Test"), dict):
            forecast_nday = es_errors["Test"].get("Forecast_NDay")
            if forecast_nday is not None and isinstance(forecast_nday, (int, float)) and not np.isnan(forecast_nday):
                forecast_day5 = round(forecast_nday, 2)


    elif model == 'ARIMA':
        forecast_title = f"Predicted Price After {steps} Days"
        arima_chart_file, arima_errors = arima_forecast(df, symbol, forecast_days=steps)
        latest_price = get_latest_close_price(df)

        forecast_day5 = None
        if arima_errors and isinstance(arima_errors.get("Test"), dict):
            forecast_nday = arima_errors["Test"].get("Forecast_NDay")
            if forecast_nday is not None and isinstance(forecast_nday, (int, float)) and not np.isnan(forecast_nday):
                forecast_day5 = round(forecast_nday, 2)



    elif model == 'HOLT':
        forecast_title = f"Predicted Price {steps} Days"
        holt_html, holt_errors = holt_forecast(df, symbol, forecast_days=steps, alpha=alpha, beta=beta)
        latest_price = get_latest_close_price(df)

        forecast_day5 = None
        if holt_errors and isinstance(holt_errors.get("Test"), dict):
            value = holt_errors["Test"].get("Forecast_NDay")
            if isinstance(value, (int, float)) and not np.isnan(value):
                forecast_day5 = round(value, 2)



    return render_template(
        'statistical_model.html',
        symbol=symbol,
        model=model,
        alpha=alpha,
        steps=steps,
        moving_averages_html=moving_averages_html,
        ma_errors=ma_errors,
        es_html=es_html,
        es_errors=es_errors,
        arima_chart_file=arima_chart_file,
        arima_errors=arima_errors,
        latest_price=latest_price,
        forecast_day5=forecast_day5,
        forecast_title=forecast_title,
        holt_html=holt_html,
        holt_errors=holt_errors,

    )



#Trang ML
@app.route('/ml', methods=['GET'])
def ml_entry():
    symbol = request.args.get('symbol', '').upper()
    if not symbol:
        return render_template('ml.html')  
    return redirect(url_for('ml_train', symbol=symbol))



@app.route('/ml/train', methods=['GET', 'POST'])
def ml_train():
    model_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
    available_models = [f.replace('_model.h5', '') for f in os.listdir(model_dir) if f.endswith('_model.h5')] if os.path.exists(model_dir) else []

    # Lấy lại các giá trị form (sau redirect)
    message = session.pop('message', None)
    loading = session.get('loading', False)
    symbol = session.get('symbol', None)
    form_model_type = session.pop('form_model_type', 'LSTM')
    form_symbol = session.pop('form_symbol', 'JPM')
    form_window = session.pop('form_window', 60)
    form_epochs = session.pop('form_epochs', 20)
    form_batch = session.pop('form_batch', 32)

    result = None
    chart_path = None

    if request.method == 'POST':
        symbol = request.form.get('symbol', 'JPM').upper()
        window_size = int(request.form.get('window_size', 60))
        epochs = int(request.form.get('epochs', 20))
        batch_size = int(request.form.get('batch_size', 32))
        model_type = request.form.get('model_type', 'lstm').lower()

        # Lưu lại form để pre-fill lại
        session['form_model_type'] = model_type.upper()
        session['form_symbol'] = symbol
        session['form_window'] = window_size
        session['form_epochs'] = epochs
        session['form_batch'] = batch_size

        file_path = os.path.join('dataset', f'{symbol}.csv')
        if not os.path.exists(file_path):
            session['message'] = f"❌ Không tìm thấy file dữ liệu: dataset/{symbol}.csv"
            return redirect(url_for('ml_train'))
        else:
            current_models = [f for f in os.listdir(model_dir) if f.endswith('_model.h5')] if os.path.exists(model_dir) else []
            session['models_before'] = current_models
            session['loading'] = True
            session['symbol'] = symbol
            session['message'] = f"⏳ Đang huấn luyện mô hình {model_type.upper()} cho {symbol}. Vui lòng chờ..."

            subprocess.Popen([
                sys.executable, 'train_worker.py',
                symbol, str(window_size), str(epochs), str(batch_size), model_type
            ])
            return redirect(url_for('ml_train'))

    # Kiểm tra kết quả huấn luyện
    if symbol:
        done_path = f"train_{symbol}_done.txt"
        if os.path.exists(done_path):
            models_before = set(session.get('models_before', []))
            models_after = set(f for f in os.listdir(model_dir) if f.endswith('_model.h5')) if os.path.exists(model_dir) else set()
            new_models = list(models_after - models_before)

            if new_models:
                message = f"✅ Đã huấn luyện xong mô hình: {', '.join(new_models)}"
            else:
                message = f"✅ Mô hình cho {symbol} đã được huấn luyện thành công!"

            session.pop('symbol', None)
        elif loading:
            message = f"⏳ Mô hình {symbol} đang được huấn luyện..."

    return render_template(
        'ml_train.html',
        available_models=available_models,
        message=message,
        symbol=symbol,
        loading=loading,
        result=result,
        chart_path=chart_path,
        form_model_type=form_model_type,
        form_symbol=form_symbol,
        form_window=form_window,
        form_epochs=form_epochs,
        form_batch=form_batch
    )




@app.route('/ml/analyze', methods=['GET', 'POST'])
def ml_analyze():
    model_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
    dataset_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dataset")

    # ✅ Chỉ lấy những model hợp lệ
    available_models = []
    for f in os.listdir(model_dir):
        if f.endswith('.h5') and ('lstm' in f or 'gru' in f):
            model_name = f.replace('.h5', '')
            parts = model_name.split('_')
            if len(parts) < 4:
                continue
            symbol, model_type, ws_part, ep_part = parts
            window = ws_part.replace('ws', '')
            epoch = ep_part.replace('ep', '')
            scaler_file = f"{symbol}_{model_type}_scaler_ws{window}_ep{epoch}.pkl"
            csv_file = f"{symbol}.csv"

            if (os.path.exists(os.path.join(model_dir, scaler_file)) and
                os.path.exists(os.path.join(dataset_dir, csv_file))):
                available_models.append(model_name)

    message = None
    result = None
    chart_html = None
    selected_symbol = None

    if request.method == 'POST':
        selected_symbol = request.form.get('trained_model', '')
        try:
            parts = selected_symbol.split('_')
            if len(parts) < 4:
                raise ValueError("❌ Tên model không hợp lệ.")

            symbol = parts[0].upper()
            model_type = parts[1].lower()
            window_size = int(parts[2].replace("ws", ""))
            epochs = int(parts[3].replace("ep", ""))

            model_path = os.path.join(model_dir, f"{symbol}_{model_type}_ws{window_size}_ep{epochs}.h5")
            scaler_path = os.path.join(model_dir, f"{symbol}_{model_type}_scaler_ws{window_size}_ep{epochs}.pkl")
            csv_path = os.path.join(dataset_dir, f"{symbol}.csv")

            # ✅ Kiểm tra tồn tại
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Không tìm thấy model tại: {model_path}")
            if not os.path.exists(scaler_path):
                raise FileNotFoundError(f"Không tìm thấy scaler tại: {scaler_path}")
            if not os.path.exists(csv_path):
                raise FileNotFoundError(f"Không tìm thấy dữ liệu tại: {csv_path}")

            # ✅ Gọi hàm phân tích
            from ml import analyze_trained_model
            result = analyze_trained_model(symbol, model_type=model_type, window_size=window_size, epochs=epochs)
            chart_html = result.get("chart_html")
            message = f"✅ Đã phân tích mô hình {selected_symbol} thành công."

        except Exception as e:
            message = f"❌ Lỗi khi phân tích: {e}"

    return render_template(
        'ml_analyze.html',
        available_models=available_models,
        message=message,
        result=result,
        chart_html=chart_html,
        selected_symbol=selected_symbol
    )




@app.route('/ml/predict', methods=['GET', 'POST'])
def ml_predict():
    model_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
    dataset_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dataset")

    available_models = [
        f.replace('.h5', '') for f in os.listdir(model_dir)
        if f.endswith('.h5') and ('lstm' in f or 'gru' in f)
    ]

    message = None
    chart_html = None
    forecast_rows = []
    has_future_forecast = False
    selected_model = None  
    

    if request.method == 'POST':
        selected_model = request.form.get('trained_model', '')
        forecast_days = int(request.form.get('days', 5))

        if not selected_model:
            message = "⚠️ Bạn cần chọn mô hình đã huấn luyện."
        else:
            try:
                parts = selected_model.split('_')
                if len(parts) < 4:
                    raise ValueError("❌ Tên mô hình không hợp lệ.")

                symbol = parts[0].upper()
                model_type = parts[1].lower()
                window_size = int(parts[2].replace("ws", ""))
                epochs = int(parts[3].replace("ep", ""))

                model_path = os.path.join(model_dir, f"{symbol}_{model_type}_ws{window_size}_ep{epochs}.h5")
                scaler_path = os.path.join(model_dir, f"{symbol}_{model_type}_scaler_ws{window_size}_ep{epochs}.pkl")
                csv_path = os.path.join(dataset_dir, f"{symbol}.csv")

                if not (os.path.exists(model_path) and os.path.exists(scaler_path) and os.path.exists(csv_path)):
                    message = f"❌ Không tìm thấy model, scaler hoặc dữ liệu cho {selected_model}"
                else:
                    df = pd.read_csv(csv_path)
                    df['Date'] = pd.to_datetime(df['Date'])
                    scaler = joblib.load(scaler_path)

                    forecast_df = predict_future(
                        df=df,
                        scaler=scaler,
                        window_size=window_size,
                        steps=forecast_days,
                        symbol=symbol,
                        model_type=model_type,
                        model_path=model_path
                    )

                    if forecast_df is not None and not forecast_df.empty:
                        has_future_forecast = True
                        forecast_rows = forecast_df.values.tolist()
                        chart_html = plot_forecast_chart(df, forecast_df)
                    else:
                        message = f"⚠️ Không thể dự báo với model {selected_model}"

            except Exception as e:
                message = f"❌ Lỗi khi dự đoán: {str(e)}"

    return render_template(
        'ml_predict.html',
        available_models=available_models,
        message=message,
        forecast_rows=forecast_rows,
        has_future_forecast=has_future_forecast,
        chart_html=chart_html,
        selected_model=selected_model,
        forecast_days=forecast_days if request.method == 'POST' else 5
    )



# Trang Model Evaluation
@app.route('/evaluation')
def evaluation():
    return render_template('evaluation.html')

@app.route('/evaluation/compare', methods=['GET', 'POST'])
def evaluation_compare():
    result_table = None
    chart_html = None
    error_metrics = None
    message = None
    selected_model = ''
    steps = 5

    # 1. Lấy danh sách mô hình hợp lệ (có cả .h5 và .pkl)
    model_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
    model_files = os.listdir(model_dir)
    available_models = []

    for f in model_files:
        if f.endswith(".h5") and ("lstm" in f or "gru" in f):
            base = f.replace(".h5", "")
            parts = base.split("_")
            if len(parts) >= 4:
                symbol = parts[0]
                model_type = parts[1]
                window_size = parts[2].replace("ws", "")
                epochs = parts[3].replace("ep", "")
                scaler_file = f"{symbol}_{model_type}_scaler_ws{window_size}_ep{epochs}.pkl"
                if scaler_file in model_files:
                    available_models.append(base)

    if request.method == 'POST':
        selected_model = request.form.get('trained_model', '')
        steps = int(request.form.get('steps', 5))

        try:
            parts = selected_model.split('_')
            if len(parts) < 4:
                raise ValueError("Tên mô hình không hợp lệ.")

            symbol = parts[0].upper()
            model_type = parts[1].lower()
            window_size = int(parts[2].replace("ws", ""))
            epochs = int(parts[3].replace("ep", ""))

            model_path = os.path.join("models", f"{symbol}_{model_type}_ws{window_size}_ep{epochs}.h5")
            scaler_path = os.path.join("models", f"{symbol}_{model_type}_scaler_ws{window_size}_ep{epochs}.pkl")
            data_path = os.path.join("dataset", f"{symbol}.csv")
            eval_path = os.path.join("evaluation", f"{symbol}.csv")

            if not os.path.exists(model_path) or not os.path.exists(scaler_path):
                message = "Không tìm thấy model hoặc scaler phù hợp."
            elif not os.path.exists(data_path):
                message = f"Không tìm thấy file dữ liệu: {data_path}"
            elif not os.path.exists(eval_path):
                message = f"Không tìm thấy file thực tế: {eval_path}"
            else:
                df_train = pd.read_csv(data_path)
                df_train['Date'] = pd.to_datetime(df_train['Date'])

                df_eval = pd.read_csv(eval_path)
                df_eval['Date'] = pd.to_datetime(df_eval['Date'])

                scaler = joblib.load(scaler_path)
                from ml import predict_future
                forecast_df = predict_future(
                    df=df_train,
                    scaler=scaler,
                    window_size=window_size,
                    steps=steps,
                    symbol=symbol,
                    model_type=model_type,
                    model_path=model_path
                )
                forecast_df['Date'] = pd.to_datetime(forecast_df['Date'])

                merged = pd.merge(
                    forecast_df.rename(columns={'Close': 'Predicted'}),
                    df_eval.rename(columns={'Close': 'Actual'}),
                    on='Date', how='inner'
                )
                result_table = merged[['Date', 'Predicted', 'Actual']]

                from sklearn.metrics import mean_absolute_error, mean_squared_error
                mae = mean_absolute_error(result_table['Actual'], result_table['Predicted'])
                rmse = np.sqrt(mean_squared_error(result_table['Actual'], result_table['Predicted']))
                mape = np.mean(np.abs((result_table['Actual'] - result_table['Predicted']) / result_table['Actual'])) * 100
                error_metrics = {'mae': round(mae, 4), 'rmse': round(rmse, 4), 'mape': round(mape, 2)}

                # Vẽ biểu đồ
                trace_actual = go.Scatter(x=result_table['Date'], y=result_table['Actual'], mode='lines+markers', name='Actual', line=dict(color='royalblue'))
                trace_pred = go.Scatter(x=result_table['Date'], y=result_table['Predicted'], mode='lines+markers', name='Predicted', line=dict(color='orangered', dash='dash'))
                layout = go.Layout(title=f'{symbol} - Forecast vs Actual', xaxis_title='Date', yaxis_title='Price', template='plotly_dark')
                fig = go.Figure([trace_actual, trace_pred], layout)
                chart_html = fig.to_html(full_html=False, include_plotlyjs='cdn')

        except Exception as e:
            message = f"Lỗi: {str(e)}"

    return render_template(
        'evaluation_compare.html',
        result_table=result_table,
        error_metrics=error_metrics,
        chart_html=chart_html,
        message=message,
        selected_model=selected_model,
        steps=steps,
        available_models=available_models
    )

# Route: /evaluation/model
@app.route('/evaluation/model', methods=['GET', 'POST'])
def evaluation_model():
    def get_model_group(model_type):
        if model_type.upper() in ['LSTM', 'GRU']:
            return 'DL'
        elif model_type.upper() in ['ARIMA', 'HOLT', 'ES'] or model_type.upper().startswith('MA'):
            return 'STAT'
        return 'UNKNOWN'

    model_a = request.form.get('model_a', '')
    model_b = request.form.get('model_b', '')

    error_dir = os.path.join(os.path.dirname(__file__), "errors")
    if not os.path.exists(error_dir):
        return render_template('evaluation_model.html', message="❌ Thư mục errors không tồn tại")

    pattern = re.compile(r"(\w+?)_([a-z0-9]+)_ws(\d+)_ep(\d+)_bs(\d+)\.csv")
    all_models = []
    selected_models = []
    selected_symbol = None

    for file in os.listdir(error_dir):
        match = pattern.match(file)
        if match:
            sym, model_type, ws, ep, bs = match.groups()
            model_name = f"{sym}_{model_type}_ws{ws}_ep{ep}_bs{bs}"

            model_entry = {
                'model_name': model_name,
                'symbol': sym.upper(),
                'model_type': model_type.upper() if not model_type.startswith('ma') else f"MA{ws}",
                'window_size': int(ws),
                'epochs': int(ep),
                'batch_size': int(bs),
                'file_path': os.path.join(error_dir, file)
            }

            all_models.append(model_entry)

            if model_name == model_a or model_name == model_b:
                try:
                    df = pd.read_csv(model_entry['file_path'])
                    if all(col in df.columns for col in ['MAE', 'RMSE', 'MAPE']):
                        model_entry.update({
                            'MAE': df.loc[0, 'MAE'],
                            'RMSE': df.loc[0, 'RMSE'],
                            'MAPE': df.loc[0, 'MAPE']
                        })
                        selected_models.append({
                            "model_name": model_entry['model_name'],
                            "mae": model_entry['MAE'],
                            "rmse": model_entry['RMSE'],
                            "mape": model_entry['MAPE'],
                            "chart_html": None
                        })
                        selected_symbol = sym.upper()
                except:
                    continue

    if len(selected_models) == 2:
        df_errors = pd.DataFrame(selected_models)

        bar_chart = go.Figure([go.Bar(
            x=df_errors['model_name'],
            y=df_errors['mape'],
            marker=dict(color=[
                'orange' if get_model_group(m['model_name']) == 'DL' else 'green' for m in selected_models
            ])
        )])
        bar_chart.update_layout(
            title='So sánh MAPE giữa hai mô hình',
            yaxis_title='MAPE (%)',
            template='plotly_dark',
            plot_bgcolor='#1f2937',
            paper_bgcolor='#1f2937',
            font=dict(color='white')
        )

        metric_chart = go.Figure()
        metrics = ['mae', 'rmse', 'mape']
        for m in selected_models:
            metric_chart.add_trace(go.Bar(
                name=m['model_name'],
                x=['MAE', 'RMSE', 'MAPE'],
                y=[m['mae'], m['rmse'], m['mape']]
            ))
        metric_chart.update_layout(
            barmode='group',
            title='So sánh lỗi giữa hai mô hình',
            yaxis_title='Giá trị lỗi',
            template='plotly_dark',
            plot_bgcolor='#1f2937',
            paper_bgcolor='#1f2937',
            font=dict(color='white')
        )

        return render_template(
            'evaluation_model.html',
            model_list=[m['model_name'] for m in all_models],
            model_a=model_a,
            model_b=model_b,
            model_metrics=selected_models,
            bar_chart=bar_chart.to_html(full_html=False, include_plotlyjs='cdn'),
            line_chart=None,
            metric_chart=metric_chart.to_html(full_html=False, include_plotlyjs=False),
            message=None
        )

    # ✅ return mặc định nếu chưa đủ 2 mô hình
    return render_template(
        'evaluation_model.html',
        model_list=[m['model_name'] for m in all_models],
        model_a=model_a,
        model_b=model_b,
        model_metrics=None,
        bar_chart=None,
        line_chart=None,
        metric_chart=None
    )



if __name__ == '__main__':
    app.run(debug=True)