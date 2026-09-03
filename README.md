# 📈 S&P 500 Stock Price Forecasting Using RNN

A time-series forecasting project that analyzes historical stock market data and predicts future stock prices using traditional forecasting methods and Recurrent Neural Network (RNN) models.

This project was developed as a final project for the **Applied Data Science** course.

---

## 📌 Project Overview

The project focuses on analyzing historical stock price data from companies in the **S&P 500** and developing forecasting models to predict future closing prices.

The system combines:

- Exploratory Data Analysis
- Descriptive Statistical Analysis
- Time-Series Analysis
- Technical Analysis
- Traditional Forecasting Models
- Recurrent Neural Networks
- Model Evaluation and Comparison
- Interactive Web Visualization

The final system is implemented as a **Flask web application**, allowing users to enter a stock ticker and explore statistical analysis, time-series analysis, forecasting results, and RNN predictions.

---

## 🎯 Objectives

- Analyze historical stock market data.
- Explore the statistical characteristics of stock prices and trading volume.
- Analyze stock price behavior using time-series techniques and technical indicators.
- Apply traditional forecasting methods to stock price data.
- Build RNN-based forecasting models using **LSTM** and **GRU**.
- Evaluate and compare different forecasting models.
- Build an interactive web application for stock analysis and price forecasting.

---

## 🔄 System Workflow

```text
Historical Stock Data
        ↓
Data Collection
        ↓
Data Cleaning & Preprocessing
        ↓
Descriptive Statistical Analysis
        ↓
Time-Series & Technical Analysis
        ↓
Traditional Forecasting Models
        ↓
RNN Model Development
        ↓
LSTM / GRU Training
        ↓
Model Evaluation
        ↓
Model Comparison
        ↓
Future Stock Price Forecast
        ↓
Interactive Flask Web Application
```

---

## 📊 Data Analysis

Before building forecasting models, the project performs descriptive and exploratory analysis of historical stock data.

### Descriptive Statistics

The system analyzes:

- Open Price
- High Price
- Low Price
- Close Price
- Trading Volume

Statistical measures include:

- Mean
- Standard Deviation
- Minimum
- Maximum
- Mode
- Variance
- Skewness
- Kurtosis
- Range
- 95% Confidence Interval

### Visualizations

The application provides several visualizations:

- Box Plot
- Correlation Heatmap
- Close Price Histogram
- Trading Volume Histogram
- Price & Volume Chart
- Candlestick Chart

---

## 📈 Time-Series Analysis

The project analyzes stock price behavior using time-series and technical analysis.

Implemented analyses include:

- Monthly Total Closing Price
- Monthly Average Closing Price
- Monthly Traded Value
- RSI (Relative Strength Index)
- Seasonality Analysis
- Time-Series Decomposition
- Price and Trading Volume Movements

These analyses are used to explore trends, seasonality, and relationships within historical stock data.

---

## 📉 Traditional Forecasting Models

Several traditional time-series forecasting methods are implemented for comparison with RNN models.

### Moving Average (MA)

Uses historical observations within a specified window to estimate future prices.

### Exponential Smoothing (ES)

Assigns exponentially decreasing weights to past observations, giving more importance to recent values.

### ARIMA

Uses autoregressive, differencing, and moving-average components for time-series forecasting.

### Holt Linear Trend

Models both the level and trend of the time series to forecast future values.

---

## 🧠 RNN Models

The main machine learning component uses **Recurrent Neural Networks** for stock price forecasting.

Two architectures are implemented:

### LSTM – Long Short-Term Memory

The implemented LSTM architecture is:

```text
Input Sequence
      ↓
Bidirectional LSTM (128 units)
      ↓
Dropout (0.2)
      ↓
Bidirectional LSTM (64 units)
      ↓
Dropout (0.2)
      ↓
Dense Output
      ↓
Predicted Close Price
```

The model uses:

- Adam optimizer
- Mean Squared Error (MSE) loss
- Early stopping

### GRU – Gated Recurrent Unit

The implemented GRU architecture is:

```text
Input Sequence
      ↓
GRU (64 units)
      ↓
Dropout (0.2)
      ↓
Dense Output
      ↓
Predicted Close Price
```

The GRU model also uses historical closing prices as the main prediction variable.

---

## 🔧 Data Preprocessing

The RNN pipeline includes the following steps:

### 1. Data Sorting

Historical observations are sorted chronologically by date.

### 2. Feature Scaling

The closing price is normalized using:

```text
MinMaxScaler
```

### 3. Sliding Window

Historical observations are transformed into sequential input windows.

For example:

```text
[t-60, ..., t-2, t-1] → Predict t
```

The window size can be configured during model training.

### 4. Train/Test Split

The dataset is divided into training and testing sets for model development and evaluation.

---

## 📏 Model Evaluation

Forecasting performance is evaluated using:

### MAE – Mean Absolute Error

Measures the average absolute difference between actual and predicted prices.

### RMSE – Root Mean Squared Error

Measures the square root of the average squared prediction error and gives greater weight to larger errors.

### MAPE – Mean Absolute Percentage Error

Measures prediction error as a percentage of the actual value.

The models are compared using:

- MAE
- RMSE
- MAPE

Models include:

- Moving Average
- Exponential Smoothing
- ARIMA
- Holt
- LSTM
- GRU

---

## 🧪 Model Experiments

The project evaluates different model configurations by changing:

- Window Size
- Epochs
- Batch Size

Multiple configurations are tested for both **LSTM** and **GRU**, followed by performance evaluation and comparison.

The models are also tested on different stock symbols, including:

- JPM
- MA
- BAC
- GS
- V

---

## 🔮 Future Price Forecasting

After training and evaluating the models, the system generates future stock price predictions.

The project includes forecasting for the **next 10 trading days** using trained RNN models.

The results are compared between:

- LSTM
- GRU
- Traditional Forecasting Models

This comparison helps evaluate the differences between traditional time-series forecasting and RNN-based forecasting.

---

## 🌐 Web Application

The project provides an interactive web application built with **Flask**.

Users can enter a stock ticker and access different analysis modules.

### 📊 Descriptive Statistics

Provides statistical summaries and distribution visualizations.

### 📈 Time-Series Analysis

Provides:

- Price & Volume Analysis
- Candlestick Charts
- Monthly Analysis
- RSI
- Seasonality
- Time-Series Decomposition

### 📉 Statistical Forecasting

Users can select:

- Moving Average
- Exponential Smoothing
- ARIMA
- Holt Linear Trend

### 🧠 Machine Learning

Users can:

- Select a stock symbol
- Select LSTM or GRU
- Configure window size
- Configure epochs
- Configure batch size
- Train the model
- View prediction results
- Evaluate model performance

---

## 🛠️ Technologies

### Programming Language

- Python

### Web Framework

- Flask

### Data Processing

- Pandas
- NumPy

### Data Visualization

- Plotly

### Statistical Analysis

- SciPy
- Statsmodels

### Machine Learning

- Scikit-learn
- TensorFlow / Keras

### Financial Data

- Yahoo Finance (`yfinance`)

### Model Serialization

- Joblib

### Deployment

- Gunicorn
- Docker
- Render

---

## 📁 Project Structure

```text
├── app.py
├── functions.py
├── ml.py
├── train_worker.py
├── requirements.txt
├── Dockerfile
├── render.yaml
├── .gitignore
├── .gitattributes
│
├── dataset/
│   ├── JPM.csv
│   ├── MA.csv
│   ├── BAC.csv
│   ├── GS.csv
│   ├── V.csv
│   └── ...
│
├── models/
│   ├── *_lstm_*.h5
│   ├── *_gru_*.h5
│   └── ...
│
├── errors/
│   └── model evaluation results
│
├── templates/
│   ├── index.html
│   ├── ticker.html
│   ├── statistics.html
│   ├── timeseries.html
│   ├── statistical_model.html
│   ├── ml.html
│   └── ...
│
└── static/
    └── generated charts and static assets
```

---

## 📄 Main Files

| File | Description |
|------|-------------|
| `app.py` | Flask application, routing, and web interface logic |
| `functions.py` | Data processing, statistical analysis, visualization, and traditional forecasting |
| `ml.py` | LSTM and GRU model training, prediction, and evaluation |
| `train_worker.py` | Background process for training RNN models |
| `requirements.txt` | Python package dependencies |
| `Dockerfile` | Docker configuration |
| `render.yaml` | Render deployment configuration |

---

## 🚀 Installation & Setup

### 1. Clone the Repository

```bash
git clone <repository-url>
cd <project-folder>
```

### 2. Create a Virtual Environment

```bash
python -m venv venv
```

### 3. Activate the Environment

**Windows:**

```bash
venv\Scripts\activate
```

**macOS / Linux:**

```bash
source venv/bin/activate
```

### 4. Install Dependencies

```bash
pip install -r requirements.txt
```

### 5. Run the Flask Application

```bash
python app.py
```

Then open the local URL provided by Flask in your browser.

---

## 🐳 Docker

The project includes a `Dockerfile` for containerized deployment.

### Build the Docker Image

```bash
docker build -t stock-forecasting .
```

### Run the Container

```bash
docker run -p 5000:5000 stock-forecasting
```

---

## ☁️ Deployment

The project includes a `render.yaml` configuration for deployment on **Render**.

The application runs using Gunicorn:

```bash
gunicorn app:app --bind 0.0.0.0:$PORT
```

---

## 📚 Project Documentation

The accompanying project report covers:

- Introduction and research objectives
- System workflow
- Data sources and preprocessing
- Descriptive statistics
- Time-series and technical analysis
- Traditional forecasting methods
- RNN model development
- LSTM experiments
- GRU experiments
- Real-world validation
- Model comparison
- Conclusion and future directions

---

## 🎓 Academic Information

**Course:** Applied Data Science

**Project:** S&P 500 Stock Price Forecasting Using RNN

**University:** University of Social Sciences and Humanities, VNU-HCM

**Author:** Nguyễn Mai Xuân Linh

**Student ID:** 2256210028

---

## ⚠️ Disclaimer

This project is developed for **academic and research purposes only**.

The predictions generated by the models should not be considered financial advice or a recommendation to buy or sell stocks.

---

## 📸 Application Screenshots

<img width="1883" height="874" alt="Application Screenshot 1" src="https://github.com/user-attachments/assets/cfe2185f-101a-48f0-b373-23e527240e08" />

<img width="1899" height="904" alt="Application Screenshot 2" src="https://github.com/user-attachments/assets/12fa5a2c-6377-4611-8b8d-aa9067b3e039" />

<img width="1904" height="899" alt="Application Screenshot 3" src="https://github.com/user-attachments/assets/82bd5e6d-334c-4388-8619-ec462b8c6d23" />

<img width="1919" height="897" alt="Application Screenshot 4" src="https://github.com/user-attachments/assets/debd1f27-9de7-4009-9f1a-2f11646a2e09" />

<img width="1920" height="898" alt="Application Screenshot 5" src="https://github.com/user-attachments/assets/3afd5ce3-828b-4ece-b872-cc0d9be32702" />

<img width="1895" height="898" alt="Application Screenshot 6" src="https://github.com/user-attachments/assets/21889d42-edb0-4b8f-b709-2a114be013f6" />

<img width="1887" height="884" alt="Application Screenshot 7" src="https://github.com/user-attachments/assets/bc55c9cd-1cfd-4428-a6b4-a815fd1632a3" />

<img width="1918" height="907" alt="Application Screenshot 8" src="https://github.com/user-attachments/assets/5f9cdc58-cea8-46d6-a359-ee79bb6f9315" />

<img width="1920" height="908" alt="Application Screenshot 9" src="https://github.com/user-attachments/assets/a30a246e-f966-4232-bac5-8b7df1c71c29" />

<img width="1880" height="897" alt="Application Screenshot 10" src="https://github.com/user-attachments/assets/b01335ee-e8c7-4445-8f23-92d007d1eb1e" />
