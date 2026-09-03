# 📈 S&P 500 Stock Price Forecasting Using RNN

A time-series forecasting project that analyzes historical stock market data and predicts future stock prices using traditional forecasting methods and Recurrent Neural Network (RNN) models.

This project was developed as a final project for the **Applied Data Science** course at the Department of Library and Information Science, University of Social Sciences and Humanities, VNU-HCM.

---

## 📌 Project Overview

The project focuses on analyzing stock price data from companies in the **S&P 500** and developing forecasting models to predict future closing prices.

The system combines:

- Exploratory Data Analysis
- Statistical analysis
- Time-series analysis
- Technical indicators
- Traditional forecasting methods
- Recurrent Neural Networks
- Model evaluation and comparison
- Interactive web visualization

The final system is implemented as a **Flask web application**, allowing users to enter a stock ticker and explore historical data, statistical analysis, forecasting results, and RNN predictions.

---

## 🎯 Objectives

The main objectives of this project are:

- Analyze historical stock market data.
- Explore the statistical characteristics of stock prices and trading volume.
- Analyze stock price behavior using time-series techniques and technical indicators.
- Apply traditional forecasting methods to stock price data.
- Build RNN-based forecasting models using **LSTM** and **GRU**.
- Evaluate and compare the performance of different forecasting models.
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
📊 Data Analysis

Before building forecasting models, the project performs descriptive and exploratory analysis of historical stock data.

Descriptive Statistics

The system analyzes:

Open Price
High Price
Low Price
Close Price
Trading Volume

Statistical measures include:

Mean
Standard deviation
Minimum
Maximum
Mode
Variance
Skewness
Kurtosis
Range
95% confidence interval
Visualizations

The application provides several interactive visualizations:

Box Plot
Correlation Heatmap
Close Price Histogram
Trading Volume Histogram
Price & Volume Chart
Candlestick Chart
📈 Time-Series Analysis

The project further analyzes stock price behavior using time-series and technical analysis.

Implemented analyses include:

Monthly total closing price
Monthly average closing price
Monthly traded value
RSI (Relative Strength Index)
Seasonality analysis
Time-series decomposition
Price and trading volume movements

These analyses help identify patterns, trends, seasonality, and relationships within historical stock data.

📉 Traditional Forecasting Models

Before applying RNN models, the project evaluates several traditional time-series forecasting methods.

1. Moving Average (MA)

Uses historical observations within a specified window to estimate future prices.

2. Exponential Smoothing (ES)

Applies exponentially decreasing weights to past observations, giving more importance to recent values.

3. ARIMA

Uses autoregressive, differencing, and moving-average components for time-series forecasting.

4. Holt Linear Trend

Models both the level and trend of the time series to forecast future values.

The performance of these models is evaluated and compared with the RNN-based approaches.

🧠 RNN Models

The main machine learning component of the project uses Recurrent Neural Networks for stock price forecasting.

Two architectures are implemented:

LSTM – Long Short-Term Memory

The LSTM model is designed to capture dependencies in sequential stock price data.

The implemented architecture includes:

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

The model uses the Adam optimizer and Mean Squared Error as the loss function.

GRU – Gated Recurrent Unit

The GRU model provides another recurrent architecture for sequential forecasting.

Input Sequence
      ↓
GRU (64 units)
      ↓
Dropout (0.2)
      ↓
Dense Output
      ↓
Predicted Close Price

Both models use historical closing prices as the main prediction variable.

🔧 Data Preprocessing for RNN

The RNN pipeline includes several preprocessing steps:

1. Data Sorting

Historical observations are sorted chronologically by date.

2. Feature Scaling

The closing price is normalized using:

MinMaxScaler
3. Sliding Window

Historical observations are transformed into sequential input windows.

For example:

[t-60, ..., t-2, t-1] → Predict t

The window size can be configured when training the model.

4. Train/Test Split

The data is divided into training and testing sets before model evaluation.

📏 Model Evaluation

Forecasting performance is evaluated using three main metrics:

MAE – Mean Absolute Error

Measures the average absolute difference between actual and predicted prices.

RMSE – Root Mean Squared Error

Measures the square root of the average squared prediction error and gives greater weight to larger errors.

MAPE – Mean Absolute Percentage Error

Measures prediction error as a percentage of the actual value.

MAE
RMSE
MAPE

These metrics are used to compare:

Moving Average
Exponential Smoothing
ARIMA
Holt
LSTM
GRU
🧪 Model Experiments

The project evaluates different model configurations by changing parameters such as:

Window Size
Epochs
Batch Size

The report includes experiments with multiple configurations for both LSTM and GRU, followed by evaluation and comparison of their forecasting performance.

The models are also tested on multiple stock symbols, including examples such as:

JPM
MA
BAC
GS
V
🔮 Future Price Forecasting

After training and evaluating the models, the system can generate future stock price predictions.

The project includes forecasting for the next 10 trading days using the trained RNN models.

The results are compared between:

LSTM
GRU
Traditional forecasting methods

This allows the project to examine differences in forecasting performance between machine learning and traditional time-series approaches.

🌐 Web Application

The project provides an interactive web application built with Flask.

Users can enter a stock ticker and access different analysis modules.

Main Modules
📊 Descriptive Statistics

Provides statistical summaries and distribution visualizations.

📈 Time-Series Analysis

Provides:

Price & volume analysis
Candlestick charts
Monthly analysis
RSI
Seasonality
Decomposition
📉 Statistical Forecasting

Allows users to select traditional forecasting methods:

Moving Average
Exponential Smoothing
ARIMA
Holt Linear Trend
🧠 Machine Learning

Allows users to:

Select a stock symbol
Select LSTM or GRU
Configure window size
Configure epochs
Configure batch size
Train the model
View prediction results
Evaluate model performance
🛠️ Technologies
Programming Language
Python
Web Framework
Flask
Data Processing
Pandas
NumPy
Data Visualization
Plotly
Statistical Analysis
SciPy
Statsmodels
Machine Learning
Scikit-learn
TensorFlow / Keras
Financial Data
Yahoo Finance (yfinance)
Model & Object Serialization
Joblib
Deployment
Gunicorn
Docker
Render
📁 Project Structure
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
📄 Main Files
File	Description
app.py	Flask application, routing and web interface logic
functions.py	Data processing, statistical analysis, visualization and traditional forecasting
ml.py	LSTM and GRU model training, prediction and evaluation
train_worker.py	Background process for training RNN models
requirements.txt	Python package dependencies
Dockerfile	Docker configuration
render.yaml	Render deployment configuration
🚀 Installation & Setup
1. Clone the repository
git clone <repository-url>
cd <project-folder>
2. Create a virtual environment
python -m venv venv

Activate the environment.

Windows:

venv\Scripts\activate

macOS / Linux:

source venv/bin/activate
3. Install dependencies
pip install -r requirements.txt
4. Run the Flask application
python app.py

Open the local URL provided by Flask in your browser.

🐳 Docker

The project also includes a Dockerfile for containerized deployment.

Build the Docker image:

docker build -t stock-forecasting .

Run the container:

docker run -p 5000:5000 stock-forecasting
☁️ Deployment

The project includes a render.yaml configuration for deployment on Render.

The application runs using Gunicorn:

gunicorn app:app --bind 0.0.0.0:$PORT
📚 Project Documentation

The accompanying report covers:

Introduction and research objectives
System workflow
Data sources and preprocessing
Descriptive statistics
Time-series and technical analysis
Traditional forecasting methods
RNN model development
LSTM experiments
GRU experiments
Real-world validation
Model comparison
Conclusions and future directions
🎓 Academic Information

Course: Applied Data Science
Project: S&P 500 Stock Price Forecasting Using RNN

⚠️ Disclaimer

This project is developed for academic and research purposes only.

The predictions generated by the models should not be considered financial advice or a recommendation to buy or sell stocks.

<img width="1883" height="874" alt="image" src="https://github.com/user-attachments/assets/cfe2185f-101a-48f0-b373-23e527240e08" />

<img width="1899" height="904" alt="image" src="https://github.com/user-attachments/assets/12fa5a2c-6377-4611-8b8d-aa9067b3e039" />

<img width="1904" height="899" alt="image" src="https://github.com/user-attachments/assets/82bd5e6d-334c-4381-8619-ec462b8c6d23" />

<img width="1919" height="897" alt="image" src="https://github.com/user-attachments/assets/debd1f27-9de7-4009-9f1a-2f11646a2e09" />

<img width="1920" height="898" alt="image" src="https://github.com/user-attachments/assets/3afd5ce3-828b-4ece-b872-cc0d9be32702" />

<img width="1895" height="898" alt="image" src="https://github.com/user-attachments/assets/21889d42-edb0-4b8f-b709-2a114be013f6" />

<img width="1887" height="884" alt="image" src="https://github.com/user-attachments/assets/bc55c9cd-1cfd-4428-a6b4-a815fd1632a3" />

<img width="1918" height="907" alt="image" src="https://github.com/user-attachments/assets/5f9cdc58-cea8-46d6-a359-ee79bb6f9315" />

<img width="1920" height="908" alt="image" src="https://github.com/user-attachments/assets/a30a246e-f966-4232-bac5-8b7df1c71c29" />

<img width="1880" height="897" alt="image" src="https://github.com/user-attachments/assets/b01335ee-e8c7-4445-8f23-92d007d1eb1e" />


