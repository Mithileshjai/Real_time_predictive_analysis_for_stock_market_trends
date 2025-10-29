import numpy as np
import pandas as pd
import yfinance as yf
from keras.models import load_model
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, r2_score
import requests  # ✅ Added for fetching world news

st.set_page_config(page_title="Stock Predictor", page_icon="📈")

# --- LOGIN SYSTEM ---
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

def login():
    st.title("🔐 Login to Stock Market Predictor")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    if st.button("Login"):
        if username == "admin" and password == "12345":
            st.session_state.logged_in = True
            st.success("Login successful!")
            st.rerun()
        else:
            st.error("Invalid username or password")

if not st.session_state.logged_in:
    login()
    st.stop()

# --- PAGE NAVIGATION ---
if "page" not in st.session_state:
    st.session_state.page = "main"

model = load_model(r'C:\Users\Mithilesh J\OneDrive\Desktop\CLG\BDE\STOCK\Stock Predictions Model.keras')

# --- MAIN PAGE (Moving Averages + KPI Dashboard + News) ---
if st.session_state.page == "main":
    st.header('📈 Stock Market Predictor')

    stock = st.text_input('Enter Stock Symbol', 'GOOG')
    start = '2012-01-01'
    end = '2022-12-31'

    data = yf.download(stock, start, end)
    st.subheader('Stock Data')
    st.write(data)

    # --- Moving Average Charts ---
    ma_50_days = data.Close.rolling(50).mean()
    ma_100_days = data.Close.rolling(100).mean()
    ma_200_days = data.Close.rolling(200).mean()

    st.subheader('Price vs MA50')
    fig1 = plt.figure(figsize=(8,6))
    plt.plot(ma_50_days, 'r', label="MA50")
    plt.plot(data.Close, 'g', label="Close Price")
    plt.legend()
    st.pyplot(fig1)

    st.subheader('Price vs MA50 vs MA100')
    fig2 = plt.figure(figsize=(8,6))
    plt.plot(ma_50_days, 'r', label="MA50")
    plt.plot(ma_100_days, 'b', label="MA100")
    plt.plot(data.Close, 'g', label="Close Price")
    plt.legend()
    st.pyplot(fig2)

    st.subheader('Price vs MA100 vs MA200')
    fig3 = plt.figure(figsize=(8,6))
    plt.plot(ma_100_days, 'r', label="MA100")
    plt.plot(ma_200_days, 'b', label="MA200")
    plt.plot(data.Close, 'g', label="Close Price")
    plt.legend()
    st.pyplot(fig3)

    # --- 📊 KPI DASHBOARD SECTION ---
    st.markdown("### 📊 Stock Summary Dashboard")

    current_price = round(data['Close'].iloc[-1], 2)
    week_52_high = round(data['High'][-252:].max(), 2)
    week_52_low = round(data['Low'][-252:].min(), 2)
    volatility = round(data['Close'].pct_change().std() * 100, 2)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Current Price", f"${current_price}")
    col2.metric("52-Week High", f"${week_52_high}")
    col3.metric("52-Week Low", f"${week_52_low}")
    col4.metric("Volatility", f"{volatility}%")

    # --- 🌍 WORLD NEWS SECTION ---
    st.markdown("## 🌍 Latest World & Market News")

    try:
        url = "https://newsapi.org/v2/top-headlines"
        params = {
            "category": "business",
            "language": "en",
            "pageSize": 5,
            "apiKey": "381990b4bd1f4756b5886ec2d134ff1e"  # 🔑 Replace with your actual NewsAPI key
        }
        response = requests.get(url, params=params)
        data_news = response.json()

        if data_news.get("status") == "ok":
            for article in data_news["articles"]:
                st.markdown(f"### 📰 {article['title']}")
                if article.get("urlToImage"):
                    st.image(article["urlToImage"], use_container_width=True)
                st.write(article.get("description") or "")
                st.markdown(f"[Read more...]({article['url']})")
                st.markdown("---")
        else:
            st.warning("⚠ Unable to fetch news at this time. Try again later.")
    except Exception as e:
        st.error(f"Error loading news: {e}")

    # Button to go to prediction page
    if st.button("➡ View Prediction Chart"):
        st.session_state.page = "prediction"
        st.session_state.stock = stock
        st.rerun()

# --- PREDICTION PAGE ---
elif st.session_state.page == "prediction":
    st.header("📊 Original Price vs Predicted Price")

    stock = st.session_state.stock
    start = '2012-01-01'
    end = '2022-12-31'

    data = yf.download(stock, start, end)

    # Split data
    data_train = pd.DataFrame(data.Close[0:int(len(data)*0.80)])
    data_test = pd.DataFrame(data.Close[int(len(data)*0.80):])

    scaler = MinMaxScaler(feature_range=(0,1))
    past_100_days = data_train.tail(100)
    data_test = pd.concat([past_100_days, data_test], ignore_index=True)
    data_test_scale = scaler.fit_transform(data_test)

    # Prepare test data
    x = []
    y = []
    for i in range(100, data_test_scale.shape[0]):
        x.append(data_test_scale[i-100:i])
        y.append(data_test_scale[i,0])

    x, y = np.array(x), np.array(y)
    predict = model.predict(x)

    scale = 1 / scaler.scale_
    predict = predict * scale
    y = y * scale

    dates = data.index[int(len(data)*0.80):]

    # Plot Prediction Chart
    fig4 = plt.figure(figsize=(10,6))
    plt.plot(dates, y, 'r', label='Original Price')
    plt.plot(dates, predict, 'g', label='Predicted Price')
    plt.xlabel('Year')
    plt.ylabel('Price')
    plt.legend()
    plt.grid(True)
    plt.xticks(rotation=45)
    st.pyplot(fig4)

    # --- ACCURACY METRICS ---
    mae = mean_absolute_error(y, predict)
    mape = np.mean(np.abs((y - predict) / y)) * 100
    r2 = r2_score(y, predict)
    accuracy = 100 - mape

    st.subheader("📏 Model Accuracy Metrics")
    colA, colB, colC, colD = st.columns(4)
    colA.metric("R² Score", f"{r2:.4f}")
    colB.metric("MAE", f"{mae:.2f}")
    colC.metric("MAPE", f"{mape:.2f}%")
    colD.metric("Accuracy", f"{accuracy:.2f}%")

    # Back button
    if st.button("⬅ Back to Main Page"):
        st.session_state.page = "main"
        st.rerun()
