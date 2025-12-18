import numpy as np
import pandas as pd
import yfinance as yf
import streamlit as st
import matplotlib.pyplot as plt
import requests

from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, r2_score

# ---------------- USER AUTH CLASS ----------------
class UserAuth:
    def login_page(self):
        st.title("🔐 Login to Stock Market Predictor")
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")

        if st.button("Login"):
            if self.validate_user(username, password):
                st.session_state.logged_in = True
                st.success("Login Successful")
                st.rerun()
            else:
                st.error("Invalid Credentials")

    def validate_user(self, username, password):
        return username == "admin" and password == "12345"

# ---------------- STOCK DATA CLASS ----------------
class StockDataFetcher:
    def __init__(self, symbol, start, end):
        self.symbol = symbol
        self.start = start
        self.end = end

    def fetch_data(self):
        return yf.download(self.symbol, self.start, self.end)

# ---------------- VISUALIZATION CLASS ----------------
class StockVisualizer:
    def plot_moving_averages(self, data):
        ma50 = data.Close.rolling(50).mean()
        ma100 = data.Close.rolling(100).mean()
        ma200 = data.Close.rolling(200).mean()

        self._plot_chart(data.Close, ma50, "Price vs MA50")
        self._plot_chart(data.Close, ma50, "Price vs MA50 & MA100", ma100)
        self._plot_chart(data.Close, ma100, "Price vs MA100 & MA200", ma200)

    def _plot_chart(self, close, ma1, title, ma2=None):
        st.subheader(title)
        fig = plt.figure(figsize=(8,6))
        plt.plot(close, label="Close Price")
        plt.plot(ma1, label="MA")
        if ma2 is not None:
            plt.plot(ma2, label="MA2")
        plt.legend()
        st.pyplot(fig)

# ---------------- ML MODEL CLASS ----------------
class PredictionModel:
    def __init__(self, model_path):
        self.model = load_model(model_path)

    def predict(self, data):
        scaler = MinMaxScaler(feature_range=(0,1))
        data_train = data[:int(len(data)*0.80)]
        data_test = data[int(len(data)*0.80):]

        past_100 = data_train.tail(100)
        final_df = pd.concat([past_100, data_test], ignore_index=True)
        scaled_data = scaler.fit_transform(final_df)

        x, y = [], []
        for i in range(100, len(scaled_data)):
            x.append(scaled_data[i-100:i])
            y.append(scaled_data[i,0])

        x, y = np.array(x), np.array(y)
        predictions = self.model.predict(x)

        scale = 1 / scaler.scale_
        predictions = predictions * scale
        y = y * scale

        return y, predictions

# ---------------- NEWS CLASS ----------------
class NewsFetcher:
    def fetch_news(self):
        url = "https://newsapi.org/v2/top-headlines"
        params = {
            "category": "business",
            "language": "en",
            "pageSize": 5,
            "apiKey": "YOUR_NEWS_API_KEY"
        }
        response = requests.get(url, params=params)
        return response.json()

# ---------------- MAIN APP CLASS ----------------
class StockApp:
    def __init__(self):
        st.set_page_config(page_title="Stock Predictor", page_icon="📈")
        if "logged_in" not in st.session_state:
            st.session_state.logged_in = False
        if "page" not in st.session_state:
            st.session_state.page = "main"

        self.auth = UserAuth()
        self.model = PredictionModel("StockPredictionsModel.keras")
        self.news = NewsFetcher()

    def run(self):
        if not st.session_state.logged_in:
            self.auth.login_page()
            st.stop()

        if st.session_state.page == "main":
            self.main_page()
        else:
            self.prediction_page()

    def main_page(self):
        st.header("📈 Stock Market Predictor")
        stock = st.text_input("Enter Stock Symbol", "GOOG")

        data_fetcher = StockDataFetcher(stock, "2012-01-01", "2022-12-31")
        data = data_fetcher.fetch_data()

        st.subheader("Stock Data")
        st.write(data.tail())

        visualizer = StockVisualizer()
        visualizer.plot_moving_averages(data)

        st.markdown("### 🌍 Business News")
        news_data = self.news.fetch_news()
        if news_data.get("status") == "ok":
            for article in news_data["articles"]:
                st.markdown(f"**{article['title']}**")
                st.write(article.get("description", ""))

        if st.button("➡ View Prediction"):
            st.session_state.page = "prediction"
            st.session_state.stock = stock
            st.rerun()

    def prediction_page(self):
        st.header("📊 Prediction Result")
        stock = st.session_state.stock

        data_fetcher = StockDataFetcher(stock, "2012-01-01", "2022-12-31")
        data = data_fetcher.fetch_data()

        y, predict = self.model.predict(data.Close)

        fig = plt.figure(figsize=(10,6))
        plt.plot(y, label="Original Price")
        plt.plot(predict, label="Predicted Price")
        plt.legend()
        st.pyplot(fig)

        mae = mean_absolute_error(y, predict)
        r2 = r2_score(y, predict)

        st.metric("MAE", f"{mae:.2f}")
        st.metric("R² Score", f"{r2:.4f}")

        if st.button("⬅ Back"):
            st.session_state.page = "main"
            st.rerun()

# ---------------- RUN APP ----------------
app = StockApp()
app.run()
