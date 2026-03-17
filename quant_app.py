import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from ta.momentum import RSIIndicator
from sklearn.linear_model import LinearRegression
import plotly.graph_objects as go # Thư viện vẽ biểu đồ chuyên nghiệp hơn
import warnings

warnings.filterwarnings("ignore")

st.set_page_config(page_title="AI Hedge Fund Dashboard", layout="wide")
st.title("🛡️ AI Quant System - Châu's Trading Terminal")

# 1. Sidebar
st.sidebar.header("Cấu hình hệ thống")
tickers = st.sidebar.multiselect("Chọn danh sách cổ phiếu:", 
                                 ['GEE.VN', 'HPG.VN', 'FPT.VN', 'VNM.VN', 'VCB.VN', 'VIC.VN'],
                                 default=['GEE.VN', 'HPG.VN', 'FPT.VN'])
days_to_predict = st.sidebar.slider("Số ngày dự báo tương lai (AI):", 1, 15, 5)

@st.cache_data
def load_data(ticker_list):
    all_tickers = ticker_list + ['^VNINDEX']
    df = yf.download(all_tickers, start="2024-01-01", auto_adjust=True)['Close']
    return df

if tickers:
    prices = load_data(tickers)
    vni = prices['^VNINDEX']
    stocks = prices[tickers]

    # 2. AI Model & Visualization
    st.subheader("🤖 AI Trend Prediction & Interactive Chart")
    target_stock = st.selectbox("Chọn mã để xem dự báo chi tiết:", tickers)
    
    # Huấn luyện AI
    df_target = stocks[target_stock].dropna()
    y = df_target.values.reshape(-1, 1)
    X = np.arange(len(y)).reshape(-1, 1)
    model = LinearRegression().fit(X, y)
    
    # Dự báo
    future_X = np.arange(len(y), len(y) + days_to_predict).reshape(-1, 1)
    forecast = model.predict(future_X)
    
    # Tạo Index cho tương lai (tạm tính theo ngày tiếp theo)
    last_date = df_target.index[-1]
    future_dates = pd.date_range(last_date + pd.Timedelta(days=1), periods=days_to_predict)

    # VẼ BIỂU ĐỒ CHUYÊN NGHIỆP VỚI PLOTLY
    fig = go.Figure()
    # Đường giá thực tế
    fig.add_trace(go.Scatter(x=df_target.index, y=df_target.values.flatten(), name='Giá thực tế', line=dict(color='#2E86C1', width=2)))
    # Đường AI dự báo
    fig.add_trace(go.Scatter(x=future_dates, y=forecast.flatten(), name='AI Dự báo', line=dict(color='#E74C3C', width=3, dash='dot')))
    
    # VẼ BIỂU ĐỒ CHUYÊN NGHIỆP - DARK MODE
    fig = go.Figure()
    
    # 1. Đường giá thực tế (Màu xanh dương đậm của sự tin cậy)
    fig.add_trace(go.Scatter(
        x=df_target.index, 
        y=df_target.values.flatten(), 
        name='Giá lịch sử', 
        line=dict(color='#00D4FF', width=2)
    ))
    
    # 2. Đường AI dự báo (Màu cam sáng để gây chú ý)
    fig.add_trace(go.Scatter(
        x=future_dates, 
        y=forecast.flatten(), 
        name='AI Dự báo', 
        line=dict(color='#FF9F0A', width=3, dash='dot')
    ))
    
    # Cấu hình giao diện Dark Mode cực ngầu
    fig.update_layout(
        template='plotly_dark', # Chuyển sang nền tối
        hovermode='x unified',
        height=600,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        xaxis=dict(showgrid=False),
        yaxis=dict(showgrid=True, gridcolor='#333333'),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    st.plotly_chart(fig, use_container_width=True)

    # 3. Market Filter & Metrics
    col1, col2 = st.columns(2)
    current_price = float(y[-1].item())
    predicted_price = float(forecast[-1].item())
    
    with col1:
        st.metric(f"Giá dự báo sau {days_to_predict} ngày", f"{predicted_price:,.0f} VNĐ", f"{predicted_price - current_price:,.0f} VNĐ")
    
    with col2:
        vni_sma = vni.rolling(50).mean().iloc[-1]
        if vni.iloc[-1] > vni_sma:
            st.success("✅ THỊ TRƯỜNG AN TOÀN (VNI > SMA50)")
        else:
            st.warning("⚠️ THỊ TRƯỜNG RỦI RO (VNI < SMA50)")

    # 4. Bảng xếp hạng
    st.subheader("🏆 Top Stocks Ranking")
    # ... (Phần code bảng xếp hạng giữ nguyên như bản trước)
    scores = []
    for t in tickers:
        series = stocks[t].dropna()
        if len(series) > 20:
            rsi_val = RSIIndicator(series).rsi().iloc[-1]
            mom_val = series.pct_change(20).iloc[-1]
            score_val = (50 - abs(rsi_val - 50)) + (mom_val * 100)
            scores.append({"Mã": t, "Giá": f"{series.iloc[-1]:,.0f}", "RSI": round(rsi_val, 1), "Momentum": f"{mom_val*100:.1f}%", "Điểm": round(score_val, 2)})
    st.table(pd.DataFrame(scores).sort_values("Điểm", ascending=False))

else:
    st.info("👈 Hãy chọn cổ phiếu.")