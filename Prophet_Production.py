import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from prophet import Prophet
from scipy.stats import norm
# ------------------ ฟังก์ชันหลัก ------------------
def forecasting_fn(df, plant, coin):
    name = f"{plant}: {coin}"
    df_filtered = df[df['PLANTNAME'] == plant][['date', coin]]
    df_filtered.columns = ['ds', 'y']
    df_filtered['ds'] = pd.to_datetime(df_filtered['ds'])

    model = Prophet(changepoint_prior_scale=0.2)
    model.add_seasonality(name='Quarterly', period=91.25, fourier_order=2)
    model.fit(df_filtered)

    future = model.make_future_dataframe(periods=12, freq='ME')
    forecast = model.predict(future)

    df_filtered.reset_index(inplace=True)
    return model, forecast, future, name, df_filtered

def plot_forecast_plotly(name, df, forecast, fiscal_year=None, bound_margin=0):
    if fiscal_year:
        year_ad = fiscal_year - 543
        start_date = pd.to_datetime(f"{year_ad - 1}-10-01")
        end_date = pd.to_datetime(f"{year_ad}-09-30")
        df = df[(df['ds'] >= start_date) & (df['ds'] <= end_date)]
        forecast = forecast[(forecast['ds'] >= start_date) & (forecast['ds'] <= end_date)]

    upper_bound = forecast['yhat'] + bound_margin
    lower_bound = forecast['yhat'] - bound_margin

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df['ds'], y=df['y'], mode='markers', name='Actual', marker=dict(color='rgba(137, 196, 244, 0.9)')))
    fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat'], mode='lines', name='Forecast', line=dict(color='rgba(255, 99, 132, 0.9)')))
    fig.add_trace(go.Scatter(x=forecast['ds'], y=upper_bound, mode='lines', name='Upper Bound', line=dict(width=0), showlegend=False))
    fig.add_trace(go.Scatter(x=forecast['ds'], y=lower_bound, mode='lines', name='Forecast Range', fill='tonexty', line=dict(width=0), fillcolor='rgba(255, 99, 132, 0.2)', showlegend=True))
    fig.update_layout(title=f'{name} Forecasting (Prophet)', xaxis_title='Month', yaxis_title='Coins', width=1000, height=500)
    st.plotly_chart(fig, use_container_width=True)

# ------------------ Streamlit App ------------------
st.set_page_config(page_title="Forecasting Coins", layout="wide")
st.title("🔮 Owl Mint Forecast Dashboard")

data = pd.read_excel("Data_Monthly.xlsx", index_col=0)
df = data.copy()
df.rename(columns={'Fiscal_Year': 'FiscalYear'}, inplace=True)

coin_options = ['รวม', '0.25', '0.5', '1.0', '2.0', '5.0', '10.0']
center_options = df['PLANTNAME'].unique().tolist()
fiscal_years = sorted(df['FiscalYear'].unique().tolist())

col1, col2, col3 = st.columns(3)
with col1:
    selected_center = st.selectbox("เลือกศูนย์ (Center)", center_options)
with col2:
    selected_coin = st.selectbox("เลือกเหรียญ (Coin)", coin_options)
    coin_column = 'รวม'
with col3:
    year_options = ["ทั้งหมด"] + fiscal_years[::-1]
    selected_year = st.selectbox("เลือกปีงบประมาณ (เพื่อดูกราฟเฉพาะช่วงปี)", year_options)

# 🔧 ปรับระดับการให้บริการ (Service Level)
service_level = st.slider("ระดับการให้บริการ (Service Level %)", min_value=50, max_value=99, value=80)
z = norm.ppf(service_level / 100) # ปรับปรุงการคำนวณ Z-score 

model, forecast, future, name, df_filtered = forecasting_fn(df, plant=selected_center, coin=selected_coin)

# กำหนดหน่วยเหรียญ
if selected_coin == 'รวม':
    coin_unit = 'บาท'
elif float(selected_coin) < 1:
    coin_unit = 'สตางค์'
else:
    coin_unit = 'บาท'

# คำนวณค่าคลาดเคลื่อนและ Bound

merged = pd.merge(df_filtered, forecast[['ds', 'yhat']], on='ds', how='inner') # ปรับปรุงเพื่อป้องกันการ mismatch
errors = merged['y'] - merged['yhat']

std_error = np.std(errors)
lead_time = 1  # เดือน
safety_stock = z * std_error * np.sqrt(lead_time)
mean_forecast = forecast['yhat'].mean()
total_required = mean_forecast + safety_stock

# คำนวณแบบรายปี (12 เดือน)
mean_forecast_year = mean_forecast * 12
safety_stock_year = safety_stock * 12
total_required_year = total_required * 12

# คำนวณระดับการให้บริการจริงจากข้อมูลย้อนหลัง
service_level_empirical = np.mean(df_filtered['y'] <= forecast['yhat'].iloc[:len(df_filtered)]) * 100

st.subheader(f"📊 ผลการทำนายเหรียญ {selected_coin} {coin_unit if selected_coin != 'รวม' else '' } @ {selected_center}")
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("ค่าเฉลี่ยที่ควรมีต่อเดือน", f"{mean_forecast:,.2f}")
with col2:
    st.metric(f"Safety Stock ต่อเดือน (ที่ระดับการให้บริการ: {service_level}%)", f"{safety_stock:,.2f}")
with col3:
    st.metric("ขั้นต่ำต่อเดือนที่ควรมี", f"{total_required:,.2f}")

col4, col5, col6 = st.columns(3)
with col4:
    st.metric("ค่าเฉลี่ยที่ควรมีต่อปี", f"{mean_forecast_year:,.2f}")
with col5:
    st.metric(f"Safety Stock ต่อปี (ที่ระดับการให้บริการ:{service_level}%)", f"{safety_stock_year:,.2f}")
with col6:
    st.metric("ขั้นต่ำต่อปีที่ควรมี", f"{total_required_year:,.2f}")

# แสดงระดับการให้บริการจริง
st.info(f"🔍 ระดับการให้บริการย้อนหลังจริง (Empirical Service Level): {service_level_empirical:.2f}%")

# แสดงกราฟตามปีงบประมาณที่เลือก
plot_forecast_plotly(name, df_filtered, forecast, fiscal_year=None if selected_year == "ทั้งหมด" else selected_year, bound_margin=safety_stock)

#st.video("https://youtu.be/3KalfTj3xDw")

# 🔍 ฟังก์ชันดูทั้งหมด
if st.checkbox("แสดงผลการทำนายทั้งหมด"):
    st.dataframe(forecast[['ds', 'yhat']].reset_index(drop=True))
