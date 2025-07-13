import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from prophet import Prophet

# ------------------ ฟังก์ชันหลัก ------------------
def forecasting_fn(df, plant, coin, interval_width):
    name = f"{plant}: {coin}"
    df_filtered = df[df['PLANTNAME'] == plant][['date', coin]]
    df_filtered.columns = ['ds', 'y']
    df_filtered['ds'] = pd.to_datetime(df_filtered['ds'])

    model = Prophet(
        interval_width=interval_width,
        changepoint_prior_scale=0.09983219300142447,
        changepoint_range=0.8349896986260539,
        seasonality_prior_scale=9.433629187865968,
        seasonality_mode='additive',
        yearly_seasonality=1,
        growth='linear'
    )
    model.fit(df_filtered)

    future = model.make_future_dataframe(periods=24, freq='ME')
    forecast = model.predict(future)

    df_filtered.reset_index(inplace=True)
    return model, forecast, future, name, df_filtered

# ------------------ ฟังก์ชันกราฟรวม ------------------
def plot_forecast_plotly(name, df, forecast, fiscal_year=None):
    if fiscal_year:
        year_ad = fiscal_year - 543
        start_date = pd.to_datetime(f"{year_ad - 1}-10-01")
        end_date = pd.to_datetime(f"{year_ad}-09-30")
        df = df[(df['ds'] >= start_date) & (df['ds'] <= end_date)]
        forecast = forecast[(forecast['ds'] >= start_date) & (forecast['ds'] <= end_date)]

    upper_bound = forecast['yhat_upper']
    lower_bound = forecast['yhat_lower']

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df['ds'], y=df['y'], mode='markers', name='Actual', marker=dict(color='rgba(137, 196, 244, 0.9)')))
    fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat'], mode='lines', name='Forecast', line=dict(color='rgba(255, 99, 132, 0.9)')))
    fig.add_trace(go.Scatter(x=forecast['ds'], y=upper_bound, mode='lines', name='Upper Bound', line=dict(width=0), showlegend=False))
    fig.add_trace(go.Scatter(x=forecast['ds'], y=lower_bound, mode='lines', name='Forecast Range', fill='tonexty', line=dict(width=0), fillcolor='rgba(255, 99, 132, 0.2)', showlegend=True))
    fig.update_layout(title=f'{name} Forecasting (Prophet)', xaxis_title='Month', yaxis_title='Coins', width=1000, height=500)
    st.plotly_chart(fig, use_container_width=True)

# ------------------ Streamlit App ------------------
st.set_page_config(page_title="Forecasting Coins", layout="wide")
st.title("Owl Mint Forecast Dashboard")

# โหลดข้อมูล
data = pd.read_excel("Data_Monthly.xlsx", index_col=0)
df = data.copy()
df.rename(columns={'Fiscal_Year': 'FiscalYear'}, inplace=True)

coin_options = ['รวม', '0.25', '0.5', '1.0', '2.0', '5.0', '10.0']
center_options = df['PLANTNAME'].unique().tolist()
fiscal_years = sorted(df['FiscalYear'].unique().tolist())

# เลือกตัวกรอง
col1, col2, col3 = st.columns(3)
with col1:
    selected_center = st.selectbox("เลือกศูนย์ (Center)", center_options)
with col2:
    selected_coin = st.selectbox("เลือกเหรียญ (Coin)", coin_options)
with col3:
    year_options = ["ทั้งหมด"] + fiscal_years[::-1]
    selected_year = st.selectbox("เลือกปีงบประมาณ (เพื่อดูกราฟเฉพาะช่วงปี)", year_options)

# Slider ความเชื่อมั่น
interval_width_percent = st.slider(
    "ระดับความเชื่อมั่น (%) ที่ใช้สร้างช่วงการพยากรณ์ (Prediction Interval)",
    min_value=50, max_value=99, value=80
)
interval_width = interval_width_percent / 100

# Forecast
model, forecast, future, name, df_filtered = forecasting_fn(df, plant=selected_center, coin=selected_coin, interval_width=interval_width)

# หน่วยเหรียญ
coin_unit = 'บาท' if selected_coin == 'รวม' or float(selected_coin) >= 1 else 'สตางค์'

# คำนวณค่าพยากรณ์และ Safety Stock
forecast['safety_stock'] = forecast['yhat_upper'] - forecast['yhat']
mean_forecast = forecast['yhat'].mean()
mean_safety_stock = forecast['safety_stock'].mean()
total_required = mean_forecast + mean_safety_stock

# รายปี
mean_forecast_year = mean_forecast * 12
safety_stock_year = mean_safety_stock * 12
total_required_year = total_required * 12

# ระดับการให้บริการย้อนหลัง
merged = pd.merge(df_filtered, forecast[['ds', 'yhat']], on='ds', how='inner')
service_level_empirical = np.mean(merged['y'] <= merged['yhat']) * 100

# แสดงผลสรุป
st.subheader(f"ผลการทำนายเหรียญ {selected_coin} {coin_unit} @ {selected_center}")
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("ค่าเฉลี่ยที่ควรมีต่อเดือน", f"{mean_forecast:,.2f}")
with col2:
    st.metric("Safety Stock ต่อเดือน (จากช่วงความมั่นใจ)", f"{mean_safety_stock:,.2f}")
with col3:
    st.metric("ขั้นต่ำต่อเดือนที่ควรมี", f"{total_required:,.2f}")

col4, col5, col6 = st.columns(3)
with col4:
    st.metric("ค่าเฉลี่ยที่ควรมีต่อปี", f"{mean_forecast_year:,.2f}")
with col5:
    st.metric("Safety Stock ต่อปี (จากช่วงความมั่นใจ)", f"{safety_stock_year:,.2f}")
with col6:
    st.metric("ขั้นต่ำต่อปีที่ควรมี", f"{total_required_year:,.2f}")

st.info(f"ระดับการให้บริการย้อนหลังจริง (Empirical Service Level): {service_level_empirical:.2f}%")
st.info(f"ใช้ช่วงความเชื่อมั่นจาก Prophet ({interval_width_percent}%) เพื่อประเมิน Safety Stock")

# แสดงกราฟ
plot_forecast_plotly(name, df_filtered, forecast, fiscal_year=None if selected_year == "ทั้งหมด" else selected_year)

# ตารางรายเดือน (เฉพาะอนาคต 12 เดือน)
monthly_forecast = forecast[['ds', 'yhat', 'yhat_upper']].copy()
monthly_forecast['safety_stock'] = monthly_forecast['yhat_upper'] - monthly_forecast['yhat']
monthly_forecast['total_required'] = monthly_forecast['yhat'] + monthly_forecast['safety_stock']
monthly_forecast['month'] = monthly_forecast['ds'].dt.strftime('%b %Y')
latest_date = df_filtered['ds'].max()
monthly_forecast = monthly_forecast[monthly_forecast['ds'] > latest_date].head(12)
monthly_forecast_display = monthly_forecast[['month', 'yhat', 'safety_stock', 'total_required']].copy()
monthly_forecast_display.columns = ['เดือน', 'ค่าพยากรณ์', 'Safety Stock', 'รวมที่ควรมี']
monthly_forecast_display = monthly_forecast_display.round(2)

st.subheader("ตารางการเตรียมพร้อมเหรียญรายเดือน (12 เดือนข้างหน้า)")
st.dataframe(monthly_forecast_display, use_container_width=True)

# กราฟ grouped bar รายเดือน ม.ค. - ธ.ค.
monthly_chart = forecast[['ds', 'yhat', 'yhat_upper']].copy()
monthly_chart['safety_stock'] = monthly_chart['yhat_upper'] - monthly_chart['yhat']
monthly_chart['total_required'] = monthly_chart['yhat_upper']
monthly_chart['month'] = monthly_chart['ds'].dt.month
monthly_chart['month_name'] = monthly_chart['ds'].dt.strftime('%b')

next_year = latest_date.year + 1
monthly_chart = monthly_chart[(monthly_chart['ds'].dt.year == next_year) & (monthly_chart['ds'].dt.month <= 12)]

monthly_grouped = monthly_chart.groupby(['month', 'month_name']).agg({
    'yhat': 'mean',
    'safety_stock': 'mean',
    'total_required': 'mean'
}).reset_index()

month_order = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
               'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
monthly_grouped['month_name'] = pd.Categorical(monthly_grouped['month_name'], categories=month_order, ordered=True)
monthly_grouped.sort_values('month', inplace=True)

monthly_long = monthly_grouped.melt(
    id_vars='month_name',
    value_vars=['yhat', 'safety_stock', 'total_required'],
    var_name='ประเภท',
    value_name='จำนวน'
)
label_map = {'yhat': 'ค่าพยากรณ์', 'safety_stock': 'Safety Stock', 'total_required': 'รวมที่ควรมี'}
monthly_long['ประเภท'] = monthly_long['ประเภท'].map(label_map)

fig_bar = px.bar(
    monthly_long,
    x='month_name',
    y='จำนวน',
    color='ประเภท',
    barmode='group',
    text='จำนวน',
    labels={'month_name': 'เดือน', 'จำนวน': 'ปริมาณ'},
    title=f'ปริมาณเหรียญที่ต้องเตรียมแต่ละเดือน (ม.ค. - ธ.ค. {next_year})'
)
fig_bar.update_traces(texttemplate='%{text:.2f}', textposition='outside')
fig_bar.update_layout(width=1000, height=500)
st.plotly_chart(fig_bar, use_container_width=True)

# ตัวเลือกแสดงผลเต็ม
if st.checkbox("แสดงผลการทำนายทั้งหมด"):
    st.dataframe(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].reset_index(drop=True))
