import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from prophet import Prophet
from scipy.stats import norm

# ------------------ ฟังก์ชันหลัก ------------------
def forecasting_fn(df, plant, coin):
    """
    ฟังก์ชันสำหรับสร้างโมเดล Prophet และพยากรณ์ข้อมูล
    หมายเหตุ: ตัด interval_width ออก เพราะจะคำนวณ Safety Stock แยก
    """
    name = f"{plant}: {coin}"
    df_filtered = df[df['PLANTNAME'] == plant][['date', coin]]
    df_filtered.columns = ['ds', 'y']
    df_filtered['ds'] = pd.to_datetime(df_filtered['ds'])
    
    # ทำให้ข้อมูลไม่มีค่า Null เพื่อให้โมเดลทำงานได้
    df_filtered.dropna(inplace=True)

    model = Prophet(
        interval_width=0.95,  # กำหนดค่าคงที่สำหรับแสดงผลบนกราฟเท่านั้น
        changepoint_prior_scale=0.09983219300142447,
        changepoint_range=0.8349896986260539,
        seasonality_prior_scale=9.433629187865968,
        seasonality_mode='additive',
        yearly_seasonality=1,
        growth='linear'
    )
    model.fit(df_filtered)

    future = model.make_future_dataframe(periods=60, freq='ME')
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
    fig.add_trace(go.Scatter(
        x=df['ds'],
        y=df['y'],
        mode='markers',
        name='Actual',
        marker=dict(color='blue', size=8, symbol='circle')
    ))

    fig.add_trace(go.Scatter(
        x=forecast['ds'],
        y=forecast['yhat'],
        mode='lines',
        name='Forecast',
        line=dict(color='rgba(255, 99, 132, 0.9)', width=3)
    ))

    # ช่วงความเชื่อมั่น (แสดงผลเพื่อดูแนวโน้ม ไม่เกี่ยวกับ Safety Stock)
    fig.add_trace(go.Scatter(
        x=forecast['ds'],
        y=upper_bound,
        mode='lines',
        line=dict(width=0),
        showlegend=False
    ))
    fig.add_trace(go.Scatter(
        x=forecast['ds'],
        y=lower_bound,
        mode='lines',
        fill='tonexty',
        fillcolor='rgba(255, 99, 132, 0.15)',
        line=dict(width=0),
        name='Confidence Interval (95%)'
    ))

    fig.update_layout(
        title=f'{name} Forecasting (Prophet)',
        xaxis_title='Month',
        yaxis_title='Coins (ล้านเหรียญ)',
        width=1100,
        height=550,
        barmode='overlay',
        legend=dict(
            orientation='h',
            yanchor='bottom',
            y=1.05,
            xanchor='right',
            x=1
        ),
        margin=dict(l=50, r=50, t=80, b=50)
    )

    st.plotly_chart(fig, use_container_width=True)

# ------------------ Streamlit App ------------------
st.set_page_config(page_title="Forecasting Coins", layout="wide")
st.title("🦉 Owl Mint Forecast Dashboard")

# โหลดข้อมูล
data = pd.read_excel("Data_Monthly_Updated.xlsx", index_col=0)
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

# Slider ระดับการให้บริการ (Service Level)
service_level_percent = st.slider(
    "ระดับการให้บริการที่ต้องการ (%) เพื่อคำนวณ Safety Stock (Service Level)",
    min_value=50.0, max_value=99.9, value=95.0, step=0.1
)
service_level = service_level_percent / 100

# Forecast
model, forecast, future, name, df_filtered = forecasting_fn(df, plant=selected_center, coin=selected_coin)

# --- คำนวณ SAFETY STOCK แบบใหม่ ---
# 1. รวมข้อมูลจริงในอดีตกับค่าพยากรณ์ในอดีตเข้าด้วยกัน
merged_historical = pd.merge(df_filtered, forecast[['ds', 'yhat']], on='ds', how='inner')

# 2. คำนวณ Forecast Error (ค่าจริง - ค่าพยากรณ์)
merged_historical['error'] = merged_historical['y'] - merged_historical['yhat']

# 3. คำนวณ Standard Deviation ของ Error
std_error = merged_historical['error'].std()

# 4. หาค่า Z-score จาก Service Level ที่เลือก
z_score = norm.ppf(service_level)

# 5. คำนวณ Safety Stock (ป้องกันค่าติดลบหาก Service Level < 50%)
safety_stock_value = max(0, z_score * std_error) if pd.notna(std_error) else 0

# กำหนดค่า Safety Stock ที่คำนวณได้ให้กับทุกช่วงเวลาในอนาคต
forecast['safety_stock'] = safety_stock_value
# -----------------------------------

# หน่วยเหรียญ
coin_unit = 'บาท' if selected_coin == 'รวม' or float(selected_coin) >= 1 else 'สตางค์'

# คำนวณค่าเฉลี่ย
mean_forecast = forecast['yhat'].mean()
total_required = mean_forecast + safety_stock_value

# รายปี
mean_forecast_year = mean_forecast * 12
safety_stock_year = safety_stock_value * 12
total_required_year = total_required * 12

# แสดงผลสรุป
st.subheader(f"ผลการทำนายเหรียญ {selected_coin} {coin_unit} @ {selected_center}: ล้านเหรียญ")
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("ค่าเฉลี่ยที่ควรมีต่อเดือน (Forecast)", f"{mean_forecast:,.2f}")
with col2:
    # แสดงค่า Safety Stock ที่คำนวณได้
    st.metric("Safety Stock ต่อเดือน (คำนวณจาก Forecast Error)", f"{safety_stock_value:,.2f}")
with col3:
    st.metric("ขั้นต่ำต่อเดือนที่ควรมี (Forecast + Safety Stock)", f"{total_required:,.2f}")

col4, col5, col6 = st.columns(3)
with col4:
    st.metric("ค่าเฉลี่ยที่ควรมีต่อปี", f"{mean_forecast_year:,.2f}")
with col5:
    st.metric("Safety Stock ต่อปี", f"{safety_stock_year:,.2f}")
with col6:
    st.metric("ขั้นต่ำต่อปีที่ควรมี", f"{total_required_year:,.2f}")

st.info(f"Safety Stock คำนวณจากความคลาดเคลื่อนของการพยากรณ์ในอดีต ณ ระดับการให้บริการ (Service Level) {service_level_percent:.1f}% (Z-score: {z_score:.2f}, Std Dev of Error: {std_error:.2f})")

# แสดงกราฟ
plot_forecast_plotly(name, df_filtered, forecast, fiscal_year=None if selected_year == "ทั้งหมด" else selected_year)


# ตารางรายเดือน (เฉพาะอนาคต 12 เดือน)
monthly_forecast = forecast[['ds', 'yhat']].copy()
monthly_forecast['safety_stock'] = safety_stock_value # ใช้ค่าที่คำนวณใหม่
monthly_forecast['total_required'] = monthly_forecast['yhat'] + monthly_forecast['safety_stock']
monthly_forecast['month'] = monthly_forecast['ds'].dt.strftime('%b %Y')
latest_date = df_filtered['ds'].max()
monthly_forecast = monthly_forecast[monthly_forecast['ds'] > latest_date].head(12)
monthly_forecast_display = monthly_forecast[['month', 'yhat', 'safety_stock', 'total_required']].copy()
monthly_forecast_display.columns = ['เดือน', 'ค่าพยากรณ์', 'Safety Stock', 'รวมที่ควรมี']
monthly_forecast_display = monthly_forecast_display.round(2)

st.subheader("ตารางการเตรียมพร้อมเหรียญรายเดือน (12 เดือนข้างหน้า)")
st.dataframe(monthly_forecast_display, use_container_width=True)

# (ส่วนที่เหลือของโค้ดสำหรับการสร้างกราฟและตารางยังคงเดิม แต่จะใช้ค่า Safety Stock ใหม่โดยอัตโนมัติ)
# ... (วางโค้ดส่วนที่เหลือตั้งแต่ กราฟ grouped bar เป็นต้นไป ที่นี่)
month_order = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
label_map = {'yhat': 'ค่าพยากรณ์', 'safety_stock': 'Safety Stock', 'total_required': 'รวมที่ควรมี'}

# กราฟ grouped bar รายปี ครบ 5 ปี
st.subheader("กราฟปริมาณเหรียญรายเดือนแยกตามปี พร้อมยอดรวม")
for year in range(latest_date.year + 1, latest_date.year + 6):
    yearly_chart = forecast.copy()
    yearly_chart['year'] = yearly_chart['ds'].dt.year
    yearly_chart['month'] = yearly_chart['ds'].dt.month
    yearly_chart['month_name'] = yearly_chart['ds'].dt.strftime('%b')
    yearly_chart['safety_stock'] = safety_stock_value # ใช้ค่าใหม่
    yearly_chart['total_required'] = yearly_chart['yhat'] + yearly_chart['safety_stock'] # คำนวณใหม่

    this_year_data = yearly_chart[yearly_chart['year'] == year]
    monthly_grouped = this_year_data.groupby(['month', 'month_name']).agg({
        'yhat': 'mean',
        'safety_stock': 'mean',
        'total_required': 'mean'
    }).reset_index()

    monthly_grouped['month_name'] = pd.Categorical(monthly_grouped['month_name'], categories=month_order, ordered=True)
    monthly_grouped.sort_values('month', inplace=True)

    monthly_long = monthly_grouped.melt(
        id_vars='month_name',
        value_vars=['yhat', 'safety_stock', 'total_required'],
        var_name='ประเภท',
        value_name='จำนวน'
    )
    monthly_long['ประเภท'] = monthly_long['ประเภท'].map(label_map)

    fig_bar = px.bar(
        monthly_long,
        x='month_name',
        y='จำนวน',
        color='ประเภท',
        barmode='group',
        text=monthly_long['จำนวน'].apply(lambda x: f"{x:,.2f}"),
        labels={'month_name': 'เดือน', 'จำนวน': 'ปริมาณ'},
        title=f'ปริมาณเหรียญที่ต้องเตรียมแต่ละเดือน ({year})'
    )
    fig_bar.update_traces(textposition='outside')

    total_yhat = this_year_data['yhat'].sum()
    total_safety = this_year_data['safety_stock'].sum()
    total_total = this_year_data['total_required'].sum()

    fig_bar.update_layout(
        width=1000,
        height=500,
        annotations=[
            dict(
                x=0.5,
                y=1.15,
                xref='paper',
                yref='paper',
                text=f"ยอดรวมปี {year}: ค่าพยากรณ์ = {total_yhat:,.2f}, Safety Stock = {total_safety:,.2f}, รวมที่ควรมี = {total_total:,.2f}",
                showarrow=False,
                font=dict(size=12)
            )
        ]
    )
    st.plotly_chart(fig_bar, use_container_width=True)


if st.checkbox("แสดงผลการทำนายทั้งหมด"):
    full_results = pd.merge(
        forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']],
        df_filtered[['ds', 'y']],
        on='ds',
        how='left'
    )
    full_results.rename(columns={
        'ds': 'วันที่',
        'y': 'ค่าจริง (Actual)',
        'yhat': 'ค่าพยากรณ์ (Forecast)',
        'yhat_lower': 'ช่วงต่ำสุด (Lower Bound)',
        'yhat_upper': 'ช่วงสูงสุด (Upper Bound)'
    }, inplace=True)

    full_results['ส่วนต่าง (Actual - Forecast)'] = full_results['ค่าจริง (Actual)'] - full_results['ค่าพยากรณ์ (Forecast)']

    # คำนวณ RMSE และ MAPE จากข้อมูลที่มีค่าจริงเท่านั้น
    valid_results = full_results.dropna(subset=['ค่าจริง (Actual)', 'ค่าพยากรณ์ (Forecast)'])
    actual = valid_results['ค่าจริง (Actual)']
    forecast_vals = valid_results['ค่าพยากรณ์ (Forecast)']
    
    # ตรวจสอบว่ามีข้อมูลสำหรับคำนวณหรือไม่
    if not actual.empty:
        rmse = np.sqrt(np.mean((actual - forecast_vals) ** 2))
        # ป้องกันการหารด้วยศูนย์สำหรับ MAPE
        mape = np.mean(np.abs((actual - forecast_vals) / actual[actual != 0])) * 100
    else:
        rmse = np.nan
        mape = np.nan


    full_results = full_results.round(2)
    st.dataframe(full_results.reset_index(drop=True), use_container_width=True)

    st.write(f"**RMSE (Root Mean Square Error):** {rmse:,.2f}")
    st.write(f"**MAPE (Mean Absolute Percentage Error):** {mape:,.2f}%")
