import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from prophet import Prophet
import locale
from sklearn.metrics import r2_score

# ------------------ การตั้งค่าเริ่มต้น ------------------
st.set_page_config(page_title="AI Forecast Dashboard", layout="wide", initial_sidebar_state="expanded")
try:
    locale.setlocale(locale.LC_TIME, 'th_TH')
except locale.Error:
    st.warning("ไม่สามารถตั้งค่า Locale ภาษาไทยได้ อาจแสดงผลเดือนเป็นภาษาอังกฤษ")

# ------------------ Custom CSS for Tech UI ------------------
st.markdown("""
<style>
    /* Main App Font and Background */
    html, body, [class*="st-"] {
        font-family: 'IBM Plex Sans', sans-serif;
    }
    .main {
        background-color: #0E1117;
    }
    
    /* Sidebar Styling */
    [data-testid="stSidebar"] {
        background-color: #1E1E2F;
        border-right: 2px solid #4A4A6A;
    }
    [data-testid="stSidebar"] h2, [data-testid="stSidebar"] .st-emotion-cache-10oheor {
        color: #FFFFFF;
    }

    /* Metric Box Styling */
    [data-testid="stMetric"] {
        background-color: rgba(44, 51, 64, 0.3);
        border: 1px solid #4A4A6A;
        border-radius: 12px;
        padding: 20px;
        box-shadow: 0 4px 30px rgba(0, 0, 0, 0.1);
        backdrop-filter: blur(5px);
        -webkit-backdrop-filter: blur(5px);
    }
    [data-testid="stMetricLabel"] {
        color: #A0AEC0; /* Light gray for label */
    }
    [data-testid="stMetricValue"] {
        color: #FFFFFF;
        font-size: 1.75rem;
    }

    /* Headers and Titles */
    h1 {
        color: #FFFFFF;
        text-shadow: 0 0 10px rgba(0, 191, 255, 0.5);
    }
    h2, h3 {
        color: #E2E8F0;
        border-left: 4px solid #00BFFF;
        padding-left: 10px;
    }

    /* Radio Buttons as Modern Toggles */
    div[role="radiogroup"] > label {
        background-color: #2D3748;
        color: #E2E8F0;
        padding: 8px 12px;
        border-radius: 8px;
        margin: 0 5px;
        border: 1px solid transparent;
        transition: all 0.3s ease;
    }
    div[role="radiogroup"] > label:hover {
        background-color: #4A5568;
        border-color: #00BFFF;
    }
    
    /* Divider */
    hr {
        background: linear-gradient(to right, #00BFFF, transparent);
        height: 2px;
        border: none;
    }

</style>
""", unsafe_allow_html=True)


# ------------------ พารามิเตอร์สำหรับโมเดล Prophet ------------------
PARAMS_DISTRIBUTION = {
    'changepoint_prior_scale': 0.09983219300142447, 'changepoint_range': 0.8349896986260539,
    'seasonality_prior_scale': 9.433629187865968, 'seasonality_mode': 'additive',
    'yearly_seasonality': 1, 'growth': 'linear'
}
PARAMS_RETURNS = {
    'changepoint_prior_scale': 0.016589549889387597, 'changepoint_range': 0.8542895834435257,
    'seasonality_prior_scale': 6.285629251291158, 'seasonality_mode': 'additive',
    'yearly_seasonality': 1, 'growth': 'linear'
}

# ------------------ ฟังก์ชัน Utility ------------------
@st.cache_data
def load_data(file_path, sheet_name, date_col):
    try:
        df = pd.read_excel(file_path, sheet_name=sheet_name)
        df.rename(columns={date_col: 'ds'}, inplace=True)
        df['ds'] = pd.to_datetime(df['ds'])
        processed_dfs = []
        numeric_cols = [col for col in df.columns if pd.api.types.is_numeric_dtype(df[col])]
        for plant in df['PLANTNAME'].unique():
            plant_df = df[df['PLANTNAME'] == plant].copy()
            plant_df.set_index('ds', inplace=True)
            plant_df_resampled = plant_df[numeric_cols].resample('M').sum()
            plant_df_resampled['PLANTNAME'] = plant
            processed_dfs.append(plant_df_resampled)
        final_df = pd.concat(processed_dfs).reset_index()
        return final_df
    except FileNotFoundError:
        st.error(f"ไม่พบไฟล์ข้อมูล: {file_path}")
        return None

@st.cache_data
def forecasting_fn(df_name, plant, value_col, prophet_params, periods, interval_width):
    if df_name == 'dist':
        df = load_data("DATA_Distribution_Sum_Center.xlsx", 'Sheet1', 'date')
    else:
        df = load_data("DATA_Exchange_Sum_Center.xlsx", 'Sheet2', 'dc')
    if df is None: return None, None, "ไม่สามารถโหลดไฟล์ข้อมูลได้"
    if value_col not in df.columns: return None, None, f"ไม่พบคอลัมน์ '{value_col}'"
    df_filtered = df.loc[df['PLANTNAME'] == plant, ['ds', value_col]].copy()
    df_filtered.columns = ['ds', 'y']
    df_filtered = df_filtered[df_filtered['y'] != 0]
    df_filtered.dropna(inplace=True)
    if len(df_filtered) < 2: return None, None, "มีข้อมูลไม่เพียงพอสำหรับการพยากรณ์"
    model = Prophet(**prophet_params, interval_width=interval_width)
    model.fit(df_filtered)
    future = model.make_future_dataframe(periods=periods, freq='M')
    forecast = model.predict(future)
    return forecast, df_filtered, None

# ------------------ ฟังก์ชันการแสดงผล ------------------
def display_forecast_output(column_container, title, forecast, df_filtered, confidence_label, name_for_title):
    with column_container:
        st.subheader(title)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df_filtered['ds'], y=df_filtered['y'], mode='markers', name='Actual', marker=dict(color='#00BFFF', size=8, line=dict(width=1, color='white'))))
        fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat'], mode='lines', name='Forecast', line=dict(color='#FF1493', width=3)))
        fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat_upper'], mode='lines', line=dict(width=0), showlegend=False))
        fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat_lower'], mode='lines', fill='tonexty', fillcolor='rgba(255, 20, 147, 0.2)', line=dict(width=0), name=f'CI ({confidence_label})'))
        fig.update_layout(title=f'Forecast: {name_for_title}', template='plotly_dark', legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1))
        st.plotly_chart(fig, use_container_width=True)
        
        results_df = pd.merge(forecast[['ds', 'yhat']], df_filtered[['ds', 'y']], on='ds', how='left')
        results_df.rename(columns={'y': 'ค่าจริง', 'yhat': 'ค่าพยากรณ์'}, inplace=True)
        results_df['ค่าความแตกต่าง'] = results_df['ค่าพยากรณ์'] - results_df['ค่าจริง']
        results_df['Error (%)'] = np.where(results_df['ค่าจริง'].notna() & (results_df['ค่าจริง'] != 0), np.abs(results_df['ค่าความแตกต่าง'] / results_df['ค่าจริง']) * 100, np.nan)
        
        st.write("ตารางผลการพยากรณ์")
        def format_thai_date(dt): return f"{dt.year + 543} {dt.strftime('%B')}"
        results_df['เดือน'] = results_df['ds'].apply(format_thai_date)
        display_df = results_df[['เดือน', 'ค่าพยากรณ์', 'ค่าจริง', 'ค่าความแตกต่าง', 'Error (%)']].copy()
        for col in ['ค่าพยากรณ์', 'ค่าจริง', 'ค่าความแตกต่าง', 'Error (%)']:
            display_df[col] = display_df[col].apply(lambda x: f"{x:,.2f}" if pd.notna(x) else '-')
        st.dataframe(display_df, use_container_width=True, hide_index=True, height=400)

        valid_results = results_df.dropna(subset=['ค่าจริง'])
        mse, rmse, mape, r2 = [np.nan] * 4
        if not valid_results.empty:
            actual, forecast_vals = valid_results['ค่าจริง'], valid_results['ค่าพยากรณ์']
            mse = np.mean((actual - forecast_vals)**2)
            rmse = np.sqrt(mse) # FIX: Correctly calculate RMSE from MSE
            r2 = r2_score(actual, forecast_vals)
            if not actual[actual != 0].empty:
                 mape = np.mean(np.abs((actual - forecast_vals) / actual)[actual != 0]) * 100
        
        st.write("ประเมินความคลาดเคลื่อน (Accuracy Metrics)")
        m1, m2 = st.columns(2)
        m3, m4 = st.columns(2)
        m1.metric("MSE", f"{mse:,.2f}" if pd.notna(mse) else "N/A")
        m2.metric("RMSE", f"{rmse:,.2f}" if pd.notna(rmse) else "N/A")
        m3.metric("MAPE", f"{mape:,.2f}%" if pd.notna(mape) else "N/A")
        m4.metric("R-squared", f"{r2:,.2f}" if pd.notna(r2) else "N/A")

# ------------------ โครงสร้างหลักของแอป ------------------
st.title("🦉 AI Mint Forecast Dashboard")

# --- โหลดข้อมูล ---
df_dist_check = load_data("DATA_Distribution_Sum_Center.xlsx", 'Sheet1', 'date')
df_ret_check = load_data("DATA_Exchange_Sum_Center.xlsx", 'Sheet2', 'dc')
if df_dist_check is None or df_ret_check is None: st.stop()

# --- Sidebar ---
st.sidebar.header("⚙️ ตัวกรอง")
center_options = sorted(list(set(df_dist_check['PLANTNAME'].unique())|set(df_ret_check['PLANTNAME'].unique())))
selected_center = st.sidebar.selectbox("เลือกศูนย์ (Center)", center_options, index=(center_options.index('ทั่วประเทศ') if 'ทั่วประเทศ' in center_options else 0))
coin_display_map = {'10 บาท': '10.0', '5 บาท': '5.0', '2 บาท': '2.0', '1 บาท': '1.0', '50 สตางค์': '0.5', '25 สตางค์': '0.25', 'รวมทั้งหมด': 'รวม'}
selected_coin_display = st.sidebar.selectbox("เลือกชนิดราคา", options=list(coin_display_map.keys()))
forecast_periods = st.sidebar.number_input("พยากรณ์ล่วงหน้า (เดือน)", 1, 120, 24, 1)
confidence_options = {'90%': 0.90, '95%': 0.95, '99%': 0.99}
selected_confidence_label = st.sidebar.selectbox("ระดับความเชื่อมั่น", options=list(confidence_options.keys()), index=1)
return_type_map = {'จ่ายได้': 'G', 'ชำรุด': 'B', 'รวม': 'A'}
return_coin_map = {'10.0': '10', '5.0': '5', '2.0': '2', '1.0': '1', '0.5': '0.50', '0.25': '0.25', 'รวม': 'total'}

# --- ส่วนด้านการประมาณการ ---
st.header("📈 ประมาณการส่วนต่างสุทธิ")
selected_return_type_est_display = st.radio("เลือกสภาพเหรียญ (รับคืน) เพื่อเปรียบเทียบ", options=list(return_type_map.keys()), horizontal=True, key="estimation_return_type", index=2)

dist_value_col = coin_display_map[selected_coin_display]
forecast_dist, df_filtered_dist, error_dist = forecasting_fn('dist', selected_center, dist_value_col, PARAMS_DISTRIBUTION, forecast_periods, confidence_options[selected_confidence_label])
ret_val_est = coin_display_map[selected_coin_display]
ret_col_est = f"{return_coin_map[ret_val_est]}_{return_type_map[selected_return_type_est_display]}" if return_coin_map[ret_val_est] == 'total' else f"{return_coin_map[ret_val_est]}{return_type_map[selected_return_type_est_display]}"
forecast_ret_est, df_filtered_ret_est, error_ret_est = forecasting_fn('ret', selected_center, ret_col_est, PARAMS_RETURNS, forecast_periods, confidence_options[selected_confidence_label])

if error_dist is None and error_ret_est is None:
    dist_future = pd.merge(forecast_dist[['ds', 'yhat']], df_filtered_dist[['ds', 'y']], on='ds', how='left').query("y.isna()").rename(columns={'yhat': 'yhat_dist'})
    ret_future = pd.merge(forecast_ret_est[['ds', 'yhat']], df_filtered_ret_est[['ds', 'y']], on='ds', how='left').query("y.isna()").rename(columns={'yhat': 'yhat_ret'})
    future_net_df = pd.merge(dist_future[['ds', 'yhat_dist']], ret_future[['ds', 'yhat_ret']], on='ds', how='outer').sort_values('ds').fillna(0)
    
    if not future_net_df.empty:
        future_net_df['net_forecast'] = future_net_df['yhat_dist'] - future_net_df['yhat_ret']
        
        # --- IMPROVEMENT: Enhanced Graph for Net Estimation ---
        fig_net = go.Figure()
        fig_net.add_trace(go.Scatter(
            x=future_net_df['ds'], y=future_net_df['yhat_dist'], name='พยากรณ์จ่ายแลก', 
            line=dict(color='#00BFFF', width=2),
            hovertemplate='จ่ายแลก: %{y:,.2f}<extra></extra>'
        ))
        fig_net.add_trace(go.Scatter(
            x=future_net_df['ds'], y=future_net_df['yhat_ret'], name='พยากรณ์รับคืน', 
            line=dict(color='#FFA500', width=2),
            hovertemplate='รับคืน: %{y:,.2f}<extra></extra>'
        ))
        fig_net.add_trace(go.Bar(
            x=future_net_df['ds'], y=future_net_df['net_forecast'], name='ส่วนต่างสุทธิ',
            marker_color=np.where(future_net_df['net_forecast'] >= 0, '#28a745', '#dc3545'),
            customdata=future_net_df[['yhat_dist', 'yhat_ret']],
            hovertemplate=(
                '<b>เดือน: %{x|%B %Y}</b><br>' +
                '<b>ส่วนต่างสุทธิ: %{y:,.2f}</b><br>' +
                'จ่ายแลก: %{customdata[0]:,.2f}<br>' +
                'รับคืน: %{customdata[1]:,.2f}<extra></extra>'
            )
        ))
        fig_net.update_layout(
            title=f"ประมาณการ: {selected_coin_display} (เทียบกับรับคืน '{selected_return_type_est_display}')", 
            template='plotly_dark',
            legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
            barmode='relative'
        )
        st.plotly_chart(fig_net, use_container_width=True)
        
        st.write("ตารางข้อมูลส่วนต่างสุทธิ")
        display_net_df = future_net_df[['ds', 'yhat_dist', 'yhat_ret', 'net_forecast']].rename(columns={'ds': 'เดือน', 'yhat_dist': 'พยากรณ์จ่ายแลก', 'yhat_ret': 'พยากรณ์รับคืน', 'net_forecast': 'ส่วนต่างสุทธิ'})
        display_net_df['เดือน'] = display_net_df['เดือน'].apply(lambda dt: f"{dt.year + 543} {dt.strftime('%B')}")
        for col in ['พยากรณ์จ่ายแลก', 'พยากรณ์รับคืน', 'ส่วนต่างสุทธิ']:
            display_net_df[col] = display_net_df[col].apply(lambda x: f"{x:,.2f}")
        st.dataframe(display_net_df, use_container_width=True, hide_index=True)
else:
    if error_dist: st.warning(f"คำนวณส่วนต่างไม่ได้ (จ่ายแลก): {error_dist}")
    if error_ret_est: st.warning(f"คำนวณส่วนต่างไม่ได้ (รับคืน): {error_ret_est}")

st.divider()

# --- แบ่งหน้าจอเป็น 2 คอลัมน์ ---
col_dist, col_ret = st.columns(2)
with col_dist:
    if error_dist: st.error(f"**การจ่ายแลก:** {error_dist}")
    elif forecast_dist is not None:
        display_forecast_output(st.container(), "📊 การจ่ายแลก", forecast_dist, df_filtered_dist, selected_confidence_label, f"{selected_coin_display} @ {selected_center}")
with col_ret:
    sub_col, radio_col = st.columns([1, 2])
    with sub_col: st.subheader("📊 การรับคืน")
    with radio_col:
        selected_return_type_display = st.radio("เลือกสภาพเหรียญ", options=list(return_type_map.keys()), horizontal=True, key="return_type", label_visibility="collapsed")
    ret_val = coin_display_map[selected_coin_display]
    ret_col = f"{return_coin_map[ret_val]}_{return_type_map[selected_return_type_display]}" if return_coin_map[ret_val] == 'total' else f"{return_coin_map[ret_val]}{return_type_map[selected_return_type_display]}"
    forecast_ret, df_filtered_ret, error_ret = forecasting_fn('ret', selected_center, ret_col, PARAMS_RETURNS, forecast_periods, confidence_options[selected_confidence_label])
    if error_ret: st.error(f"**การรับคืน:** {error_ret}")
    elif forecast_ret is not None:
        display_forecast_output(st.container(), "", forecast_ret, df_filtered_ret, selected_confidence_label, f"{selected_coin_display} ({selected_return_type_display}) @ {selected_center}")

