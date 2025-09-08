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

# ------------------ Custom CSS for Light UI ------------------
st.markdown("""
<style>
    /* Main App Font and Background */
    html, body {
        font-family: 'IBM Plex Sans', sans-serif;
        color: #1a1a1a; /* Dark text for light background */
    }
    .main {
        background-color: #F0F2F6;
    }
    
    /* Sidebar Styling */
    [data-testid="stSidebar"] {
        background-color: #FFFFFF;
        border-right: 2px solid #E0E0E0;
    }
    [data-testid="stSidebar"] h2, [data-testid="stSidebar"] .st-emotion-cache-10oheor {
        color: #1a1a1a;
    }

    /* Metric Box Styling */
    [data-testid="stMetric"] {
        background-color: #FFFFFF;
        border: 1px solid #E0E0E0;
        border-radius: 12px;
        padding: 20px;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.05);
    }
    [data-testid="stMetricLabel"] {
        color: #555555; /* Medium gray for label */
    }
    [data-testid="stMetricValue"] {
        color: #1a1a1a;
        font-size: 1.75rem;
    }

    /* Headers and Titles */
    h1 {
        color: #0052CC;
    }
    h2, h3 {
        color: #1a1a1a;
        border-left: 4px solid #0052CC;
        padding-left: 10px;
    }
    h4 {
        color: #333;
    }

    /* Radio Buttons as Modern Toggles */
    div[role="radiogroup"] > label {
        background-color: #E9ECEF;
        color: #333;
        padding: 8px 12px;
        border-radius: 8px;
        margin: 0 5px;
        border: 1px solid #DEE2E6;
        transition: all 0.3s ease;
    }
    div[role="radiogroup"] > label:hover {
        background-color: #DDE2E6;
        border-color: #0052CC;
    }
    
    /* Divider */
    hr {
        background: linear-gradient(to right, #0052CC, transparent);
        height: 1px;
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

# Thai month abbreviations
thai_month_abbr = {1: 'ม.ค.', 2: 'ก.พ.', 3: 'มี.ค.', 4: 'เม.ย.', 5: 'พ.ค.', 6: 'มิ.ย.', 7: 'ก.ค.', 8: 'ส.ค.', 9: 'ก.ย.', 10: 'ต.ค.', 11: 'พ.ย.', 12: 'ธ.ค.'}
def format_thai_date_short(dt): return f"{thai_month_abbr[dt.month]} {str(dt.year + 543)[-2:]}"
def format_month_year_thai(dt): return f"{dt.strftime('%B')} {dt.year + 543}"
def get_fiscal_year(dt): return dt.year + 543 if dt.month >= 10 else dt.year + 542

# ------------------ ฟังก์ชันการแสดงผล ------------------
def display_forecast_output(column_container, title, forecast, df_filtered, confidence_label, name_for_title):
    with column_container:
        st.subheader(title)
        
        plot_df_filtered = df_filtered.copy()
        plot_forecast = forecast.copy()
        plot_df_filtered['ds_str'] = plot_df_filtered['ds'].apply(format_thai_date_short)
        plot_forecast['ds_str'] = plot_forecast['ds'].apply(format_thai_date_short)

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=plot_df_filtered['ds_str'], y=plot_df_filtered['y'], mode='markers', name='Actual', marker=dict(color='#0052CC', size=8, line=dict(width=1, color='white'))))
        fig.add_trace(go.Scatter(x=plot_forecast['ds_str'], y=plot_forecast['yhat'], mode='lines', name='Forecast', line=dict(color='#D6336C', width=3)))
        fig.add_trace(go.Scatter(x=plot_forecast['ds_str'], y=plot_forecast['yhat_upper'], mode='lines', line=dict(width=0), showlegend=False))
        fig.add_trace(go.Scatter(x=plot_forecast['ds_str'], y=plot_forecast['yhat_lower'], mode='lines', fill='tonexty', fillcolor='rgba(214, 51, 108, 0.15)', line=dict(width=0), name=f'CI ({confidence_label})'))
        fig.update_layout(title=f'Forecast: {name_for_title}', template='plotly_white', legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1), xaxis_tickangle=-90)
        st.plotly_chart(fig, use_container_width=True)
        
        with st.expander("แสดง/ซ่อนตารางผลการพยากรณ์และค่าสถิติ"):
            results_df = pd.merge(forecast[['ds', 'yhat']], df_filtered[['ds', 'y']], on='ds', how='left')
            results_df.rename(columns={'y': 'ค่าจริง', 'yhat': 'ค่าพยากรณ์'}, inplace=True)
            results_df['ค่าความแตกต่าง'] = results_df['ค่าพยากรณ์'] - results_df['ค่าจริง']
            results_df['Error (%)'] = np.where(results_df['ค่าจริง'].notna() & (results_df['ค่าจริง'] != 0), np.abs(results_df['ค่าความแตกต่าง'] / results_df['ค่าจริง']) * 100, np.nan)
            st.write("ตารางผลการพยากรณ์")
            results_df['เดือน'] = results_df['ds'].apply(lambda dt: f"{dt.strftime('%B')} {dt.year + 543}")
            display_df = results_df[['เดือน', 'ค่าพยากรณ์', 'ค่าจริง', 'ค่าความแตกต่าง', 'Error (%)']].copy()
            numeric_cols_to_format = ['ค่าพยากรณ์', 'ค่าจริง', 'ค่าความแตกต่าง', 'Error (%)']
            formatter = {col: "{:,.2f}" for col in numeric_cols_to_format}
            st.dataframe(display_df.style.format(formatter, na_rep="-"), use_container_width=True, hide_index=True, height=400)
            valid_results = results_df.dropna(subset=['ค่าจริง'])
            mse, rmse, mape, r2 = [np.nan] * 4
            if not valid_results.empty:
                actual, forecast_vals = valid_results['ค่าจริง'], valid_results['ค่าพยากรณ์']
                mse = np.mean((actual - forecast_vals)**2)
                rmse = np.sqrt(mse)
                r2 = r2_score(actual, forecast_vals)
                if not actual[actual != 0].empty:
                    mape = np.mean(np.abs((actual - forecast_vals) / actual)[actual != 0]) * 100
            st.write("ประเมินความคลาดเคลื่อน (Accuracy Metrics)")
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("MSE", f"{mse:,.2f}" if pd.notna(mse) else "N/A")
            m2.metric("RMSE", f"{rmse:,.2f}" if pd.notna(rmse) else "N/A")
            m3.metric("MAPE", f"{mape:,.2f}%" if pd.notna(mape) else "N/A")
            m4.metric("R-squared", f"{r2:,.2f}" if pd.notna(r2) else "N/A")

# ------------------ โครงสร้างหลักของแอป ------------------
st.title("🦉 AI Mint Forecast Dashboard")
df_dist_check = load_data("DATA_Distribution_Sum_Center.xlsx", 'Sheet1', 'date')
df_ret_check = load_data("DATA_Exchange_Sum_Center.xlsx", 'Sheet2', 'dc')
if df_dist_check is None or df_ret_check is None: st.stop()

# --- Sidebar ---
st.sidebar.header("⚙️ ตัวกรอง")
center_options = sorted(list(set(df_dist_check['PLANTNAME'].unique())|set(df_ret_check['PLANTNAME'].unique())))
selected_center = st.sidebar.selectbox("เลือกศูนย์ (Center)", center_options, index=(center_options.index('ทั่วประเทศ') if 'ทั่วประเทศ' in center_options else 0))
coin_display_map = {'10 บาท': '10.0', '5 บาท': '5.0', '2 บาท': '2.0', '1 บาท': '1.0', '50 สตางค์': '0.5', '25 สตางค์': '0.25', 'รวมทั้งหมด': 'รวม'}
selected_coin_display = st.sidebar.selectbox("เลือกชนิดราคา", list(coin_display_map.keys()), index=list(coin_display_map.keys()).index('รวมทั้งหมด'))
forecast_periods = st.sidebar.number_input("พยากรณ์ล่วงหน้า (เดือน)", 1, 120, 24, 1)
df_dist_check['fiscal_year'] = df_dist_check['ds'].apply(get_fiscal_year)
historical_years = sorted(df_dist_check['fiscal_year'].unique(), reverse=True)
last_historical_date = df_dist_check['ds'].max()
future_dates = pd.date_range(start=last_historical_date, periods=forecast_periods + 1, freq='M')
future_years = sorted(list(set([get_fiscal_year(d) for d in future_dates])), reverse=True)
combined_years = sorted(list(set(historical_years + future_years)), reverse=True)
fiscal_year_options_sidebar = ["ทั้งหมด"] + combined_years
selected_fiscal_year_sidebar = st.sidebar.selectbox("เลือกปีงบประมาณ", fiscal_year_options_sidebar)
confidence_options = {'90%': 0.90, '95%': 0.95, '99%': 0.99}
selected_confidence_label = st.sidebar.selectbox("ระดับความเชื่อมั่น", list(confidence_options.keys()), index=1)
return_type_map = {'จ่ายได้': 'G', 'ชำรุด': 'B', 'รวม': 'A'}
st.sidebar.subheader("ตัวกรองการประมาณการ")
selected_return_type_estimation = st.sidebar.selectbox("เทียบกับการรับคืนแบบ", list(return_type_map.keys()), index=2)
return_coin_map = {'10.0': '10', '5.0': '5', '2.0': '2', '1.0': '1', '0.5': '0.50', '0.25': '0.25', 'รวม': 'total'}

# --- ส่วนด้านการประมาณการ ---
st.header("📈 ผลการประมาณการความต้องการเหรียญ บต.")
st.info('**หมายเหตุ:** อยู่ระหว่างการพัฒนา')
with st.spinner("กำลังคำนวณผลการประมาณการสำหรับทุกชนิดราคา..."):
    all_net_forecasts = []
    for coin_name, coin_val in coin_display_map.items():
        forecast_dist, _, error_dist = forecasting_fn('dist', selected_center, coin_val, PARAMS_DISTRIBUTION, forecast_periods, confidence_options[selected_confidence_label])
        return_prefix = return_coin_map[coin_val]
        return_suffix = return_type_map[selected_return_type_estimation]
        ret_col = f"{return_prefix}_{return_suffix}" if return_prefix == 'total' else f"{return_prefix}{return_suffix}"
        forecast_ret, _, error_ret = forecasting_fn('ret', selected_center, ret_col, PARAMS_RETURNS, forecast_periods, confidence_options[selected_confidence_label])
        if error_dist is None and error_ret is None:
            future_df = pd.merge(forecast_dist[['ds', 'yhat']], forecast_ret[['ds', 'yhat']], on='ds', how='outer').rename(columns={'yhat_x': 'yhat_dist', 'yhat_y': 'yhat_ret'}).fillna(0)
            future_net_df = future_df[future_df['ds'] > df_dist_check['ds'].max()]
            if not future_net_df.empty:
                future_net_df['net_forecast'] = future_net_df['yhat_dist'] - future_net_df['yhat_ret']
                future_net_df['ชนิดราคา'] = coin_name
                all_net_forecasts.append(future_net_df)
if all_net_forecasts:
    final_net_df = pd.concat(all_net_forecasts)
    final_net_df['fiscal_year'] = final_net_df['ds'].apply(get_fiscal_year)
    
    if selected_fiscal_year_sidebar != "ทั้งหมด":
        final_net_df_display = final_net_df[final_net_df['fiscal_year'] == selected_fiscal_year_sidebar]
    else:
        final_net_df_display = final_net_df

    total_net_df = final_net_df_display.groupby('ds')['net_forecast'].sum().reset_index()
    total_net_df['ds_str'] = total_net_df['ds'].apply(format_thai_date_short)
    st.subheader("ภาพรวมความต้องการเหรียญสุทธิทุกชนิดราคา")
    if not total_net_df.empty:
        start_date, end_date = total_net_df['ds'].min(), total_net_df['ds'].max()
        st.write(f"แสดงผลช่วง: **{format_month_year_thai(start_date)}** ถึง **{format_month_year_thai(end_date)}**")
    fig_total = px.bar(total_net_df, x='ds_str', y='net_forecast', title=f"ภาพรวมความต้องการเหรียญสุทธิ ที่: {selected_center}", template='plotly_white')
    fig_total.update_traces(marker_color='#0052CC', hovertemplate='<b>เดือน: %{x}</b><br><b>ความต้องการสุทธิ: %{y:,.2f}</b><extra></extra>')
    fig_total.update_layout(yaxis_title="จำนวน (ล้านเหรียญ)", xaxis_title="เดือน", xaxis_tickangle=-90)
    st.plotly_chart(fig_total, use_container_width=True)
    st.subheader("รายละเอียดความต้องการเหรียญสุทธิรายชนิดราคา")
    color_map = {'10 บาท': '#FFC107', '5 บาท': '#28a745', '2 บาท': '#6f42c1', '1 บาท': '#0d6efd', '50 สตางค์': '#fd7e14', '25 สตางค์': '#dc3545'}
    estimation_cols = st.columns(3)
    col_index = 0
    for coin_name in [c for c in coin_display_map if c != "รวมทั้งหมด"]:
        with estimation_cols[col_index % 3]:
            coin_df = final_net_df_display[final_net_df_display['ชนิดราคา'] == coin_name]
            coin_df['ds_str'] = coin_df['ds'].apply(format_thai_date_short)
            if not coin_df.empty:
                fig_net = px.bar(coin_df, x='ds_str', y='net_forecast', title=f"{coin_name}", template='plotly_white', custom_data=['yhat_dist', 'yhat_ret'])
                fig_net.update_traces(marker_color=color_map.get(coin_name, '#888888'), hovertemplate='<b>เดือน: %{x}</b><br><b>ความต้องการสุทธิ: %{y:,.2f}</b><br><br>จ่ายแลก: %{customdata[0]:,.2f}<br>รับคืน: %{customdata[1]:,.2f}<extra></extra>')
                fig_net.update_layout(yaxis_title="", xaxis_title="", showlegend=False, xaxis_tickangle=-90)
                st.plotly_chart(fig_net, use_container_width=True)
        col_index += 1

st.divider()
st.header("🔎 การพยากรณ์รายชนิดราคา")
col_dist, col_ret = st.columns(2)
with col_dist:
    dist_value_col = coin_display_map[selected_coin_display]
    forecast_dist, df_filtered_dist, error_dist = forecasting_fn('dist', selected_center, dist_value_col, PARAMS_DISTRIBUTION, forecast_periods, confidence_options[selected_confidence_label])
    if error_dist:
        st.error(f"**การจ่ายแลก:** {error_dist}")
    elif forecast_dist is not None:
        forecast_dist['fiscal_year'] = forecast_dist['ds'].apply(get_fiscal_year)
        df_filtered_dist['fiscal_year'] = df_filtered_dist['ds'].apply(get_fiscal_year)
        forecast_to_display = forecast_dist[forecast_dist['fiscal_year'] == selected_fiscal_year_sidebar] if selected_fiscal_year_sidebar != "ทั้งหมด" else forecast_dist
        df_to_display = df_filtered_dist[df_filtered_dist['fiscal_year'] == selected_fiscal_year_sidebar] if selected_fiscal_year_sidebar != "ทั้งหมด" else df_filtered_dist
        display_forecast_output(st.container(), "📊 การจ่ายแลก", forecast_to_display, df_to_display, selected_confidence_label, f"{selected_coin_display} @ {selected_center}")
with col_ret:
    st.subheader("📊 การรับคืน")
    selected_return_type_display = st.radio("เลือกสภาพเหรียญ", list(return_type_map.keys()), horizontal=True, key="return_type_display")
    ret_val = coin_display_map[selected_coin_display]
    ret_col_display = f"{return_coin_map[ret_val]}_{return_type_map[selected_return_type_display]}" if return_coin_map[ret_val] == 'total' else f"{return_coin_map[ret_val]}{return_type_map[selected_return_type_display]}"
    forecast_ret, df_filtered_ret, error_ret = forecasting_fn('ret', selected_center, ret_col_display, PARAMS_RETURNS, forecast_periods, confidence_options[selected_confidence_label])
    if error_ret:
        st.error(f"**การรับคืน:** {error_ret}")
    elif forecast_ret is not None:
        forecast_ret['fiscal_year'] = forecast_ret['ds'].apply(get_fiscal_year)
        df_filtered_ret['fiscal_year'] = df_filtered_ret['ds'].apply(get_fiscal_year)
        forecast_to_display = forecast_ret[forecast_ret['fiscal_year'] == selected_fiscal_year_sidebar] if selected_fiscal_year_sidebar != "ทั้งหมด" else forecast_ret
        df_to_display = df_filtered_ret[df_filtered_ret['fiscal_year'] == selected_fiscal_year_sidebar] if selected_fiscal_year_sidebar != "ทั้งหมด" else df_filtered_ret
        display_forecast_output(st.container(), "", forecast_to_display, df_to_display, selected_confidence_label, f"{selected_coin_display} ({selected_return_type_display}) @ {selected_center}")
st.divider()
st.subheader("ทีมผู้พัฒนา")
dev_cols = st.columns(3)
with dev_cols[0]:
    st.markdown("<div style='padding: 10px; border: 1px solid #E0E0E0; border-radius: 10px; background-color: #FFFFFF; height: 100%; text-align: center;'><b>นางสาวออมสิน จินดากุลเวศ</b><br><i>นักวิชาการคลังชำนาญการ</i></div>", unsafe_allow_html=True)
with dev_cols[1]:
    st.markdown("<div style='padding: 10px; border: 1px solid #E0E0E0; border-radius: 10px; background-color: #FFFFFF; height: 100%; text-align: center;'><b>นายธนาคาร จักรธำรงค์</b><br><i>นักวิชาการคลังปฏิบัติการ</i></div>", unsafe_allow_html=True)
with dev_cols[2]:
    st.markdown("<div style='padding: 10px; border: 1px solid #E0E0E0; border-radius: 10px; background-color: #FFFFFF; height: 100%; text-align: center;'><b>นางสาวจารุวรรณ ตาลดี</b><br><i>เจ้าพนักงานดูเงินชำนาญงาน</i></div>", unsafe_allow_html=True)

