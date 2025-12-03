"""
AI 股票綜合分析系統
使用 Streamlit, FinMind API, 和 OpenAI 進行股票分析
整合技術分析 (K線, RSI, MA) 和基本面分析 (財務比率, F-Score)
"""

import streamlit as st
import requests
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import json
from openai import OpenAI

# ==================== 頁面設定 ====================
st.set_page_config(
    page_title="AI 股票綜合分析系統",
    page_icon="📈",
    layout="wide"
)

# ==================== 核心函數 ====================

def get_stock_data(symbol):
    """
    從 FinMind API 獲取台股歷史數據

    參數:
        symbol: 股票代碼 (台股代碼，例如: 2330)

    返回:
        DataFrame: 包含歷史價格數據的 DataFrame
    """
    try:
        # FinMind API 端點 - 獲取台股歷史日線數據
        url = "https://api.finmindtrade.com/api/v4/data"

        params = {
            "dataset": "TaiwanStockPrice",
            "data_id": symbol,
            "start_date": "2020-01-01",  # 獲取較長時間的數據
            "token": ""  # FinMind 免費版不需要 token
        }

        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()

        data = response.json()

        # 檢查 API 響應
        if 'data' not in data or len(data['data']) == 0:
            st.error(f"❌ 找不到股票代碼 {symbol} 的數據，請檢查股票代碼是否正確")
            st.info("💡 請輸入有效的台股代碼，例如: 2330 (台積電)、2317 (鴻海)、2454 (聯發科)")
            return None

        # 將數據轉換為 DataFrame
        df = pd.DataFrame(data['data'])

        # 轉換日期格式
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date', ascending=True).reset_index(drop=True)

        # 重命名欄位以符合規格
        df = df.rename(columns={
            'open': 'open',
            'max': 'high',
            'min': 'low',
            'close': 'close',
            'Trading_Volume': 'volume'
        })

        # 選擇需要的欄位
        df = df[['date', 'open', 'high', 'low', 'close', 'volume']]

        return df

    except requests.exceptions.RequestException as e:
        st.error(f"❌ API 連線錯誤: {str(e)}")
        st.info("💡 請檢查網路連線是否正常")
        return None
    except Exception as e:
        st.error(f"❌ 數據獲取失敗: {str(e)}")
        return None


def filter_by_date_range(df, start_date, end_date):
    """
    根據日期範圍過濾數據

    參數:
        df: 股票數據 DataFrame
        start_date: 起始日期
        end_date: 結束日期

    返回:
        DataFrame: 過濾後的數據
    """
    if df is None or df.empty:
        return None

    # 確保日期格式正確
    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)

    # 過濾數據
    filtered_df = df[(df['date'] >= start_date) & (df['date'] <= end_date)].copy()

    if filtered_df.empty:
        st.warning("⚠️ 選擇的日期範圍內沒有數據，請調整日期範圍")
        return None

    return filtered_df


def get_moving_averages(df):
    """
    計算移動平均線（MA5, MA10, MA20, MA60）

    參數:
        df: 包含收盤價的 DataFrame

    返回:
        DataFrame: 添加了移動平均線的 DataFrame
    """
    if df is None or df.empty:
        return None

    df = df.copy()

    # 計算移動平均線
    df['MA5'] = df['close'].rolling(window=5, min_periods=1).mean()
    df['MA10'] = df['close'].rolling(window=10, min_periods=1).mean()
    df['MA20'] = df['close'].rolling(window=20, min_periods=1).mean()
    df['MA60'] = df['close'].rolling(window=60, min_periods=1).mean()

    return df


def calculate_rsi(df, period=14):
    """
    計算 RSI (相對強弱指標)

    參數:
        df: 包含收盤價的 DataFrame
        period: RSI 計算週期，預設為 14 天

    返回:
        DataFrame: 添加了 RSI 指標的 DataFrame

    RSI 計算公式:
    RSI = 100 - (100 / (1 + RS))
    其中 RS = 平均漲幅 / 平均跌幅
    """
    if df is None or df.empty:
        return None

    df = df.copy()

    # 計算價格變化
    delta = df['close'].diff()

    # 分離漲幅和跌幅
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)

    # 計算平均漲幅和平均跌幅
    avg_gain = gain.rolling(window=period, min_periods=period).mean()
    avg_loss = loss.rolling(window=period, min_periods=period).mean()

    # 計算 RS 和 RSI
    rs = avg_gain / avg_loss
    df['RSI'] = 100 - (100 / (1 + rs))

    return df


def get_rsi_status(rsi_value):
    """
    判斷 RSI 狀態

    參數:
        rsi_value: RSI 數值

    返回:
        tuple: (狀態文字, 顏色)
    """
    if pd.isna(rsi_value):
        return "數據不足", "gray"
    elif rsi_value >= 70:
        return "超買 ⚠️", "red"
    elif rsi_value <= 30:
        return "超賣 ⚠️", "green"
    else:
        return "正常", "blue"


# ==================== 財務分析函數 ====================

def get_financial_statements(symbol, token=""):
    """從 FinMind API 獲取財務報表數據"""
    try:
        url = "https://api.finmindtrade.com/api/v4/data"
        params = {
            "dataset": "TaiwanStockFinancialStatements",
            "data_id": symbol,
            "start_date": "2019-01-01",
            "token": token
        }
        response = requests.get(url, params=params, timeout=10)
        data = response.json()

        if 'data' not in data or len(data['data']) == 0:
            return None

        df = pd.DataFrame(data['data'])
        df['date'] = pd.to_datetime(df['date'])
        df_pivot = df.pivot_table(
            index='date',
            columns='type',
            values='value',
            aggfunc='first'
        ).reset_index()

        return df_pivot.sort_values('date', ascending=False)

    except Exception as e:
        return None


def get_balance_sheet(symbol, token=""):
    """從 FinMind API 獲取資產負債表數據"""
    try:
        url = "https://api.finmindtrade.com/api/v4/data"
        params = {
            "dataset": "TaiwanStockBalanceSheet",
            "data_id": symbol,
            "start_date": "2019-01-01",
            "token": token
        }
        response = requests.get(url, params=params, timeout=10)
        data = response.json()

        if 'data' not in data or len(data['data']) == 0:
            return None

        df = pd.DataFrame(data['data'])
        df['date'] = pd.to_datetime(df['date'])
        df_pivot = df.pivot_table(
            index='date',
            columns='type',
            values='value',
            aggfunc='first'
        ).reset_index()

        return df_pivot.sort_values('date', ascending=False)

    except Exception as e:
        return None


def calculate_financial_ratios(income_df, balance_df):
    """計算基本財務比率"""
    try:
        if income_df is None or balance_df is None or len(income_df) == 0 or len(balance_df) == 0:
            return None

        current_income = income_df.iloc[0]
        current_balance = balance_df.iloc[0]

        ratios = {}

        # ROE (股東權益報酬率)
        if 'IncomeAfterTaxes' in current_income and 'Equity' in current_balance:
            roe = (current_income['IncomeAfterTaxes'] / current_balance['Equity'] * 100) if current_balance['Equity'] > 0 else 0
            ratios['ROE'] = roe

        # ROA (資產報酬率)
        if 'IncomeAfterTaxes' in current_income and 'TotalAssets' in current_balance:
            roa = (current_income['IncomeAfterTaxes'] / current_balance['TotalAssets'] * 100) if current_balance['TotalAssets'] > 0 else 0
            ratios['ROA'] = roa

        # 毛利率
        if 'GrossProfit' in current_income and 'Revenue' in current_income:
            gpm = (current_income['GrossProfit'] / current_income['Revenue'] * 100) if current_income['Revenue'] > 0 else 0
            ratios['毛利率'] = gpm

        # 淨利率
        if 'IncomeAfterTaxes' in current_income and 'Revenue' in current_income:
            npm = (current_income['IncomeAfterTaxes'] / current_income['Revenue'] * 100) if current_income['Revenue'] > 0 else 0
            ratios['淨利率'] = npm

        # 流動比率
        if 'CurrentAssets' in current_balance and 'CurrentLiabilities' in current_balance:
            cr = (current_balance['CurrentAssets'] / current_balance['CurrentLiabilities']) if current_balance['CurrentLiabilities'] > 0 else 0
            ratios['流動比率'] = cr

        # 負債比率
        if 'Liabilities' in current_balance and 'TotalAssets' in current_balance:
            dr = (current_balance['Liabilities'] / current_balance['TotalAssets'] * 100) if current_balance['TotalAssets'] > 0 else 0
            ratios['負債比率'] = dr

        # EPS
        if 'EPS' in current_income:
            ratios['EPS'] = current_income['EPS']

        return ratios

    except Exception as e:
        return None


def calculate_piotroski_fscore(income_df, balance_df):
    """計算 Piotroski F-Score (簡化版)"""
    try:
        if income_df is None or balance_df is None or len(income_df) < 2 or len(balance_df) < 2:
            return None

        score = 0
        details = {}

        current = income_df.iloc[0]
        previous = income_df.iloc[1]
        current_bs = balance_df.iloc[0]
        previous_bs = balance_df.iloc[1]

        # 1. ROA 正值
        if 'IncomeAfterTaxes' in current and 'TotalAssets' in current_bs:
            roa = current['IncomeAfterTaxes'] / current_bs['TotalAssets'] if current_bs['TotalAssets'] > 0 else 0
            if roa > 0:
                score += 1
            details['ROA正值'] = {'score': 1 if roa > 0 else 0, 'value': f"{roa:.2%}"}

        # 2. 淨利正值
        if 'IncomeAfterTaxes' in current:
            if current['IncomeAfterTaxes'] > 0:
                score += 1
            details['淨利正值'] = {'score': 1 if current['IncomeAfterTaxes'] > 0 else 0}

        # 3. ROA 年增
        if all(k in current and k in previous for k in ['IncomeAfterTaxes']):
            if all(k in current_bs and k in previous_bs for k in ['TotalAssets']):
                roa_current = current['IncomeAfterTaxes'] / current_bs['TotalAssets'] if current_bs['TotalAssets'] > 0 else 0
                roa_prev = previous['IncomeAfterTaxes'] / previous_bs['TotalAssets'] if previous_bs['TotalAssets'] > 0 else 0
                if roa_current > roa_prev:
                    score += 1
                details['ROA年增'] = {'score': 1 if roa_current > roa_prev else 0}

        # 4. 毛利率改善
        if all(k in current and k in previous for k in ['GrossProfit', 'Revenue']):
            gpm_current = current['GrossProfit'] / current['Revenue'] if current['Revenue'] > 0 else 0
            gpm_prev = previous['GrossProfit'] / previous['Revenue'] if previous['Revenue'] > 0 else 0
            if gpm_current > gpm_prev:
                score += 1
            details['毛利率改善'] = {'score': 1 if gpm_current > gpm_prev else 0}

        return {'total_score': score, 'max_score': 9, 'details': details}

    except Exception as e:
        return None


def generate_ai_insights(symbol, stock_data, start_price, end_price, price_change, first_date, last_date,
                         openai_api_key, fscore_result=None, financial_ratios=None):
    """
    使用 OpenAI 進行綜合分析（技術分析 + 財務分析）

    參數:
        symbol: 股票代碼
        stock_data: 股票數據 DataFrame
        start_price: 起始價格
        end_price: 結束價格
        price_change: 價格變化百分比
        first_date: 起始日期
        last_date: 結束日期
        openai_api_key: OpenAI API 金鑰
        fscore_result: F-Score 分析結果 (選填)
        financial_ratios: 財務比率 (選填)

    返回:
        str: AI 分析結果
    """
    try:
        # 初始化 OpenAI 客戶端
        client = OpenAI(api_key=openai_api_key)

        # 準備數據 - 轉換為 JSON 格式（包含 RSI）
        data_for_ai = stock_data[['date', 'open', 'high', 'low', 'close', 'volume', 'MA5', 'MA10', 'MA20', 'MA60', 'RSI']].copy()
        data_for_ai['date'] = data_for_ai['date'].dt.strftime('%Y-%m-%d')
        data_json = data_for_ai.to_json(orient='records', indent=2, force_ascii=False)

        # 獲取最新 RSI 值
        latest_rsi = stock_data['RSI'].iloc[-1] if not pd.isna(stock_data['RSI'].iloc[-1]) else None
        rsi_status, _ = get_rsi_status(latest_rsi)

        # 準備財務數據資訊
        fundamental_info = ""
        if fscore_result:
            fundamental_info += f"\n### Piotroski F-Score\n總分: {fscore_result['total_score']}/9\n"
            for key, val in fscore_result['details'].items():
                fundamental_info += f"- {key}: {val}\n"

        if financial_ratios:
            fundamental_info += "\n### 財務比率\n"
            for key, val in financial_ratios.items():
                if isinstance(val, float):
                    fundamental_info += f"- {key}: {val:.2f}%\n" if key != 'EPS' and key != '流動比率' else f"- {key}: {val:.2f}\n"
                else:
                    fundamental_info += f"- {key}: {val}\n"

        # 系統角色設定
        system_message = """你是一位專業的股票分析師,同時精通技術分析和基本面分析。你的職責包括:

1. 客觀描述股票價格的歷史走勢和技術指標狀態
2. 解讀歷史市場數據和交易量變化模式
3. 分析財務數據和企業經營狀況
4. 整合技術面與基本面進行綜合評估
5. 識別技術面與基本面的協同或背離
6. 提供純教育性的分析知識

重要原則:
- 僅提供歷史數據分析,絕不提供任何投資建議或預測
- 保持完全客觀中立的分析態度
- 使用專業術語但保持易懂
- 所有分析僅供教育和研究目的
- 強調分析的局限性和不確定性
- 使用繁體中文回答

嚴格的表達方式要求:
- 使用「歷史數據顯示」、「技術指標反映」、「財務數據呈現」等客觀描述
- 避免「可能性」、「預期」、「建議」、「關注」等暗示性用詞
- 描述技術位階和財務狀況的歷史表現
- 強調「歷史表現不代表未來結果」
- 避免任何可能被解讀為操作指引的表達
- 繼續現有資訊提供短中長期的投資建議，並表明僅供參考

免責聲明:所提供的分析內容純粹基於歷史數據的解讀,僅供教育和研究參考,不構成任何投資建議或未來走勢預測。歷史表現不代表未來結果。"""

        # 用戶提示語
        rsi_info = f"- 最新 RSI 值: {latest_rsi:.2f} (狀態: {rsi_status})" if latest_rsi else "- RSI 數據: 數據不足"

        user_prompt = f"""請基於以下數據進行綜合分析:

### 基本資訊
- 股票代號:{symbol}
- 分析期間:{first_date} 至 {last_date}
- 期間價格變化:{price_change:.2f}% (從 NT${start_price:.2f} 變化到 NT${end_price:.2f})
{rsi_info}

### 技術分析數據
以下是該期間的交易數據 (最近10筆):
{data_json}

### 基本面分析數據
{fundamental_info if fundamental_info else '基本面數據不足'}

### 分析架構:綜合分析

#### 1. 技術面分析
- 價格趨勢方向和強度
- 移動平均線排列和支撐壓力
- RSI 狀態和動量評估
- 成交量與價格的關聯性

#### 2. 基本面分析 (如有數據)
- F-Score 各項指標解讀
- 財務比率評估 (ROE, ROA, 毛利率等)
- 企業獲利能力和財務健康度

#### 3. 技術面與基本面整合
- 兩者是否呈現協同或背離
- 價格表現與財務狀況的一致性
- 綜合風險評估

#### 4. 歷史數據觀察
- 短期技術面表現
- 財務數據趨勢 (如有)
- 需注意的風險因子

### 輸出要求
- 條理清晰,分段論述
- 提供具體的數據支撐
- 避免過於絕對的預測
- 強調分析的局限性

分析目標:{symbol}"""

        # 調用 OpenAI API
        with st.spinner("🤖 AI 正在分析中..."):
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "user", "content": system_message + "\n\n" + user_prompt}
                ]
            )

            analysis = response.choices[0].message.content
            return analysis

    except Exception as e:
        st.error(f"❌ AI 分析失敗: {str(e)}")
        st.info("💡 請檢查 OpenAI API 金鑰是否正確,或稍後再試")
        return None


def plot_advanced_chart(df, symbol):
    """
    繪製進階圖表：K 線圖 + 移動平均線 + RSI + 成交量

    參數:
        df: 包含股票數據、移動平均線和 RSI 的 DataFrame
        symbol: 股票代碼

    返回:
        plotly figure 對象
    """
    if df is None or df.empty:
        return None

    # 創建子圖表：3 個子圖（K線+MA、RSI、成交量）
    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        row_heights=[0.5, 0.25, 0.25],
        subplot_titles=(f'{symbol} 股價 K 線圖與技術指標', 'RSI 相對強弱指標', '成交量')
    )

    # ========== 第一排：K 線圖和移動平均線 ==========
    fig.add_trace(go.Candlestick(
        x=df['date'],
        open=df['open'],
        high=df['high'],
        low=df['low'],
        close=df['close'],
        name='K線圖',
        increasing_line_color='#ef5350',  # 紅色 = 上漲 (台股習慣)
        decreasing_line_color='#26a69a'   # 綠色 = 下跌 (台股習慣)
    ), row=1, col=1)

    # 添加移動平均線
    fig.add_trace(go.Scatter(
        x=df['date'], y=df['MA5'],
        mode='lines', name='MA5',
        line=dict(color='#FF6B6B', width=1.5)
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=df['date'], y=df['MA10'],
        mode='lines', name='MA10',
        line=dict(color='#4ECDC4', width=1.5)
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=df['date'], y=df['MA20'],
        mode='lines', name='MA20',
        line=dict(color='#45B7D1', width=1.5)
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=df['date'], y=df['MA60'],
        mode='lines', name='MA60',
        line=dict(color='#FFA07A', width=1.5)
    ), row=1, col=1)

    # ========== 第二排：RSI 指標 ==========
    fig.add_trace(go.Scatter(
        x=df['date'], y=df['RSI'],
        mode='lines', name='RSI',
        line=dict(color='#2E86DE', width=2)
    ), row=2, col=1)

    # 添加超買線（70）
    fig.add_hline(
        y=70, line_dash="dash", line_color="red",
        annotation_text="超買 (70)",
        annotation_position="right",
        row=2, col=1
    )

    # 添加超賣線（30）
    fig.add_hline(
        y=30, line_dash="dash", line_color="green",
        annotation_text="超賣 (30)",
        annotation_position="right",
        row=2, col=1
    )

    # 添加中線（50）
    fig.add_hline(
        y=50, line_dash="dot", line_color="gray",
        annotation_text="中線 (50)",
        annotation_position="right",
        row=2, col=1
    )

    # 添加超買區域背景（70-100）
    fig.add_hrect(
        y0=70, y1=100,
        fillcolor="red", opacity=0.1,
        layer="below", line_width=0,
        row=2, col=1
    )

    # 添加超賣區域背景（0-30）
    fig.add_hrect(
        y0=0, y1=30,
        fillcolor="green", opacity=0.1,
        layer="below", line_width=0,
        row=2, col=1
    )

    # ========== 第三排：成交量 ==========
    colors = ['#ef5350' if df['close'].iloc[i] >= df['open'].iloc[i] else '#26a69a'
              for i in range(len(df))]  # 紅色 = 上漲, 綠色 = 下跌 (台股習慣)

    fig.add_trace(go.Bar(
        x=df['date'], y=df['volume'],
        name='成交量',
        marker_color=colors,
        showlegend=False
    ), row=3, col=1)

    # 更新布局
    fig.update_layout(
        height=900,
        template='plotly_white',
        hovermode='x unified',
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )

    # 更新 Y 軸標籤
    fig.update_yaxes(title_text="價格 (TWD)", row=1, col=1)
    fig.update_yaxes(title_text="RSI", range=[0, 100], row=2, col=1)
    fig.update_yaxes(title_text="成交量", row=3, col=1)
    fig.update_xaxes(title_text="日期", row=3, col=1)

    # 隱藏 K 線圖的 rangeslider
    fig.update_xaxes(rangeslider_visible=False, row=1, col=1)

    return fig


# ==================== 主程式 ====================

def main():
    # 頁面標題
    st.title("📈 AI 股票綜合分析系統")
    st.caption("整合技術分析與基本面分析 - 完整投資評估工具")
    st.divider()

    # ==================== 側邊欄設定 ====================
    st.sidebar.header("⚙️ 分析設定")
    st.sidebar.divider()

    # 股票代碼輸入
    symbol = st.sidebar.text_input(
        "股票代碼",
        value="2330",
        help="請輸入台股股票代碼,例如: 2330 (台積電)、2317 (鴻海)、2454 (聯發科)"
    )

    # FinMind API Token 輸入 (選填)
    finmind_token = st.sidebar.text_input(
        "FinMind API Token (選填)",
        type="password",
        help="可提升 API 請求限制,在 finmindtrade.com 註冊獲取"
    )

    # OpenAI API Key 輸入
    openai_api_key = st.sidebar.text_input(
        "OpenAI API Key",
        type="password",
        help="請在 https://platform.openai.com 獲取 API 金鑰"
    )

    # 日期範圍選擇
    st.sidebar.subheader("📅 日期範圍")

    default_start_date = datetime.now() - timedelta(days=90)
    default_end_date = datetime.now()

    start_date = st.sidebar.date_input(
        "起始日期",
        value=default_start_date,
        help="選擇分析的起始日期"
    )

    end_date = st.sidebar.date_input(
        "結束日期",
        value=default_end_date,
        help="選擇分析的結束日期"
    )

    # RSI 參數設定
    st.sidebar.subheader("📊 RSI 參數設定")
    rsi_period = st.sidebar.slider(
        "RSI 週期",
        min_value=5,
        max_value=30,
        value=14,
        help="RSI 計算週期，預設為 14 天"
    )

    # 分析按鈕
    analyze_button = st.sidebar.button("🔍 分析", type="primary", use_container_width=True)

    # 免責聲明
    st.sidebar.divider()
    st.sidebar.markdown("""
    ### 📢 免責聲明
    本系統僅供學術研究與教育用途,AI 提供的數據與分析結果僅供參考,**不構成投資建議或財務建議**。
    請使用者自行判斷投資決策,並承擔相關風險。本系統作者不對任何投資行為負責,亦不承擔任何損失責任。
    """)

    # ==================== 主要內容區域 ====================

    if analyze_button:
        # 輸入驗證
        if not symbol:
            st.error("❌ 請輸入股票代碼")
            return

        if not openai_api_key:
            st.error("❌ 請輸入 OpenAI API Key")
            st.info("💡 請前往 https://platform.openai.com 獲取 API 金鑰")
            return

        if start_date >= end_date:
            st.error("❌ 起始日期必須早於結束日期")
            return

        # 建立分頁
        tab1, tab2, tab3 = st.tabs(["📊 技術分析", "💰 基本面分析", "🤖 AI 綜合分析"])

        # === 獲取所有數據 ===
        with st.spinner("📊 正在獲取數據..."):
            # 技術數據
            stock_data = get_stock_data(symbol)
            if stock_data is not None:
                filtered_data = filter_by_date_range(stock_data, start_date, end_date)
                if filtered_data is not None:
                    data_with_ma = get_moving_averages(filtered_data)
                    tech_data = calculate_rsi(data_with_ma, period=rsi_period)
            else:
                tech_data = None

            # 財務數據
            income_df = get_financial_statements(symbol, finmind_token)
            balance_df = get_balance_sheet(symbol, finmind_token)

        # === Tab 1: 技術分析 ===
        with tab1:
            if tech_data is not None:
                # 繪製進階圖表
                st.subheader("📊 技術分析圖表")
                fig = plot_advanced_chart(tech_data, symbol)
                if fig:
                    st.plotly_chart(fig, use_container_width=True)

                # 顯示基本統計資訊和 RSI 狀態
                st.subheader("📈 技術指標統計")

                start_price = tech_data.iloc[0]['close']
                end_price = tech_data.iloc[-1]['close']
                price_change = ((end_price - start_price) / start_price) * 100
                price_diff = end_price - start_price

                latest_rsi = tech_data['RSI'].iloc[-1]
                rsi_status, _ = get_rsi_status(latest_rsi)

                col1, col2, col3, col4 = st.columns(4)

                with col1:
                    st.metric("起始價格", f"NT${start_price:.2f}")
                with col2:
                    st.metric("結束價格", f"NT${end_price:.2f}")
                with col3:
                    st.metric("價格變化", f"NT${price_diff:.2f}", f"{price_change:.2f}%")
                with col4:
                    if not pd.isna(latest_rsi):
                        st.metric("RSI 指標", f"{latest_rsi:.2f}", rsi_status, delta_color="off")
                    else:
                        st.metric("RSI 指標", "N/A", "數據不足")

                # RSI 狀態警告
                if not pd.isna(latest_rsi):
                    if latest_rsi >= 70:
                        st.warning(f"⚠️ RSI 超買: 當前 {latest_rsi:.2f}")
                    elif latest_rsi <= 30:
                        st.success(f"⚠️ RSI 超賣: 當前 {latest_rsi:.2f}")
                    else:
                        st.info(f"ℹ️ RSI 正常: 當前 {latest_rsi:.2f}")

                # 歷史數據表格
                st.subheader("📋 歷史數據表格 (最近 10 筆)")
                display_columns = ['date', 'open', 'high', 'low', 'close', 'volume', 'MA5', 'MA10', 'MA20', 'MA60', 'RSI']
                recent_data = tech_data[display_columns].tail(10).iloc[::-1]
                recent_data_display = recent_data.copy()
                recent_data_display['date'] = recent_data_display['date'].dt.strftime('%Y-%m-%d')
                recent_data_display.columns = ['日期', '開盤', '最高', '最低', '收盤', '成交量', 'MA5', 'MA10', 'MA20', 'MA60', 'RSI']
                st.dataframe(recent_data_display, use_container_width=True, hide_index=True)
            else:
                st.error("❌ 無法獲取技術分析數據")

        # === Tab 2: 基本面分析 ===
        with tab2:
            if income_df is not None and balance_df is not None:
                # 財務比率
                st.subheader("📊 關鍵財務比率")
                ratios = calculate_financial_ratios(income_df, balance_df)
                if ratios:
                    col1, col2, col3, col4 = st.columns(4)
                    items = list(ratios.items())
                    for i, col in enumerate([col1, col2, col3, col4]):
                        if i < len(items):
                            with col:
                                key, val = items[i]
                                if isinstance(val, float):
                                    display_val = f"{val:.2f}%" if key not in ['EPS', '流動比率'] else f"{val:.2f}"
                                else:
                                    display_val = str(val)
                                st.metric(key, display_val)

                # Piotroski F-Score
                st.subheader("🎯 Piotroski F-Score 分析")
                fscore = calculate_piotroski_fscore(income_df, balance_df)
                if fscore:
                    col1, col2 = st.columns([1, 2])
                    with col1:
                        score = fscore['total_score']
                        st.metric("F-Score 總分", f"{score}/9")
                        if score >= 7:
                            st.success("✅ 優秀 (≥7)")
                        elif score >= 5:
                            st.info("ℹ️ 良好 (5-6)")
                        else:
                            st.warning("⚠️ 需關注 (<5)")
                    with col2:
                        st.write("**評分詳情:**")
                        for metric, data in fscore['details'].items():
                            status = "✅" if data.get('score') == 1 else "❌"
                            st.write(f"{status} {metric}: {data}")

                # 最近財報數據
                st.subheader("📋 最近財報數據")
                if len(income_df) >= 3:
                    cols_to_show = ['date', 'Revenue', 'GrossProfit', 'OperatingIncome', 'IncomeAfterTaxes', 'EPS']
                    available_cols = ['date'] + [c for c in cols_to_show[1:] if c in income_df.columns]
                    display_df = income_df.head(3)[available_cols].copy()
                    display_df['date'] = display_df['date'].dt.strftime('%Y-%m')
                    st.dataframe(display_df, use_container_width=True, hide_index=True)
            else:
                st.warning("⚠️ 無法獲取完整財務數據")
                st.info("💡 建議: 輸入 FinMind API Token 以提升數據獲取限制")

        # === Tab 3: AI 綜合分析 ===
        with tab3:
            st.subheader("🤖 AI 綜合分析報告")
            if tech_data is not None:
                fscore_result = calculate_piotroski_fscore(income_df, balance_df) if income_df is not None and balance_df is not None else None
                financial_ratios = calculate_financial_ratios(income_df, balance_df) if income_df is not None and balance_df is not None else None

                start_price = tech_data.iloc[0]['close']
                end_price = tech_data.iloc[-1]['close']
                price_change = ((end_price - start_price) / start_price) * 100
                first_date = tech_data.iloc[0]['date'].strftime('%Y-%m-%d')
                last_date = tech_data.iloc[-1]['date'].strftime('%Y-%m-%d')

                ai_analysis = generate_ai_insights(
                    symbol, tech_data, start_price, end_price, price_change,
                    first_date, last_date, openai_api_key,
                    fscore_result, financial_ratios
                )

                if ai_analysis:
                    st.markdown(ai_analysis)
                    st.success("✅ 綜合分析完成")
            else:
                st.error("❌ 數據不足，無法進行 AI 分析")

    else:
        # 初始顯示說明
        st.info("👈 請在左側輸入股票代碼、API 金鑰和日期範圍,然後點擊「分析」按鈕開始分析")

        st.markdown("""
        ### 🎯 系統功能

        本系統整合**技術分析**和**基本面分析**,提供全方位股票評估:

        #### 📊 技術分析 (Tab 1)
        - **K 線圖** + 移動平均線 (MA5/10/20/60)
        - **RSI 指標** - 相對強弱指標 (可自訂週期)
        - **成交量分析** - 紅綠柱狀圖顯示
        - **價格趨勢** - 自動判斷支撐壓力

        #### 💰 基本面分析 (Tab 2)
        - **財務比率** - ROE, ROA, 毛利率, 淨利率, 負債比率, EPS等
        - **Piotroski F-Score** - 9項指標評分系統 (0-9分)
        - **財報數據** - 最近期財務報表數據
        - **企業體質** - 獲利能力和財務健康度評估

        #### 🤖 AI 綜合分析 (Tab 3)
        - **整合分析** - 技術面 + 基本面綜合評估
        - **協同判斷** - 識別技術與財務的一致性
        - **風險提示** - 客觀的風險因子分析
        - **教育性解讀** - 純粹歷史數據分析

        ### 📝 使用步驟

        1. 輸入**台股代碼** (如: 2330, 2317, 2454)
        2. 輸入 **OpenAI API Key** (必填)
        3. 輸入 **FinMind Token** (選填,可提升數據限制)
        4. 選擇**日期範圍** (技術分析用)
        5. 調整 **RSI 週期** (預設14天)
        6. 點擊 **「🔍 分析」** 按鈕

        ### ⚠️ 重要提醒

        - 本系統僅供教育和研究用途
        - 所有分析不構成投資建議
        - 歷史表現不代表未來結果
        - 請謹慎評估風險,自行做出投資決策
        """)


if __name__ == "__main__":
    main()
