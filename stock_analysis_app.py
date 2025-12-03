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
        df['revenue'] = pd.to_numeric(df['revenue'], errors='coerce')

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


def calculate_kd(df, n=9, m1=3, m2=3):
    """
    計算 KD 指標 (隨機指標)

    參數:
        df: 包含 high, low, close 的 DataFrame
        n: RSV 計算週期，預設 9
        m1: K 值平滑參數，預設 3
        m2: D 值平滑參數，預設 3

    返回:
        DataFrame: 添加了 K, D 指標的 DataFrame
    """
    if df is None or df.empty:
        return None

    df = df.copy()

    # 計算 RSV (未成熟隨機值)
    low_min = df['low'].rolling(window=n, min_periods=n).min()
    high_max = df['high'].rolling(window=n, min_periods=n).max()

    df['RSV'] = 100 * (df['close'] - low_min) / (high_max - low_min)

    # 計算 K 值 (RSV 的移動平均)
    df['K'] = df['RSV'].ewm(span=m1, adjust=False).mean()

    # 計算 D 值 (K 值的移動平均)
    df['D'] = df['K'].ewm(span=m2, adjust=False).mean()

    return df


def calculate_macd(df, fast=12, slow=26, signal=9):
    """
    計算 MACD 指標 (指數平滑異同移動平均線)

    參數:
        df: 包含收盤價的 DataFrame
        fast: 快速 EMA 週期，預設 12
        slow: 慢速 EMA 週期，預設 26
        signal: 信號線週期，預設 9

    返回:
        DataFrame: 添加了 MACD, Signal, Histogram 的 DataFrame
    """
    if df is None or df.empty:
        return None

    df = df.copy()

    # 計算快速和慢速 EMA
    ema_fast = df['close'].ewm(span=fast, adjust=False).mean()
    ema_slow = df['close'].ewm(span=slow, adjust=False).mean()

    # MACD 線 = 快速 EMA - 慢速 EMA
    df['MACD'] = ema_fast - ema_slow

    # 信號線 = MACD 的 EMA
    df['MACD_Signal'] = df['MACD'].ewm(span=signal, adjust=False).mean()

    # 柱狀圖 = MACD - 信號線
    df['MACD_Hist'] = df['MACD'] - df['MACD_Signal']

    return df


def calculate_willr(df, period=14):
    """
    計算威廉指標 (Williams %R)

    參數:
        df: 包含 high, low, close 的 DataFrame
        period: 計算週期，預設 14

    返回:
        DataFrame: 添加了 WillR 指標的 DataFrame

    威廉指標公式:
    %R = -100 * (最高價 - 收盤價) / (最高價 - 最低價)
    """
    if df is None or df.empty:
        return None

    df = df.copy()

    # 計算週期內的最高價和最低價
    high_max = df['high'].rolling(window=period, min_periods=period).max()
    low_min = df['low'].rolling(window=period, min_periods=period).min()

    # 計算威廉指標
    df['WillR'] = -100 * (high_max - df['close']) / (high_max - low_min)

    return df


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
        df['value'] = pd.to_numeric(df['value'], errors='coerce')
        df['date'] = pd.to_datetime(df['date'])
        df_pivot = df.pivot_table(
            index='date',
            columns='type',
            values='value',
            aggfunc='first'
        ).reset_index()

        return df_pivot.sort_values('date', ascending=False).reset_index(drop=True)

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
        df['value'] = pd.to_numeric(df['value'], errors='coerce')
        df['date'] = pd.to_datetime(df['date'])
        df_pivot = df.pivot_table(
            index='date',
            columns='type',
            values='value',
            aggfunc='first'
        ).reset_index()

        return df_pivot.sort_values('date', ascending=False).reset_index(drop=True)

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
        total_assets = current_balance.get('TotalAssets', current_balance.get('Assets'))
        equity = current_balance.get('Equity')
        liabilities = current_balance.get('Liabilities')
        current_assets = current_balance.get('CurrentAssets')
        current_liabilities = current_balance.get('CurrentLiabilities')

        # ROE (股東權益報酬率)
        if 'IncomeAfterTaxes' in current_income and equity:
            roe = (current_income['IncomeAfterTaxes'] / equity * 100) if equity > 0 else 0
            ratios['ROE'] = roe

        # ROA (資產報酬率)
        if 'IncomeAfterTaxes' in current_income and total_assets:
            roa = (current_income['IncomeAfterTaxes'] / total_assets * 100) if total_assets > 0 else 0
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
        if current_assets is not None and current_liabilities is not None:
            cr = (current_assets / current_liabilities) if current_liabilities > 0 else 0
            ratios['流動比率'] = cr

        # 負債比率
        if liabilities is not None and total_assets:
            dr = (liabilities / total_assets * 100) if total_assets > 0 else 0
            ratios['負債比率'] = dr

        # EPS
        if 'EPS' in current_income:
            ratios['EPS'] = current_income['EPS']

        return ratios

    except Exception as e:
        return None


def get_monthly_revenue(symbol, token=""):
    """
    獲取月營收數據

    參數:
        symbol: 股票代碼
        token: FinMind API Token

    返回:
        DataFrame: 月營收數據
    """
    try:
        url = "https://api.finmindtrade.com/api/v4/data"
        params = {
            "dataset": "TaiwanStockMonthRevenue",
            "data_id": symbol,
            "start_date": (datetime.now() - timedelta(days=730)).strftime('%Y-%m-%d'),  # 改為2年數據以計算年增率
            "token": token
        }
        response = requests.get(url, params=params, timeout=10)
        data = response.json()

        if 'data' not in data or len(data['data']) == 0:
            st.warning(f"⚠️ 月營收數據獲取狀況: {data.get('msg', '無數據')}")
            return None

        df = pd.DataFrame(data['data'])
        df['revenue'] = pd.to_numeric(df['revenue'], errors='coerce')

        # 檢查數據結構
        st.info(f"📊 月營收數據筆數: {len(df)} | 欄位: {list(df.columns)}")

        df['revenue_date'] = pd.to_datetime(df['revenue_month'])
        df = df.sort_values('revenue_date', ascending=False).reset_index(drop=True)

        # 反轉以計算正確的年增率和月增率
        df_reversed = df.iloc[::-1].copy()

        # 計算年增率和月增率（需要至少13個月數據才能計算年增率）
        df_reversed['yoy_growth'] = df_reversed['revenue'].pct_change(periods=12) * 100  # 年增率
        df_reversed['mom_growth'] = df_reversed['revenue'].pct_change(periods=1) * 100   # 月增率

        # 再反轉回最新在前
        df = df_reversed.iloc[::-1].copy()

        return df.head(6)  # 最近6個月

    except Exception as e:
        st.error(f"❌ 月營收數據錯誤: {str(e)}")
        return None


def calculate_eps_trend(income_df):
    """
    計算近5季 EPS 趨勢（含季增、年增）

    參數:
        income_df: 損益表 DataFrame

    返回:
        DataFrame: EPS 趨勢數據
    """
    try:
        if income_df is None or len(income_df) < 5:
            return None

        df = income_df.sort_values('date', ascending=False).head(5).copy().reset_index(drop=True)

        if 'EPS' not in df.columns:
            return None

        df['EPS'] = pd.to_numeric(df['EPS'], errors='coerce')

        prev_eps = df['EPS'].shift(-1)
        year_ago_eps = df['EPS'].shift(-4)

        df['QoQ'] = ((df['EPS'] - prev_eps) / prev_eps.abs() * 100).where(prev_eps != 0)
        df['YoY'] = ((df['EPS'] - year_ago_eps) / year_ago_eps.abs() * 100).where(year_ago_eps != 0)

        return df[['date', 'EPS', 'QoQ', 'YoY']]

    except Exception as e:
        return None


def calculate_pe_ratio(current_price, eps):
    """
    計算本益比 (P/E Ratio)

    參數:
        current_price: 當前股價
        eps: 每股盈餘

    返回:
        float: 本益比
    """
    if eps and eps > 0:
        return current_price / eps
    return None


def calculate_margin_trends(income_df):
    """
    計算近4季毛利率與營益率趨勢

    參數:
        income_df: 損益表 DataFrame

    返回:
        DataFrame: 毛利率與營益率數據
    """
    try:
        if income_df is None or len(income_df) < 4:
            return None

        df = income_df.sort_values('date', ascending=False).head(4).copy().reset_index(drop=True)
        df = df.iloc[::-1]  # 取最近4季，反轉為時間順序

        margin_data = []
        for _, row in df.iterrows():
            date = row.get('date')
            revenue = row.get('Revenue', 0)
            gross_profit = row.get('GrossProfit', 0)
            operating_income = row.get('OperatingIncome', 0)

            gross_margin = (gross_profit / revenue * 100) if revenue > 0 else 0
            operating_margin = (operating_income / revenue * 100) if revenue > 0 else 0

            margin_data.append({
                'date': date,
                '毛利率': gross_margin,
                '營益率': operating_margin
            })

        return pd.DataFrame(margin_data)

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
        total_assets_current = current_bs.get('TotalAssets', current_bs.get('Assets', 0))
        total_assets_prev = previous_bs.get('TotalAssets', previous_bs.get('Assets', 0))

        # 1. ROA 正值
        if 'IncomeAfterTaxes' in current and total_assets_current:
            roa = current['IncomeAfterTaxes'] / total_assets_current if total_assets_current > 0 else 0
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
            if total_assets_current is not None and total_assets_prev is not None:
                roa_current = current['IncomeAfterTaxes'] / total_assets_current if total_assets_current > 0 else 0
                roa_prev = previous['IncomeAfterTaxes'] / total_assets_prev if total_assets_prev > 0 else 0
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

        # 準備數據 - 轉換為 JSON 格式（包含所有技術指標）
        data_for_ai = stock_data[['date', 'open', 'high', 'low', 'close', 'volume',
                                    'MA5', 'MA10', 'MA20', 'MA60',
                                    'RSI', 'K', 'D', 'MACD', 'MACD_Signal', 'MACD_Hist', 'WillR']].copy()
        data_for_ai['date'] = data_for_ai['date'].dt.strftime('%Y-%m-%d')
        data_json = data_for_ai.tail(10).to_json(orient='records', indent=2, force_ascii=False)

        # 獲取最新指標值
        latest_rsi = stock_data['RSI'].iloc[-1] if not pd.isna(stock_data['RSI'].iloc[-1]) else None
        latest_k = stock_data['K'].iloc[-1] if not pd.isna(stock_data['K'].iloc[-1]) else None
        latest_d = stock_data['D'].iloc[-1] if not pd.isna(stock_data['D'].iloc[-1]) else None
        latest_macd = stock_data['MACD'].iloc[-1] if not pd.isna(stock_data['MACD'].iloc[-1]) else None
        latest_macd_signal = stock_data['MACD_Signal'].iloc[-1] if not pd.isna(stock_data['MACD_Signal'].iloc[-1]) else None
        latest_willr = stock_data['WillR'].iloc[-1] if not pd.isna(stock_data['WillR'].iloc[-1]) else None

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

        # 用戶提示語 - 準備所有指標的當前狀態
        indicators_info = f"""
**當前技術指標狀態:**
- RSI: {latest_rsi:.2f} ({rsi_status}) {f'- RSI > 70 超買' if latest_rsi and latest_rsi >= 70 else f'- RSI < 30 超賣' if latest_rsi and latest_rsi <= 30 else ''}
- KD 指標: K值 {latest_k:.2f}, D值 {latest_d:.2f} {f'- K > D 多頭' if latest_k and latest_d and latest_k > latest_d else '- K < D 空頭' if latest_k and latest_d else ''}
- MACD: {latest_macd:.4f}, Signal: {latest_macd_signal:.4f} {f'- MACD > Signal 多頭' if latest_macd and latest_macd_signal and latest_macd > latest_macd_signal else '- MACD < Signal 空頭' if latest_macd and latest_macd_signal else ''}
- 威廉指標: {latest_willr:.2f} {f'- 超買區域' if latest_willr and latest_willr > -20 else f'- 超賣區域' if latest_willr and latest_willr < -80 else ''}
"""

        user_prompt = f"""您是資深股票分析師，請根據以下數據進行**詳細專業的綜合分析**:

### 📊 基本資訊
- 股票代號: {symbol}
- 分析期間: {first_date} 至 {last_date}
- 期間價格變化: {price_change:.2f}% (從 NT${start_price:.2f} → NT${end_price:.2f})
- 當前價位: NT${end_price:.2f}

{indicators_info}

### 📈 技術分析數據 (最近10筆完整數據)
{data_json}

### 💰 基本面分析數據
{fundamental_info if fundamental_info else '基本面數據不足'}

---

## 🎯 請按照以下架構進行**詳細分析**:

### 1. 目前位階分析
- 當前價格在歷史區間的位置 (高點/低點/中間)
- 相對於各均線的位置關係
- 價格所處的關鍵支撐/壓力區間

### 2. 量價關係分析
- 成交量與價格變動的配合度
- 是否出現價漲量增、價跌量縮等健康型態
- 異常成交量的時間點和意義

### 3. 技術指標詳細解讀

#### 3.1 KD 指標分析
- K值與D值的當前數值和交叉狀態
- 是否處於超買(>80)或超賣(<20)區域
- 鈍化現象的判斷
- KD 指標給出的訊號

#### 3.2 MACD 指標分析
- MACD 與 Signal 線的相對位置
- 柱狀圖(Histogram)的變化趨勢
- 是否出現黃金交叉或死亡交叉
- MACD 背離現象的觀察

#### 3.3 威廉指標分析
- 當前 %R 值的位置
- 超買超賣狀態判斷
- 與價格的配合度

#### 3.4 RSI 指標分析
- RSI 的當前數值和趨勢
- 超買超賣判斷
- RSI 背離的觀察

### 4. 型態分析
- K線組合型態 (如紅三兵、黑三鴉、十字星等)
- 是否形成重要的反轉或延續型態
- 缺口的觀察

### 5. 支撐與壓力分析
- **關鍵支撐位**: 列出3個重要支撐價位並說明理由
- **關鍵壓力位**: 列出3個重要壓力價位並說明理由
- 支撐壓力的強弱程度評估

### 6. 趨勢判斷
- **短期趨勢** (5-10日): 多頭/空頭/盤整
- **中期趨勢** (20-60日): 多頭/空頭/盤整
- **長期趨勢** (>60日): 多頭/空頭/盤整
- 各週期趨勢的一致性分析

### 7. 基本面與技術面整合 (如有財務數據)
- 技術面與基本面是否協同
- 價格表現與財務狀況的一致性
- 綜合評估

### 8. ⚠️ 風險評估
- 當前主要風險因子
- 需注意的警訊
- 風險等級評估 (高/中/低)

---

## 💡 操作建議 (僅供參考)

### 短期操作建議 (1-5個交易日)
- **操作方向**: 偏多/偏空/觀望
- **進場參考價位**: NT$ XXX - XXX
- **停損參考價位**: NT$ XXX (下跌X%)
- **停利參考價位**: NT$ XXX (上漲X%)
- **依據**: 基於XXX指標顯示...

### 中期操作建議 (1-4週)
- **操作方向**: 偏多/偏空/觀望
- **目標價位區間**: NT$ XXX - XXX
- **停損參考**: NT$ XXX
- **依據**: 基於XXX趨勢...

### 長期投資建議 (1個月以上)
- **投資價值評估**: 適合/不適合長期持有
- **目標價位**: NT$ XXX
- **依據**: 結合基本面和技術面...

---

**重要聲明**:
- 以上分析純粹基於歷史數據的技術分析，僅供參考學習
- 所有價位和建議都是基於當前數據的參考值，非投資建議
- 歷史表現不代表未來結果
- 投資人應自行判斷並承擔投資風險

請提供專業、詳細且結構化的分析報告。"""

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
    繪製進階圖表：K 線圖 + 移動平均線 + 多種技術指標

    參數:
        df: 包含股票數據和所有技術指標的 DataFrame
        symbol: 股票代碼

    返回:
        plotly figure 對象
    """
    if df is None or df.empty:
        return None

    # 創建子圖表：6 個子圖（K線+MA、RSI、KD、MACD、威廉指標、成交量）
    fig = make_subplots(
        rows=6, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=[0.35, 0.13, 0.13, 0.13, 0.13, 0.13],
        subplot_titles=(
            f'{symbol} 股價 K 線圖與技術指標',
            'RSI 相對強弱指標',
            'KD 隨機指標',
            'MACD 指標',
            '威廉指標 %R',
            '成交量'
        )
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

    # ========== 第三排：KD 指標 ==========
    fig.add_trace(go.Scatter(
        x=df['date'], y=df['K'],
        mode='lines', name='K值',
        line=dict(color='#FF6B6B', width=2)
    ), row=3, col=1)

    fig.add_trace(go.Scatter(
        x=df['date'], y=df['D'],
        mode='lines', name='D值',
        line=dict(color='#4ECDC4', width=2)
    ), row=3, col=1)

    # KD 超買超賣線
    fig.add_hline(y=80, line_dash="dash", line_color="red", annotation_text="超買 (80)", row=3, col=1)
    fig.add_hline(y=20, line_dash="dash", line_color="green", annotation_text="超賣 (20)", row=3, col=1)

    # ========== 第四排：MACD 指標 ==========
    fig.add_trace(go.Scatter(
        x=df['date'], y=df['MACD'],
        mode='lines', name='MACD',
        line=dict(color='#2E86DE', width=2)
    ), row=4, col=1)

    fig.add_trace(go.Scatter(
        x=df['date'], y=df['MACD_Signal'],
        mode='lines', name='Signal',
        line=dict(color='#FFA07A', width=2)
    ), row=4, col=1)

    # MACD 柱狀圖
    colors_macd = ['#ef5350' if val >= 0 else '#26a69a' for val in df['MACD_Hist']]
    fig.add_trace(go.Bar(
        x=df['date'], y=df['MACD_Hist'],
        name='MACD Hist',
        marker_color=colors_macd,
        showlegend=False
    ), row=4, col=1)

    fig.add_hline(y=0, line_dash="solid", line_color="gray", line_width=1, row=4, col=1)

    # ========== 第五排：威廉指標 ==========
    fig.add_trace(go.Scatter(
        x=df['date'], y=df['WillR'],
        mode='lines', name='Williams %R',
        line=dict(color='#9B59B6', width=2)
    ), row=5, col=1)

    # 威廉指標參考線
    fig.add_hline(y=-20, line_dash="dash", line_color="red", annotation_text="超買 (-20)", row=5, col=1)
    fig.add_hline(y=-80, line_dash="dash", line_color="green", annotation_text="超賣 (-80)", row=5, col=1)
    fig.add_hrect(y0=-20, y1=0, fillcolor="red", opacity=0.1, layer="below", line_width=0, row=5, col=1)
    fig.add_hrect(y0=-100, y1=-80, fillcolor="green", opacity=0.1, layer="below", line_width=0, row=5, col=1)

    # ========== 第六排：成交量 ==========
    colors = ['#ef5350' if df['close'].iloc[i] >= df['open'].iloc[i] else '#26a69a'
              for i in range(len(df))]  # 紅色 = 上漲, 綠色 = 下跌 (台股習慣)

    fig.add_trace(go.Bar(
        x=df['date'], y=df['volume'],
        name='成交量',
        marker_color=colors,
        showlegend=False
    ), row=6, col=1)

    # 更新布局
    fig.update_layout(
        height=1400,  # 增加高度以容納更多子圖
        template='plotly_white',
        hovermode='x unified',
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.01,
            xanchor="right",
            x=1
        )
    )

    # 更新 Y 軸標籤
    fig.update_yaxes(title_text="價格 (TWD)", row=1, col=1)
    fig.update_yaxes(title_text="RSI", range=[0, 100], row=2, col=1)
    fig.update_yaxes(title_text="KD 值", range=[0, 100], row=3, col=1)
    fig.update_yaxes(title_text="MACD", row=4, col=1)
    fig.update_yaxes(title_text="%R", range=[-100, 0], row=5, col=1)
    fig.update_yaxes(title_text="成交量", row=6, col=1)
    fig.update_xaxes(title_text="日期", row=6, col=1)

    # 隱藏 K 線圖的 rangeslider
    fig.update_xaxes(rangeslider_visible=False, row=1, col=1)

    return fig


# ==================== 基本面視覺化圖表函數 ====================

def plot_fscore_gauge(fscore_data):
    """
    繪製 F-Score 儀表盤圖

    參數:
        fscore_data: F-Score 數據 (包含 total_score)

    返回:
        Plotly Figure 物件
    """
    if not fscore_data:
        return None

    score = fscore_data['total_score']

    # 創建儀表盤圖
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=score,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Piotroski F-Score", 'font': {'size': 24}},
        delta={'reference': 5, 'increasing': {'color': "green"}, 'decreasing': {'color': "red"}},
        gauge={
            'axis': {'range': [None, 9], 'tickwidth': 1, 'tickcolor': "darkblue"},
            'bar': {'color': "darkblue"},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [0, 4], 'color': '#ffcccc'},
                {'range': [4, 7], 'color': '#fff9cc'},
                {'range': [7, 9], 'color': '#ccffcc'}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 7
            }
        }
    ))

    fig.update_layout(
        height=300,
        margin=dict(l=20, r=20, t=60, b=20)
    )

    return fig


def plot_financial_ratios_bar(ratios):
    """
    繪製財務比率橫條圖

    參數:
        ratios: 財務比率字典 (來自 calculate_financial_ratios)

    返回:
        Plotly Figure 物件
    """
    if not ratios:
        return None

    # 分類指標
    profitability_metrics = {}
    financial_health_metrics = {}

    for key, val in ratios.items():
        if isinstance(val, (int, float)):
            if key in ['ROE (%)', 'ROA (%)', '毛利率 (%)', '淨利率 (%)']:
                profitability_metrics[key] = val
            elif key in ['流動比率', '負債比率 (%)']:
                financial_health_metrics[key] = val

    # 創建子圖
    from plotly.subplots import make_subplots

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=('獲利能力指標', '財務健康指標'),
        specs=[[{"type": "bar"}, {"type": "bar"}]]
    )

    # 獲利能力指標
    if profitability_metrics:
        metrics = list(profitability_metrics.keys())
        values = list(profitability_metrics.values())
        colors = ['#66bb6a' if v > 0 else '#ef5350' for v in values]

        fig.add_trace(go.Bar(
            y=metrics,
            x=values,
            orientation='h',
            marker_color=colors,
            text=[f"{v:.2f}%" for v in values],
            textposition='outside',
            showlegend=False
        ), row=1, col=1)

    # 財務健康指標
    if financial_health_metrics:
        metrics = list(financial_health_metrics.keys())
        values = list(financial_health_metrics.values())

        # 流動比率 > 1 為好, 負債比率 < 50 為好
        colors = []
        for k, v in zip(metrics, values):
            if '流動比率' in k:
                colors.append('#66bb6a' if v > 1 else '#ef5350')
            elif '負債比率' in k:
                colors.append('#66bb6a' if v < 50 else '#ef5350')
            else:
                colors.append('#2196f3')

        fig.add_trace(go.Bar(
            y=metrics,
            x=values,
            orientation='h',
            marker_color=colors,
            text=[f"{v:.2f}%" if '%' in k else f"{v:.2f}" for k, v in zip(metrics, values)],
            textposition='outside',
            showlegend=False
        ), row=1, col=2)

    fig.update_layout(
        height=300,
        showlegend=False,
        template='plotly_white',
        margin=dict(l=20, r=20, t=60, b=20)
    )

    fig.update_xaxes(title_text="百分比 (%)", row=1, col=1)
    fig.update_xaxes(title_text="數值", row=1, col=2)

    return fig


def plot_revenue_profit_trends(income_df):
    """
    繪製營收與淨利趨勢圖

    參數:
        income_df: 損益表 DataFrame

    返回:
        Plotly Figure 物件
    """
    if income_df is None or income_df.empty:
        return None

    # 取最近8季數據（確保日期排序一致）
    df = income_df.sort_values('date', ascending=False).head(8).copy().reset_index(drop=True).iloc[::-1]

    if 'date' not in df.columns or 'Revenue' not in df.columns:
        return None

    # 創建雙軸圖表
    from plotly.subplots import make_subplots

    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=('營收趨勢', '淨利趨勢'),
        vertical_spacing=0.12,
        row_heights=[0.5, 0.5]
    )

    # 營收趨勢
    if 'Revenue' in df.columns:
        fig.add_trace(go.Scatter(
            x=df['date'].dt.strftime('%Y-%m'),
            y=df['Revenue'],
            mode='lines+markers',
            name='營收',
            line=dict(color='#2196f3', width=3),
            marker=dict(size=8),
            fill='tozeroy',
            fillcolor='rgba(33, 150, 243, 0.1)'
        ), row=1, col=1)

    # 淨利趨勢
    if 'IncomeAfterTaxes' in df.columns:
        colors = ['#66bb6a' if val >= 0 else '#ef5350' for val in df['IncomeAfterTaxes']]

        fig.add_trace(go.Bar(
            x=df['date'].dt.strftime('%Y-%m'),
            y=df['IncomeAfterTaxes'],
            name='稅後淨利',
            marker_color=colors,
            showlegend=False
        ), row=2, col=1)

    fig.update_layout(
        height=500,
        template='plotly_white',
        showlegend=True,
        hovermode='x unified'
    )

    fig.update_yaxes(title_text="營收 (千元)", row=1, col=1)
    fig.update_yaxes(title_text="淨利 (千元)", row=2, col=1)
    fig.update_xaxes(title_text="期間", row=2, col=1)

    return fig


def plot_profitability_trends(income_df, balance_df):
    """
    繪製 ROE 和 ROA 趨勢圖

    參數:
        income_df: 損益表 DataFrame
        balance_df: 資產負債表 DataFrame

    返回:
        Plotly Figure 物件
    """
    if income_df is None or balance_df is None or income_df.empty or balance_df.empty:
        return None

    # 取最近8季數據，統一排序
    income_recent = income_df.sort_values('date', ascending=False).head(8).copy().reset_index(drop=True).iloc[::-1]
    balance_recent = balance_df.sort_values('date', ascending=False).head(8).copy().reset_index(drop=True).iloc[::-1]

    # 合併數據
    merged = pd.merge(income_recent, balance_recent, on='date', how='inner')

    if merged.empty:
        return None

    # 計算 ROE 和 ROA
    roe_list = []
    roa_list = []
    dates = []

    for _, row in merged.iterrows():
        net_income = row.get('IncomeAfterTaxes', 0)
        equity = row.get('Equity', 0)
        assets = row.get('TotalAssets', row.get('Assets', 0))

        if equity and equity != 0:
            roe = (net_income / equity) * 100
            roe_list.append(roe)
        else:
            roe_list.append(None)

        if assets and assets != 0:
            roa = (net_income / assets) * 100
            roa_list.append(roa)
        else:
            roa_list.append(None)

        dates.append(row['date'])

    # 創建圖表
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=[d.strftime('%Y-%m') for d in dates],
        y=roe_list,
        mode='lines+markers',
        name='ROE (%)',
        line=dict(color='#ff9800', width=3),
        marker=dict(size=8)
    ))

    fig.add_trace(go.Scatter(
        x=[d.strftime('%Y-%m') for d in dates],
        y=roa_list,
        mode='lines+markers',
        name='ROA (%)',
        line=dict(color='#9c27b0', width=3),
        marker=dict(size=8)
    ))

    # 添加零線
    fig.add_hline(y=0, line_dash="dash", line_color="gray", line_width=1)

    fig.update_layout(
        title="獲利能力趨勢 (ROE & ROA)",
        height=350,
        template='plotly_white',
        hovermode='x unified',
        yaxis_title="百分比 (%)",
        xaxis_title="期間",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )

    return fig


def plot_monthly_revenue_chart(revenue_df):
    """
    繪製月營收趨勢圖（含年增率）

    參數:
        revenue_df: 月營收 DataFrame

    返回:
        Plotly Figure 物件
    """
    if revenue_df is None or revenue_df.empty:
        return None

    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=('月營收趨勢', '年增率 (YoY%)'),
        vertical_spacing=0.15,
        row_heights=[0.6, 0.4]
    )

    # 月營收柱狀圖
    fig.add_trace(go.Bar(
        x=revenue_df['revenue_month'],
        y=revenue_df['revenue'],
        name='月營收',
        marker_color='#2196f3',
        text=revenue_df['revenue'].apply(lambda x: f'{x/1000:.1f}' if x >= 1000 else f'{x:.0f}'),
        textposition='outside'
    ), row=1, col=1)

    # 年增率折線圖
    colors = ['#66bb6a' if val >= 0 else '#ef5350' for val in revenue_df['yoy_growth'].fillna(0)]
    fig.add_trace(go.Bar(
        x=revenue_df['revenue_month'],
        y=revenue_df['yoy_growth'],
        name='年增率',
        marker_color=colors,
        text=revenue_df['yoy_growth'].apply(lambda x: f'{x:.1f}%' if pd.notna(x) else 'N/A'),
        textposition='outside'
    ), row=2, col=1)

    fig.add_hline(y=0, line_dash="solid", line_color="gray", line_width=1, row=2, col=1)

    fig.update_layout(
        height=600,
        template='plotly_white',
        showlegend=False
    )

    fig.update_yaxes(title_text="營收 (千元)", row=1, col=1)
    fig.update_yaxes(title_text="年增率 (%)", row=2, col=1)
    fig.update_xaxes(title_text="月份", row=2, col=1)

    return fig


def plot_eps_trend_chart(eps_df):
    """
    繪製 EPS 趨勢圖（含季增、年增）

    參數:
        eps_df: EPS 趨勢 DataFrame

    返回:
        Plotly Figure 物件
    """
    if eps_df is None or eps_df.empty:
        return None

    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=('每股盈餘 (EPS)', 'EPS 成長率'),
        vertical_spacing=0.15,
        row_heights=[0.5, 0.5]
    )

    # EPS 柱狀圖
    colors_eps = ['#66bb6a' if val >= 0 else '#ef5350' for val in eps_df['EPS']]
    fig.add_trace(go.Bar(
        x=eps_df['date'].dt.strftime('%Y-Q%q'),
        y=eps_df['EPS'],
        name='EPS',
        marker_color=colors_eps,
        text=eps_df['EPS'].apply(lambda x: f'{x:.2f}'),
        textposition='outside'
    ), row=1, col=1)

    # 成長率折線圖
    fig.add_trace(go.Scatter(
        x=eps_df['date'].dt.strftime('%Y-Q%q'),
        y=eps_df['QoQ'],
        mode='lines+markers',
        name='季增率 (QoQ)',
        line=dict(color='#ff9800', width=2),
        marker=dict(size=8)
    ), row=2, col=1)

    fig.add_trace(go.Scatter(
        x=eps_df['date'].dt.strftime('%Y-Q%q'),
        y=eps_df['YoY'],
        mode='lines+markers',
        name='年增率 (YoY)',
        line=dict(color='#9c27b0', width=2),
        marker=dict(size=8)
    ), row=2, col=1)

    fig.add_hline(y=0, line_dash="dash", line_color="gray", line_width=1, row=1, col=1)
    fig.add_hline(y=0, line_dash="dash", line_color="gray", line_width=1, row=2, col=1)

    fig.update_layout(
        height=550,
        template='plotly_white',
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )

    fig.update_yaxes(title_text="EPS (元)", row=1, col=1)
    fig.update_yaxes(title_text="成長率 (%)", row=2, col=1)
    fig.update_xaxes(title_text="季度", row=2, col=1)

    return fig


def plot_margin_comparison_chart(margin_df):
    """
    繪製毛利率與營益率比較圖

    參數:
        margin_df: 毛利率與營益率 DataFrame

    返回:
        Plotly Figure 物件
    """
    if margin_df is None or margin_df.empty:
        return None

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=margin_df['date'].dt.strftime('%Y-Q%q'),
        y=margin_df['毛利率'],
        mode='lines+markers',
        name='毛利率',
        line=dict(color='#2196f3', width=3),
        marker=dict(size=10),
        text=margin_df['毛利率'].apply(lambda x: f'{x:.1f}%'),
        textposition='top center'
    ))

    fig.add_trace(go.Scatter(
        x=margin_df['date'].dt.strftime('%Y-Q%q'),
        y=margin_df['營益率'],
        mode='lines+markers',
        name='營益率',
        line=dict(color='#ff9800', width=3),
        marker=dict(size=10),
        text=margin_df['營益率'].apply(lambda x: f'{x:.1f}%'),
        textposition='bottom center'
    ))

    fig.update_layout(
        title="毛利率與營益率趨勢比較（近4季）",
        height=400,
        template='plotly_white',
        hovermode='x unified',
        yaxis_title="百分比 (%)",
        xaxis_title="季度",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )

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
                    # 計算所有技術指標
                    data_with_ma = get_moving_averages(filtered_data)
                    data_with_rsi = calculate_rsi(data_with_ma, period=rsi_period)
                    data_with_kd = calculate_kd(data_with_rsi)
                    data_with_macd = calculate_macd(data_with_kd)
                    tech_data = calculate_willr(data_with_macd)
            else:
                tech_data = None

            # 財務數據
            income_df = get_financial_statements(symbol, finmind_token)
            balance_df = get_balance_sheet(symbol, finmind_token)

            # 新增數據：月營收
            monthly_revenue_df = get_monthly_revenue(symbol, finmind_token)

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
                # === 1. 月營收概況 ===
                st.subheader("📅 營收概況與變化分析（近6個月）")
                if monthly_revenue_df is not None and not monthly_revenue_df.empty:
                    # 顯示最新月營收關鍵數據
                    latest_rev = monthly_revenue_df.iloc[0]
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("最新月營收", f"{latest_rev['revenue']/1000:.1f} 億",
                                 f"{latest_rev['mom_growth']:.1f}% MoM" if pd.notna(latest_rev['mom_growth']) else "N/A")
                    with col2:
                        yoy_val = latest_rev['yoy_growth'] if pd.notna(latest_rev['yoy_growth']) else 0
                        st.metric("年增率 (YoY)", f"{yoy_val:.1f}%",
                                 delta_color="normal" if yoy_val >= 0 else "inverse")
                    with col3:
                        period_str = str(latest_rev['revenue_month'])[:7] if 'revenue_month' in latest_rev else "N/A"
                        st.metric("期間", period_str)

                    # 月營收圖表
                    fig_monthly_rev = plot_monthly_revenue_chart(monthly_revenue_df)
                    if fig_monthly_rev:
                        st.plotly_chart(fig_monthly_rev, use_container_width=True)
                else:
                    st.info("💡 無法獲取月營收數據")

                st.divider()

                # === 2. EPS 趨勢分析 ===
                st.subheader("💎 每股盈餘（EPS）趨勢（近5季）")
                eps_trend = calculate_eps_trend(income_df)
                if eps_trend is not None and not eps_trend.empty:
                    # 顯示最新 EPS
                    latest_eps_row = eps_trend.iloc[0]
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("最新 EPS", f"{latest_eps_row['EPS']:.2f} 元")
                    with col2:
                        qoq_val = latest_eps_row['QoQ'] if pd.notna(latest_eps_row['QoQ']) else 0
                        st.metric("季增率 (QoQ)", f"{qoq_val:.1f}%" if pd.notna(latest_eps_row['QoQ']) else "N/A")
                    with col3:
                        yoy_val = latest_eps_row['YoY'] if pd.notna(latest_eps_row['YoY']) else 0
                        st.metric("年增率 (YoY)", f"{yoy_val:.1f}%" if pd.notna(latest_eps_row['YoY']) else "N/A")

                    # EPS 圖表
                    fig_eps = plot_eps_trend_chart(eps_trend)
                    if fig_eps:
                        st.plotly_chart(fig_eps, use_container_width=True)
                else:
                    st.info("💡 數據不足，無法計算 EPS 趨勢")

                st.divider()

                # === 3. 本益比與股價位階 ===
                st.subheader("📈 本益比與歷史股價位階")
                if tech_data is not None and income_df is not None:
                    current_price = tech_data.iloc[-1]['close']
                    latest_eps = income_df.iloc[0].get('EPS', 0) if 'EPS' in income_df.columns else 0
                    pe_ratio = calculate_pe_ratio(current_price, latest_eps)

                    # 計算歷史價格區間
                    all_prices = stock_data['close'] if stock_data is not None else tech_data['close']
                    price_high = all_prices.max()
                    price_low = all_prices.min()
                    price_avg = all_prices.mean()
                    price_position = ((current_price - price_low) / (price_high - price_low)) * 100 if price_high != price_low else 50

                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("目前股價", f"NT$ {current_price:.2f}")
                    with col2:
                        st.metric("本益比 (P/E)", f"{pe_ratio:.2f}" if pe_ratio else "N/A")
                    with col3:
                        st.metric("歷史高/低", f"{price_high:.2f} / {price_low:.2f}")
                    with col4:
                        position_text = "高位階" if price_position >= 70 else "低位階" if price_position <= 30 else "中位階"
                        st.metric("價格位階", f"{price_position:.1f}% ({position_text})")

                    # 估值評價
                    if pe_ratio:
                        if pe_ratio < 10:
                            valuation = "🟢 可能低估"
                        elif pe_ratio < 20:
                            valuation = "🟡 合理區間"
                        elif pe_ratio < 30:
                            valuation = "🟠 偏高"
                        else:
                            valuation = "🔴 可能高估"
                        st.info(f"估值評價: {valuation} (本益比: {pe_ratio:.2f})")
                else:
                    st.info("💡 數據不足，無法計算本益比")

                st.divider()

                # === 4. 毛利率與營益率趨勢 ===
                st.subheader("📊 毛利率與營益率變化趨勢（近4季）")
                margin_trend = calculate_margin_trends(income_df)
                if margin_trend is not None and not margin_trend.empty:
                    # 顯示最新數據
                    latest_margin = margin_trend.iloc[-1]
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("最新毛利率", f"{latest_margin['毛利率']:.2f}%")
                    with col2:
                        st.metric("最新營益率", f"{latest_margin['營益率']:.2f}%")

                    # 毛利率營益率圖表
                    fig_margin = plot_margin_comparison_chart(margin_trend)
                    if fig_margin:
                        st.plotly_chart(fig_margin, use_container_width=True)
                else:
                    st.info("💡 數據不足，無法計算毛利率與營益率趨勢")

                st.divider()

                # === 5. 財務健全度（ROE、ROA）===
                st.subheader("💪 財務健全度分析")
                ratios = calculate_financial_ratios(income_df, balance_df)
                if ratios:
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        roe = ratios.get('ROE', 0)
                        st.metric("ROE (股東權益報酬率)", f"{roe:.2f}%")
                    with col2:
                        roa = ratios.get('ROA', 0)
                        st.metric("ROA (資產報酬率)", f"{roa:.2f}%")
                    with col3:
                        current_ratio = ratios.get('流動比率', 0)
                        st.metric("流動比率", f"{current_ratio:.2f}")
                    with col4:
                        debt_ratio = ratios.get('負債比率', 0)
                        st.metric("負債比率", f"{debt_ratio:.2f}%")

                    # 財務比率視覺化
                    fig_ratios = plot_financial_ratios_bar(ratios)
                    if fig_ratios:
                        st.plotly_chart(fig_ratios, use_container_width=True)

                    # ROE/ROA 趨勢圖
                    fig_profitability = plot_profitability_trends(income_df, balance_df)
                    if fig_profitability:
                        st.plotly_chart(fig_profitability, use_container_width=True)

                st.divider()

                # === 6. F-Score 分析 ===
                st.subheader("🎯 Piotroski F-Score 財務體質評分")
                fscore = calculate_piotroski_fscore(income_df, balance_df)
                if fscore:
                    col1, col2 = st.columns([1, 2])
                    with col1:
                        score = fscore['total_score']
                        st.metric("F-Score 總分", f"{score}/9")
                        if score >= 7:
                            st.success("✅ 財務體質優秀 (≥7)")
                        elif score >= 5:
                            st.info("ℹ️ 財務體質良好 (5-6)")
                        else:
                            st.warning("⚠️ 財務體質需關注 (<5)")

                        # F-Score 儀表盤
                        fig_fscore = plot_fscore_gauge(fscore)
                        if fig_fscore:
                            st.plotly_chart(fig_fscore, use_container_width=True)

                    with col2:
                        st.write("**評分詳情:**")
                        for metric, data in fscore['details'].items():
                            status = "✅" if data.get('score') == 1 else "❌"
                            st.write(f"{status} {metric}: {data}")

                st.divider()

                # === 7. 最近財報數據表格（4季）===
                st.subheader("📋 最近財報數據（近4季）")
                if len(income_df) >= 4:
                    cols_to_show = ['date', 'Revenue', 'GrossProfit', 'OperatingIncome', 'IncomeAfterTaxes', 'EPS']
                    available_cols = ['date'] + [c for c in cols_to_show[1:] if c in income_df.columns]
                    display_df = income_df.head(4)[available_cols].copy()
                    display_df['date'] = display_df['date'].dt.strftime('%Y-Q%q')

                    # 格式化數值
                    for col in display_df.columns:
                        if col != 'date' and col in display_df.columns:
                            display_df[col] = display_df[col].apply(lambda x: f"{x:,.0f}" if pd.notna(x) else "N/A")

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
