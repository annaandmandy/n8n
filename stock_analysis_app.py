"""
AI 股票趨勢分析系統
使用 Streamlit, FinMind API, 和 OpenAI 進行股票技術分析
"""

import streamlit as st
import requests
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime, timedelta
import json
from openai import OpenAI

# ==================== 頁面設定 ====================
st.set_page_config(
    page_title="AI 股票趨勢分析系統",
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


def generate_ai_insights(symbol, stock_data, start_price, end_price, price_change, first_date, last_date, openai_api_key):
    """
    使用 OpenAI 進行技術分析

    參數:
        symbol: 股票代碼
        stock_data: 股票數據 DataFrame
        start_price: 起始價格
        end_price: 結束價格
        price_change: 價格變化百分比
        first_date: 起始日期
        last_date: 結束日期
        openai_api_key: OpenAI API 金鑰

    返回:
        str: AI 分析結果
    """
    try:
        # 初始化 OpenAI 客戶端
        client = OpenAI(api_key=openai_api_key)

        # 準備數據 - 轉換為 JSON 格式
        data_for_ai = stock_data[['date', 'open', 'high', 'low', 'close', 'volume', 'MA5', 'MA10', 'MA20', 'MA60']].copy()
        data_for_ai['date'] = data_for_ai['date'].dt.strftime('%Y-%m-%d')
        data_json = data_for_ai.to_json(orient='records', indent=2, force_ascii=False)

        # 系統角色設定
        system_message = """你是一位專業的技術分析師,專精於股票技術分析和歷史數據解讀。你的職責包括:

1. 客觀描述股票價格的歷史走勢和技術指標狀態
2. 解讀歷史市場數據和交易量變化模式
3. 識別技術面的歷史支撐阻力位
4. 提供純教育性的技術分析知識

重要原則:
- 僅提供歷史數據分析和技術指標解讀,絕不提供任何投資建議或預測
- 保持完全客觀中立的分析態度
- 使用專業術語但保持易懂
- 所有分析僅供教育和研究目的
- 強調技術分析的局限性和不確定性
- 使用繁體中文回答

嚴格的表達方式要求:
- 使用「歷史數據顯示」、「技術指標反映」、「過去走勢呈現」等客觀描述
- 避免「可能性」、「預期」、「建議」、「關注」等暗示性用詞
- 禁用「如果...則...」的假設句型,改用「歷史上當...時,曾出現...現象」
- 不提供具體價位的操作參考點,僅描述技術位階的歷史表現
- 強調「歷史表現不代表未來結果」
- 避免任何可能被解讀為操作指引的表達

免責聲明:所提供的分析內容純粹基於歷史數據的技術解讀,僅供教育和研究參考,不構成任何投資建議或未來走勢預測。歷史表現不代表未來結果。"""

        # 用戶提示語
        user_prompt = f"""請基於以下股票歷史數據進行深度技術分析:

### 基本資訊
- 股票代號:{symbol}
- 分析期間:{first_date} 至 {last_date}
- 期間價格變化:{price_change:.2f}% (從 NT${start_price:.2f} 變化到 NT${end_price:.2f})

### 完整交易數據
以下是該期間的完整交易數據,包含日期、開盤價、最高價、最低價、收盤價、成交量和移動平均線:
{data_json}

### 分析架構:技術面完整分析

#### 1. 趨勢分析
- 整體趨勢方向(上升、下降、盤整)
- 關鍵支撐位和阻力位識別
- 趨勢強度評估

#### 2. 技術指標分析
- 移動平均線分析(短期與長期MA的關係)
- 價格與移動平均線的相對位置
- 成交量與價格變動的關聯性

#### 3. 價格行為分析
- 重要的價格突破點
- 波動性評估
- 關鍵的轉折點識別

#### 4. 風險評估
- 當前價位的風險等級
- 潛在的支撐和阻力區間
- 市場情緒指標

#### 5. 市場觀察
- 短期技術面觀察(1-2週)
- 中期技術面觀察(1-3個月)
- 關鍵價位觀察點
- 技術面風險因子

### 綜合評估要求
#### 輸出格式要求
- 條理清晰,分段論述
- 提供具體的數據支撐
- 避免過於絕對的預測,強調分析的局限性
- 在適當位置使用表格或重點標記

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


def plot_candlestick_chart(df, symbol):
    """
    繪製 K 線圖和移動平均線

    參數:
        df: 包含股票數據和移動平均線的 DataFrame
        symbol: 股票代碼

    返回:
        plotly figure 對象
    """
    if df is None or df.empty:
        return None

    # 創建 K 線圖
    fig = go.Figure()

    # 添加 K 線圖
    fig.add_trace(go.Candlestick(
        x=df['date'],
        open=df['open'],
        high=df['high'],
        low=df['low'],
        close=df['close'],
        name='K線圖',
        increasing_line_color='#26a69a',  # 上漲顏色 (綠色)
        decreasing_line_color='#ef5350'   # 下跌顏色 (紅色)
    ))

    # 添加移動平均線
    fig.add_trace(go.Scatter(
        x=df['date'],
        y=df['MA5'],
        mode='lines',
        name='MA5',
        line=dict(color='#FF6B6B', width=1.5)
    ))

    fig.add_trace(go.Scatter(
        x=df['date'],
        y=df['MA10'],
        mode='lines',
        name='MA10',
        line=dict(color='#4ECDC4', width=1.5)
    ))

    fig.add_trace(go.Scatter(
        x=df['date'],
        y=df['MA20'],
        mode='lines',
        name='MA20',
        line=dict(color='#45B7D1', width=1.5)
    ))

    fig.add_trace(go.Scatter(
        x=df['date'],
        y=df['MA60'],
        mode='lines',
        name='MA60',
        line=dict(color='#FFA07A', width=1.5)
    ))

    # 更新圖表佈局
    fig.update_layout(
        title=f'{symbol} 股價 K 線圖與技術指標',
        yaxis_title='價格 (TWD)',
        xaxis_title='日期',
        template='plotly_white',
        height=600,
        hovermode='x unified',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        xaxis_rangeslider_visible=False
    )

    return fig


# ==================== 主程式 ====================

def main():
    # 頁面標題
    st.title("📈 AI 股票趨勢分析系統")
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

        # 步驟 1: 獲取股票數據
        with st.spinner("📊 正在獲取股票數據..."):
            stock_data = get_stock_data(symbol)

        if stock_data is None:
            return

        st.success(f"✅ 成功獲取 {len(stock_data)} 筆數據")

        # 步驟 2: 根據日期範圍過濾數據
        filtered_data = filter_by_date_range(stock_data, start_date, end_date)

        if filtered_data is None:
            return

        # 步驟 3: 計算移動平均線
        with st.spinner("📈 正在計算技術指標..."):
            data_with_ma = get_moving_averages(filtered_data)

        if data_with_ma is None:
            return

        # 步驟 4: 繪製 K 線圖
        st.subheader("📊 股價 K 線圖與技術指標")
        fig = plot_candlestick_chart(data_with_ma, symbol)

        if fig:
            st.plotly_chart(fig, use_container_width=True)

        # 步驟 5: 顯示基本統計資訊
        st.subheader("📈 基本統計資訊")

        start_price = data_with_ma.iloc[0]['close']
        end_price = data_with_ma.iloc[-1]['close']
        price_change = ((end_price - start_price) / start_price) * 100
        price_diff = end_price - start_price

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric(
                label="起始價格",
                value=f"NT${start_price:.2f}"
            )

        with col2:
            st.metric(
                label="結束價格",
                value=f"NT${end_price:.2f}"
            )

        with col3:
            st.metric(
                label="價格變化",
                value=f"NT${price_diff:.2f}",
                delta=f"{price_change:.2f}%"
            )

        # 步驟 6: AI 技術分析
        st.subheader("🤖 AI 技術分析")

        first_date = data_with_ma.iloc[0]['date'].strftime('%Y-%m-%d')
        last_date = data_with_ma.iloc[-1]['date'].strftime('%Y-%m-%d')

        ai_analysis = generate_ai_insights(
            symbol=symbol,
            stock_data=data_with_ma,
            start_price=start_price,
            end_price=end_price,
            price_change=price_change,
            first_date=first_date,
            last_date=last_date,
            openai_api_key=openai_api_key
        )

        if ai_analysis:
            st.markdown(ai_analysis)
            st.success("✅ 分析完成")

        # 步驟 7: 歷史數據表格
        st.subheader("📋 歷史數據表格 (最近 10 筆)")

        # 選擇要顯示的欄位
        display_columns = ['date', 'open', 'high', 'low', 'close', 'volume', 'MA5', 'MA10', 'MA20', 'MA60']
        recent_data = data_with_ma[display_columns].tail(10).iloc[::-1]  # 降序排列

        # 格式化日期
        recent_data_display = recent_data.copy()
        recent_data_display['date'] = recent_data_display['date'].dt.strftime('%Y-%m-%d')

        # 重命名欄位為中文
        recent_data_display.columns = ['日期', '開盤', '最高', '最低', '收盤', '成交量', 'MA5', 'MA10', 'MA20', 'MA60']

        st.dataframe(recent_data_display, use_container_width=True, hide_index=True)

    else:
        # 初始顯示說明
        st.info("👈 請在左側輸入股票代碼、API 金鑰和日期範圍,然後點擊「分析」按鈕開始分析")

        st.markdown("""
        ### 🎯 使用說明

        1. **輸入股票代碼**: 輸入您想分析的台股股票代碼 (例如: 2330 (台積電)、2317 (鴻海)、2454 (聯發科))
        2. **輸入 API 金鑰**:
           - OpenAI API Key: 前往 [OpenAI Platform](https://platform.openai.com) 獲取
        3. **選擇日期範圍**: 選擇您想分析的時間範圍
        4. **開始分析**: 點擊「分析」按鈕,系統將自動獲取數據並進行 AI 分析

        ### 📊 功能特色

        - ✅ **專業 K 線圖**: 互動式 K 線圖,支援縮放、平移等操作
        - ✅ **技術指標**: 自動計算 MA5、MA10、MA20、MA60 移動平均線
        - ✅ **AI 深度分析**: 使用 OpenAI o1-mini 進行專業的技術面分析
        - ✅ **數據視覺化**: 清晰的圖表和統計資訊展示
        - ✅ **免費數據源**: 使用 FinMind API,無需額外申請金鑰

        ### ⚠️ 注意事項

        - 本系統僅供教育和研究用途
        - 所有分析結果不構成投資建議
        - 請謹慎評估風險,自行做出投資決策
        """)


if __name__ == "__main__":
    main()
