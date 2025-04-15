import streamlit as st
# import subprocess # No longer needed
import os
from datetime import date
import io # For capturing stdout
import contextlib # For capturing stdout
import json # For pretty printing results
import pandas as pd # For displaying backtest results
import math # Import math for isnan check
# from tabulate import tabulate # No longer needed for webapp display
import deepl # NEW: Import DeepL library
from dateutil.relativedelta import relativedelta # NEW: Import relativedelta
import plotly.graph_objects as go # <-- Add Plotly GO
from plotly.subplots import make_subplots # <-- Add Plotly Subplots
import numpy as np # Added for np.number type check

# --- Re-enable core logic imports --- 
from main import run_hedge_fund_core 
from backtester import run_backtest_core
from llm.models import LLM_ORDER, get_model_info, get_default_model, AVAILABLE_MODELS # Import get_default_model AND AVAILABLE_MODELS
# -------------------------------------

# --- OpenAI Translation Imports (Removed as unused) ---
# --- Removed langchain_openai and langchain_core.messages imports ---
# ---------------------------------------

# --- Localization (i18n) Setup --- 
# Define translation strings
TRANSLATIONS = {
    "en": {
        "page_title": "AI Hedge Fund",
        "header_title": "📈 AI Hedge Fund Simulator & Backtester",
        "header_caption": "Use AI agents to simulate trading decisions and backtest strategies.",
        "config_header": "Configuration",
        "tickers_label": "Stock Tickers (comma-separated)",
        "tickers_help": "Enter the stock tickers you want to analyze, separated by commas.",
        "start_date_label": "Start Date (Optional)",
        "end_date_label": "End Date (Optional)",
        "show_reasoning_label": "Show Agent Reasoning",
        "show_reasoning_help": "Display the step-by-step reasoning from the LLM agents (slower). Only for Simulation.",
        "select_analysts_label": "Select Analysts (Optional - Simulation Only)",
        "select_analysts_help": "Choose specific analysts to run. If none selected, all are run.",
        "backtest_all_analysts_info": "Note: Backtest currently runs with all available analysts.",
        "advanced_options_label": "Advanced Options (Optional)",
        "advanced_options_caption": "Currently uses default analysts and models.",
        "run_simulation_button": "🚀 Run Simulation",
        "run_backtest_button": "📊 Run Backtest",
        "simulation_spinner": "Running simulation for {tickers}...",
        "backtest_spinner": "Running backtest for {tickers}...",
        "simulation_complete": "Simulation complete!",
        "backtest_complete": "Backtest complete!",
        "enter_ticker_warning": "Please enter at least one ticker.",
        "select_start_date_warning": "Please select a Start Date for backtesting.",
        "final_decisions_header": "Final Decisions",
        "error_message": "Error: {error}",
        "error_details": "Details: {details}",
        "no_decisions_warning": "No final decisions were generated.",
        "agent_signals_header": "Individual Agent Signals",
        "investor_agents_header": "Investor Agent Signals", # Keep separate header for structure
        "analytical_agents_header": "Analytical Signals",
        "analysis_for_ticker": "Analysis for {ticker}",
        "signal_label": "Signal",
        "confidence_label": "Confidence",
        "reasoning_label": "Reasoning",
        "action_label": "Action",
        "quantity_label": "Quantity",
        "other_details_label": "Other Details",
        "confidence_not_provided": "Not Provided",
        "confidence_na": "N/A",
        "no_signals_warning": "No individual agent signals were generated or found.",
        "log_expander_title": "Simulation Output Log (Logs/Reasoning)",
        "backtest_log_expander_title": "Backtest Output Log",
        "stderr_warning": "Simulation Standard Error:",
        "backtest_stderr_warning": "Backtest Standard Error:",
        "error_unexpected": "An unexpected error occurred during simulation: {e}",
        "backtest_error_unexpected": "An unexpected error occurred during backtest: {e}",
        "log_before_error": "Captured Output Before Error:",
        "performance_metrics_header": "Performance Metrics",
        "trade_log_header": "Trade Log",
        "no_metrics_warning": "Backtest completed but no performance metrics were returned.",
        "no_trade_log_warning": "No trade log was generated or returned.",
        "disclaimer_header": "Disclaimer",
        "disclaimer_text": "This project is for educational and research purposes only. Not intended for real trading or investment. No warranties or guarantees provided. Past performance does not indicate future results. Consult a financial advisor for investment decisions.",
        "select_agents_label": "Select Agents to Run",
        "mandatory_analysts_info": "Note: Technical, Fundamentals, Sentiment, Valuation, and Quantitative Analysts are always included.",
        "select_model_label": "Select AI Model (Optional)",
        "select_model_help": "Defaults to DeepSeek R1 if not specified.",
        "portfolio_value_header": "Portfolio Value Chart",
        "no_portfolio_value_data_warning": "No portfolio value data found.",
        "portfolio_value_missing_columns_warning": "Portfolio value data is missing required columns.",
        "portfolio_value_chart_error": "Error displaying portfolio value chart: {e}",
        "run_button_label": "Run",
        "mode_select_label": "Select Mode",
        "mode_simulation": "Simulation",
        "mode_backtest": "Backtest",
        "app_description": """**Welcome!** Use this tool to:
* Simulate trading decisions using various AI agents.
* Backtest AI-driven strategies against historical data.
* Understand how AI analyzes markets and manages portfolios.""",
        "mode_simulation_desc": """**Simulation Mode:**
* Runs AI agents *once* for the entire date range.
* Uses data available at the **end date** to make a single trading decision.
* Useful for getting a quick analysis based on the latest available information.""",
        "mode_backtest_desc": """**Backtest Mode:**
* Simulates trading **day-by-day** through the date range, starting with **$100,000 initial capital** (default).
* Uses only **historical data available up to each specific day** to make decisions.
* Calculates performance metrics (Return, Sharpe, Drawdown etc.) based on the simulated trades.""",
    },
    "zh": {
        "page_title": "AI 对冲基金",
        "header_title": "📈 AI 对冲基金模拟器与回测器",
        "header_caption": "使用 AI Agent 分析股票并模拟交易决策或进行历史回测。",
        "config_header": "配置",
        "tickers_label": "股票代码 (逗号分隔)",
        "tickers_help": "输入您想分析的股票代码，例如：AAPL,MSFT,GOOG",
        "start_date_label": "开始日期 (可选)",
        "end_date_label": "结束日期 (可选)",
        "show_reasoning_label": "显示 Agent 推理过程",
        "show_reasoning_help": "显示 LLM Agent 的详细推理步骤（较慢）。仅限模拟模式。",
        "select_analysts_label": "选择 Agent (可选 - 仅限模拟)",
        "select_analysts_help": "选择要运行的特定 Agent。如果未选择，则运行所有 Agent。",
        "backtest_all_analysts_info": "注意：回测目前会运行所有可用的 Agent。",
        "advanced_options_label": "高级选项 (可选)",
        "advanced_options_caption": "在此处配置模型提供商、特定模型名称等。",
        "run_simulation_button": "🚀 运行模拟",
        "run_backtest_button": "📊 运行回测",
        "simulation_spinner": "正在为 {tickers} 运行 AI 模拟...",
        "backtest_spinner": "正在为 {tickers} 运行历史回测...",
        "simulation_complete": "模拟完成！",
        "backtest_complete": "回测完成！",
        "enter_ticker_warning": "请输入至少一个股票代码。",
        "select_start_date_warning": "请为回测选择一个开始日期。",
        "final_decisions_header": "最终决策",
        "error_message": "错误: {error}",
        "error_details": "详情: {details}",
        "no_decisions_warning": "未能生成最终决策。",
        "agent_signals_header": "各 Agent 信号",
        "investor_agents_header": "投资策略 Agent 信号",
        "analytical_agents_header": "分析型 Agent 信号",
        "analysis_for_ticker": "对 {ticker} 的分析",
        "signal_label": "信号",
        "confidence_label": "置信度",
        "reasoning_label": "理由",
        "action_label": "操作",
        "quantity_label": "数量",
        "other_details_label": "其他详情",
        "confidence_not_provided": "未提供",
        "confidence_na": "不适用",
        "no_signals_warning": "未能生成或找到任何 Agent 信号。",
        "log_expander_title": "模拟输出日志 (日志/推理过程)",
        "backtest_log_expander_title": "回测输出日志",
        "stderr_warning": "模拟标准错误输出:",
        "backtest_stderr_warning": "回测标准错误输出:",
        "error_unexpected": "模拟过程中发生意外错误: {e}",
        "backtest_error_unexpected": "回测过程中发生意外错误: {e}",
        "log_before_error": "错误发生前的日志输出:",
        "performance_metrics_header": "性能指标",
        "trade_log_header": "交易日志",
        "no_metrics_warning": "回测已完成，但未返回性能指标。",
        "no_trade_log_warning": "未能生成或返回交易日志。",
        "disclaimer_header": "免责声明",
        "disclaimer_text": "本项目仅用于教育和研究目的。不构成真实交易或投资建议。不提供任何保证。过往表现不预示未来结果。投资决策请咨询财务顾问。",
        "select_agents_label": "选择要运行的 Agent",
        "mandatory_analysts_info": "注意：技术分析、基本面分析、情绪分析、估值分析和量化分析 Agent 总是包含在内。",
        "select_model_label": "选择 AI 模型 (可选)",
        "select_model_help": "如果未指定，则默认使用 DeepSeek R1。",
        "portfolio_value_header": "组合价值图表",
        "no_portfolio_value_data_warning": "未找到组合价值数据。",
        "portfolio_value_missing_columns_warning": "组合价值数据缺少必需的列。",
        "portfolio_value_chart_error": "显示组合价值图表时出错: {e}",
        "run_button_label": "运行",
        "mode_select_label": "选择模式",
        "mode_simulation": "模拟",
        "mode_backtest": "回测",
        "app_description": """**欢迎！** 使用本工具可以：
* 利用多种 AI Agent 模拟交易决策。
* 基于历史数据回测 AI 驱动的交易策略。
* 了解 AI 如何分析市场及管理投资组合。""",
        "mode_simulation_desc": """**Simulation Mode:**
* Runs AI agents *once* for the entire date range.
* Uses data available at the **end date** to make a single trading decision.
* Useful for getting a quick analysis based on the latest available information.""",
        "mode_backtest_desc": """**回测模式：**
* 在选定的日期范围内**逐日**模拟交易，默认使用 **$100,000 初始资本**。
* **仅**使用截至**当日**的历史数据做出决策。
* 基于模拟交易计算性能指标（回报率、夏普比率、最大回撤等）。""",
    },
    "zh-Hant": { # NEW: Traditional Chinese
        "page_title": "AI 對沖基金",
        "header_title": "📈 AI 對沖基金模擬器與回測器",
        "header_caption": "使用 AI 代理分析股票並模擬交易決策或進行歷史回測。",
        "config_header": "設定",
        "tickers_label": "股票代號 (以逗號分隔)",
        "tickers_help": "輸入您想要分析的股票代號，例如：AAPL,MSFT,GOOG",
        "start_date_label": "開始日期 (可選)",
        "end_date_label": "結束日期 (可選)",
        "show_reasoning_label": "顯示 Agent 推理過程",
        "show_reasoning_help": "顯示 LLM Agent 的詳細推理步驟（較慢）。僅限模擬模式。",
        "select_analysts_label": "選擇 Agent (可選 - 僅限模擬)",
        "select_analysts_help": "選擇要運行的特定 Agent。如果未選擇，則運行所有 Agent。",
        "backtest_all_analysts_info": "注意：回測目前會運行所有可用的 Agent。",
        "advanced_options_label": "進階選項 (可選)",
        "advanced_options_caption": "在此處配置模型提供者、特定模型名稱等。",
        "run_simulation_button": "🚀 運行模擬",
        "run_backtest_button": "📊 運行回測",
        "simulation_spinner": "正在為 {tickers} 運行 AI 模擬...",
        "backtest_spinner": "正在為 {tickers} 運行歷史回測...",
        "simulation_complete": "模擬完成！",
        "backtest_complete": "回測完成！",
        "final_decisions_header": "最終決策",
        "agent_signals_header": "個別代理信號",
        "investor_agents_header": "投資者代理", # Grouping Header
        "analytical_agents_header": "分析型代理", # Grouping Header
        "performance_metrics_header": "績效指標",
        "trade_log_header": "交易日誌",
        "disclaimer_header": "免責聲明",
        "disclaimer_text": "此專案僅供教育和研究目的。不用於真實交易或投資建議。不提供任何保證。過去的表現並不代表未來的結果。請諮詢財務顧問。",
        "error_unexpected": "發生意外錯誤：{e}",
        "backtest_error_unexpected": "回測期間發生意外錯誤：{e}",
        "error_message": "錯誤：{error}",
        "error_details": "詳情：{details}",
        "log_expander_title": "查看模擬日誌",
        "backtest_log_expander_title": "查看回測日誌",
        "log_before_error": "錯誤發生前的日誌：",
        "stderr_warning": "標準錯誤輸出 (可能包含警告或錯誤)：",
        "backtest_stderr_warning": "回測標準錯誤輸出：",
        "enter_ticker_warning": "請至少輸入一個股票代號。",
        "select_start_date_warning": "請為回測選擇一個開始日期。",
        "no_decisions_warning": "未能生成最終決策。",
        "no_signals_warning": "未能生成或找到個別代理信號。",
        "no_metrics_warning": "未能計算績效指標。",
        "no_trade_log_warning": "沒有交易記錄可顯示。",
        "analysis_for_ticker": "對 {ticker} 的分析",
        "signal_label": "信號",
        "confidence_label": "置信度",
        "reasoning_label": "理由",
        "action_label": "操作",
        "quantity_label": "數量",
        "other_details_label": "其他詳情",
        "confidence_not_provided": "未提供",
        "confidence_na": "無效數據", # For NaN confidence
        "value_label": "值", # For backtest metrics table header
        # Add any other required keys here, translating them
        "Initial Capital": "初始資本",
        "Final Portfolio Value": "最終投資組合價值",
        "Total Return": "總回報率",
        "Max Drawdown": "最大回撤",
        "Sharpe Ratio": "夏普比率",
        "Sortino Ratio": "索提諾比率",
        "Profit Factor": "盈利因子",
        "Total Trades": "總交易次數",
        "Winning Trades": "盈利交易次數",
        "Losing Trades": "虧損交易次數",
        "Win Rate": "勝率",
        "Average Trade Return": "平均交易回報率",
        "Average Win Return": "平均盈利回報率",
        "Average Loss Return": "平均虧損回報率",
        "Date": "日期",
        "Ticker": "代號",
        "Action": "操作",
        "Quantity": "數量",
        "Price": "價格",
        "Commission": "佣金",
        "Cash": "現金",
        "Portfolio Value": "投資組合價值",
        "select_agents_label": "選擇要運行的代理",
        "mandatory_analysts_info": "注意：技術分析、基本面分析、情緒分析、估值分析和量化分析代理總是包含在內。",
        "select_model_label": "選擇 AI 模型 (可選)",
        "select_model_help": "如果未指定，則默認使用 DeepSeek R1。",
        "portfolio_value_header": "組合價值圖表",
        "no_portfolio_value_data_warning": "未找到組合價值數據。",
        "portfolio_value_missing_columns_warning": "組合價值數據缺少必需的列。",
        "portfolio_value_chart_error": "顯示組合價值圖表時出錯: {e}",
        "run_button_label": "運行",
        "mode_select_label": "選擇模式",
        "mode_simulation": "模擬",
        "mode_backtest": "回測",
        "app_description": """**歡迎！** 使用本工具可以：
* 利用多種 AI Agent 模擬交易決策。
* 基於歷史數據回測 AI 驅動的交易策略。
* 了解 AI 如何分析市場及管理投資組合。""",
        "mode_simulation_desc": """**模擬模式：**
* 針對選定的整個日期範圍運行 AI Agent **一次**。
* 使用截至**結束日期**的可用數據做出單次交易決策。
* 適用於基於最新可用信息進行快速分析。""",
        "mode_backtest_desc": """**回測模式：**
* 在選定的日期範圍內**逐日**模擬交易，默認使用 **$100,000 初始資本**。
* **僅**使用截至**當日**的歷史數據做出決策。
* 基於模擬交易計算性能指標（回報率、夏普比率、最大回撤等）。""",
    }
}

# --- DeepL Translator Client Initialization (NEW) ---
deepl_translator = None
if os.getenv("DEEPL_API_KEY"):
    try:
        auth_key = os.getenv("DEEPL_API_KEY")
        deepl_translator = deepl.Translator(auth_key)
        # Verify connection by checking usage (optional but good practice)
        usage = deepl_translator.get_usage()
        if usage.character.limit_exceeded:
            print("WARNING: DeepL character limit exceeded. Translation might fail.")
        else: 
            print(f"INFO: DeepL Translator initialized. Characters used: {usage.character.count}/{usage.character.limit}")
    except Exception as e:
        print(f"ERROR: Failed to initialize DeepL Translator: {e}")
        deepl_translator = None # Ensure it's None if init fails
else:
    print("WARNING: DEEPL_API_KEY not found in environment variables. DeepL translation will be skipped.")
# --------------------------------------------------

# --- Updated translation function using DeepL --- 
@st.cache_data # Add caching decorator
def translate_text(text, lang):
    # --- Fixed keywords translation --- 
    signal_map_zh = { # Simplified
        "bullish": "看涨", "bearish": "看跌", "neutral": "中性",
        "buy": "买入", "sell": "卖出", "short": "做空",
        "cover": "平仓", "hold": "持有"
    }
    signal_map_hant = { # Traditional (NEW)
        "bullish": "看漲", "bearish": "看跌", "neutral": "中性",
        "buy": "買入", "sell": "賣出", "short": "做空",
        "cover": "平倉", "hold": "持有"
    }

    if lang == "zh":
        if isinstance(text, str) and text.lower() in signal_map_zh:
             return signal_map_zh[text.lower()]
        target_lang_deepl = "ZH" # Simplified Chinese for DeepL
        current_map = signal_map_zh
    elif lang == "zh-Hant":
        if isinstance(text, str) and text.lower() in signal_map_hant:
             return signal_map_hant[text.lower()]
        target_lang_deepl = "ZH-HANT" # Traditional Chinese for DeepL (NEW)
        current_map = signal_map_hant
    else:
        # Return original text for English or other unsupported languages
        return text 

    # --- Dynamic Text Translation using DeepL (Logic adjusted for target_lang) ---
    if isinstance(text, str) and len(text) > 10 and deepl_translator and text.lower() not in current_map:
        try:
            result = deepl_translator.translate_text(text, target_lang=target_lang_deepl) 
            if result and result.text:
                return result.text
            else:
                print(f"WARNING: DeepL returned empty result for: {text[:50]}...")
                return f"{text} (翻譯失敗)"
        except deepl.DeepLException as e:
            print(f"ERROR: DeepL translation failed: {e}")
            return f"{text} (翻譯錯誤)" 
        except Exception as e:
            print(f"ERROR: Unexpected error during DeepL translation: {e}")
            return f"{text} (翻譯意外錯誤)"
    elif isinstance(text, str):
         # Keep very short strings or strings when client is unavailable, or already translated keywords
         return text 
    else:
        # Return non-strings as is
        return text

# --- Agent Details (Replace with actual URLs if available) ---
# Ideally, move this to a separate config/utils file
AGENT_DETAILS = {
    # Investor Agents
    "ben_graham": {"photo_url": "https://via.placeholder.com/100.png?text=Ben+Graham", "display_name": "Ben Graham"},
    "bill_ackman": {"photo_url": "https://via.placeholder.com/100.png?text=Bill+Ackman", "display_name": "Bill Ackman"},
    "cathie_wood": {"photo_url": "https://via.placeholder.com/100.png?text=Cathie+Wood", "display_name": "Cathie Wood"},
    "charlie_munger": {"photo_url": "https://via.placeholder.com/100.png?text=Charlie+Munger", "display_name": "Charlie Munger"},
    "michael_burry": {"photo_url": "https://via.placeholder.com/100.png?text=Michael+Burry", "display_name": "Michael Burry"},
    "peter_lynch": {"photo_url": "https://via.placeholder.com/100.png?text=Peter+Lynch", "display_name": "Peter Lynch"},
    "phil_fisher": {"photo_url": "https://via.placeholder.com/100.png?text=Phil+Fisher", "display_name": "Phil Fisher"},
    "stanley_druckenmiller": {"photo_url": "https://via.placeholder.com/100.png?text=Stan+Druckenmiller", "display_name": "Stanley Druckenmiller"},
    "warren_buffett": {"photo_url": "https://via.placeholder.com/100.png?text=Warren+Buffett", "display_name": "Warren Buffett"},
    # Analytical Agents
    "technical_analyst": {"photo_url": "https://via.placeholder.com/100.png?text=Technicals", "display_name": "Technical Analyst"},
    "fundamentals": {"photo_url": "https://via.placeholder.com/100.png?text=Fundamentals", "display_name": "Fundamentals Analyst"},
    "sentiment": {"photo_url": "https://via.placeholder.com/100.png?text=Sentiment", "display_name": "Sentiment Analyst"},
    "valuation": {"photo_url": "https://via.placeholder.com/100.png?text=Valuation", "display_name": "Valuation Analyst"},
    "quantitative_analyst": {"photo_url": "https://via.placeholder.com/100.png?text=Quant", "display_name": "Quantitative Analyst"}, # Added Quant Analyst
    # Manager Agents (Optional, if their output is ever shown here)
    "risk_manager": {"photo_url": "https://via.placeholder.com/100.png?text=Risk+Manager", "display_name": "Risk Manager"},
    "portfolio_manager": {"photo_url": "https://via.placeholder.com/100.png?text=Portfolio+Mgr", "display_name": "Portfolio Manager"},
}
# Define which keys correspond to analytical agents for grouping
# FIX: Use keys exactly as received from backend (with _agent suffix, except quant)
ANALYTICAL_AGENT_KEYS = ["technical_analyst_agent", "fundamentals_agent", "sentiment_agent", "valuation_agent", "quantitative_analyst"]

# Removed DEFAULT_AGENT_DETAIL to avoid showing 'Unknown Agent'

# Removed the display_dict_as_expander helper function

# --- Moved Helper Functions UP --- 

def display_single_signal(signal_data, t, lang):
    """Displays the formatted signal, confidence, and reasoning for a single agent-ticker pair."""
    if not signal_data or not isinstance(signal_data, dict):
        st.caption("Invalid signal data.")
        return

    signal_type = signal_data.get("signal", "Not Provided")
    confidence = signal_data.get("confidence", None) # Keep as None if not provided
    reasoning = signal_data.get("reasoning", "")

    # Translate signal type if applicable
    signal_display = translate_text(signal_type, lang)

    # --- Add Color Logic based on original signal_type --- 
    color = "default" # Default color
    signal_lower = signal_type.lower() # Use original English signal for logic
    if signal_lower in ["bullish", "buy"]:
        color = "green"
    elif signal_lower in ["hold", "neutral"]:
        color = "orange" # Use orange for hold/neutral
    elif signal_lower in ["bearish", "sell", "short"]:
        color = "red"
    # ---------------------------------------------------

    # Format confidence
    if isinstance(confidence, (int, float)) and not math.isnan(confidence):
        confidence_display = f"{confidence:.1f}%"
    else:
        confidence_display = t["confidence_na"] # Use dictionary access

    # Translate and format reasoning (if applicable and translator available)
    reasoning_display = translate_text(reasoning, lang)

    # Display using columns for structure
    col1, col2 = st.columns([1, 4])
    with col1:
        # Apply color using Markdown
        # Use Markdown for label and value for better control
        st.markdown(f"**{t['signal_label']}**") # Bold label
        if color != "default":
            st.markdown(f":{color}[{signal_display}]") # Colored value
        else:
            st.markdown(signal_display) # Default value
        
        st.markdown(" ") # Add a little vertical space

        # Only display confidence if it was provided and valid
        st.markdown(f"**{t['confidence_label']}**") # Bold label
        if isinstance(confidence, (int, float)) and not math.isnan(confidence):
             st.markdown(confidence_display) # Value using markdown
        else:
            st.markdown(f"{t['confidence_na']}") # N/A using markdown

    with col2:
        st.markdown(f"**{t['reasoning_label']}:**")
        st.caption(reasoning_display if reasoning_display else "-")

# --- Plotting Function --- # <-- Move create_indicator_chart here
def create_indicator_chart(df_hist: pd.DataFrame, ticker: str):
    """Creates an interactive Plotly chart with candlestick, volume, MACD, and RSI."""
    if df_hist is None or df_hist.empty:
        return None
        
    # Ensure necessary columns exist
    required_cols = ['open', 'high', 'low', 'close', 'volume']
    if not all(col in df_hist.columns for col in required_cols):
        print(f"Warning: DataFrame for {ticker} charting is missing required OHLCV columns.")
        # Try to proceed with at least 'close' if available
        if 'close' not in df_hist.columns:
             return None

    # Create figure with subplots: 3 rows - Price/Volume, MACD, RSI
    # Adjust row_heights: Price chart larger, MACD/RSI smaller
    fig = make_subplots(
        rows=3, cols=1, 
        shared_xaxes=True, 
        vertical_spacing=0.03, 
        subplot_titles=(f'{ticker} Price & Indicators', 'MACD', 'RSI'),
        row_heights=[0.6, 0.2, 0.2] # Adjust heights: 60% price, 20% MACD, 20% RSI
    )

    # --- Row 1: Price Candlestick, Volume, MA, BBands --- 
    # Candlestick chart
    if all(col in df_hist.columns for col in required_cols):
        fig.add_trace(
            go.Candlestick(
                x=df_hist.index,
                open=df_hist['open'], high=df_hist['high'],
                low=df_hist['low'], close=df_hist['close'],
                name='Candlestick'
            ), row=1, col=1
        )
        # Volume chart (as bar chart on the primary y-axis but visually distinct)
        # Consider adding a secondary y-axis if scales differ too much
        fig.add_trace(
            go.Bar(x=df_hist.index, y=df_hist['volume'], name='Volume', marker_color='rgba(100,100,100,0.3)'),
            row=1, col=1
        )
    else: # Fallback to line chart if OHLC not fully available
         fig.add_trace(
            go.Scatter(x=df_hist.index, y=df_hist['close'], mode='lines', name='Close Price'),
            row=1, col=1
        )
        
    # Add Moving Averages if columns exist
    if 'SMA_50' in df_hist.columns:
        fig.add_trace(go.Scatter(x=df_hist.index, y=df_hist['SMA_50'], mode='lines', name='SMA 50', line=dict(color='orange', width=1)), row=1, col=1)
    if 'SMA_200' in df_hist.columns:
        fig.add_trace(go.Scatter(x=df_hist.index, y=df_hist['SMA_200'], mode='lines', name='SMA 200', line=dict(color='purple', width=1)), row=1, col=1)
        
    # Add Bollinger Bands if columns exist
    if all(c in df_hist.columns for c in ['BBU_20_2.0', 'BBM_20_2.0', 'BBL_20_2.0']):
        fig.add_trace(go.Scatter(x=df_hist.index, y=df_hist['BBU_20_2.0'], mode='lines', name='Upper BB', line=dict(color='gray', width=1, dash='dash')), row=1, col=1)
        fig.add_trace(go.Scatter(x=df_hist.index, y=df_hist['BBM_20_2.0'], mode='lines', name='Middle BB', line=dict(color='gray', width=1, dash='dot')), row=1, col=1)
        fig.add_trace(go.Scatter(x=df_hist.index, y=df_hist['BBL_20_2.0'], mode='lines', name='Lower BB', line=dict(color='gray', width=1, dash='dash'), fill='tonexty', fillcolor='rgba(128,128,128,0.1)'), row=1, col=1)

    # --- Row 2: MACD --- 
    if all(c in df_hist.columns for c in ['MACD_12_26_9', 'MACDs_12_26_9', 'MACDh_12_26_9']):
        # MACD Line
        fig.add_trace(go.Scatter(x=df_hist.index, y=df_hist['MACD_12_26_9'], mode='lines', name='MACD', line=dict(color='blue')), row=2, col=1)
        # Signal Line
        fig.add_trace(go.Scatter(x=df_hist.index, y=df_hist['MACDs_12_26_9'], mode='lines', name='Signal', line=dict(color='red')), row=2, col=1)
        # Histogram (Bar Chart)
        colors = ['green' if v >= 0 else 'red' for v in df_hist['MACDh_12_26_9']]
        fig.add_trace(go.Bar(x=df_hist.index, y=df_hist['MACDh_12_26_9'], name='Histogram', marker_color=colors), row=2, col=1)
        fig.update_yaxes(title_text="MACD", row=2, col=1)

    # --- Row 3: RSI --- 
    if 'RSI_14' in df_hist.columns:
        fig.add_trace(go.Scatter(x=df_hist.index, y=df_hist['RSI_14'], mode='lines', name='RSI', line=dict(color='green')), row=3, col=1)
        # Add overbought/oversold lines
        fig.add_hline(y=70, line_dash="dash", line_color="red", annotation_text="Overbought (70)", annotation_position="bottom right", row=3, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="green", annotation_text="Oversold (30)", annotation_position="bottom right", row=3, col=1)
        fig.update_yaxes(title_text="RSI", range=[0, 100], row=3, col=1)

    # --- Layout Updates --- 
    fig.update_layout(
        height=700, # Adjust height as needed
        # title_text=f'{ticker} Price & Indicators', # Title already set in make_subplots
        xaxis_rangeslider_visible=False, # Hide the range slider on the main chart
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    # Remove gaps in time series
    fig.update_xaxes(rangebreaks=[dict(bounds=["sat", "mon"])]) # Remove weekends
    # Maybe add more holidays if needed: dict(values=list_of_holidays)

    return fig

# --- Helper Function for Displaying All Agent Signals --- # Moving display_agent_signals from below
def display_agent_signals(signals, selected_tickers, t, lang):
    if not signals:
        st.warning(t["no_signals_warning"])
        return
    
    # --- DEBUG: Print received signals ---
    # st.write("--- DEBUG: display_agent_signals received: ---")
    # st.json(signals, expanded=False)
    # -------------------------------------

    # Define agent categories (customize as needed)
    # These keys should match the keys returned in the 'analyst_signals' dictionary
    investor_agents = [
        "ben_graham_agent", "bill_ackman_agent", "cathie_wood_agent", "charlie_munger_agent",
        "michael_burry_agent", "peter_lynch_agent", "phil_fisher_agent", "stanley_druckenmiller_agent",
        "warren_buffett_agent"
    ]
    # ANALYTICAL_AGENT_KEYS is defined globally now

    # Separate signals by category
    investor_signals_by_ticker = {ticker: {} for ticker in selected_tickers}
    analytical_signals_by_ticker = {ticker: {} for ticker in selected_tickers}
    other_signals_by_ticker = {ticker: {} for ticker in selected_tickers}

    # Check if signals is a dictionary before iterating
    if not isinstance(signals, dict):
        st.error(f"DEBUG: Expected signals to be a dict, but got {type(signals)}")
        st.json(signals)
        return
        
    # --- FIX: Categorize agent *before* looping through tickers --- 
    for agent_key, ticker_signals in signals.items():
        # Skip risk manager signals in this display area
        if agent_key == "risk_management_agent": 
            continue
        
        # Ensure ticker_signals is a dict
        if not isinstance(ticker_signals, dict):
             st.warning(f"Skipping agent '{agent_key}' due to unexpected signal format: {type(ticker_signals)}")
             continue
             
        # --- FINAL DEBUG: Check keys right before comparison --- 
        # st.write(f"DEBUG: Checking agent_key: '{repr(agent_key)}' (Type: {type(agent_key)})")
        # st.write(f"DEBUG: investor_agents list: {investor_agents}")
        is_investor_check = agent_key in investor_agents
        # st.write(f"DEBUG: Result of '{repr(agent_key)}' in investor_agents: {is_investor_check}")
        # -------------------------------------------------------
        
        # Determine agent category ONCE
        agent_category = "OTHER" # Default category
        if is_investor_check: # Use the result from debug check
            agent_category = "INVESTOR"
            # st.write(f"DEBUG: Determined category for {agent_key} as INVESTOR") # Moved debug print
        elif agent_key in ANALYTICAL_AGENT_KEYS:
            agent_category = "ANALYTICAL"
            # st.write(f"DEBUG: Determined category for {agent_key} as ANALYTICAL") # Optional debug
        else:
             st.write(f"DEBUG: Determined category for {agent_key} as OTHER") # Moved debug print

        # Loop through tickers for this agent
        for ticker, signal_data in ticker_signals.items():
            if ticker not in selected_tickers:
                continue # Skip if ticker wasn't in the initial request
            
            # Populate the correct dictionary based on the determined category
            if agent_category == "INVESTOR":
                investor_signals_by_ticker[ticker][agent_key] = signal_data
            elif agent_category == "ANALYTICAL":
                analytical_signals_by_ticker[ticker][agent_key] = signal_data
            else: # OTHER
                other_signals_by_ticker[ticker][agent_key] = signal_data
                
    # --- Display Signals Ticker by Ticker (Vertical Layout) --- 
    for ticker in selected_tickers:
        st.markdown(f"### {t['analysis_for_ticker'].format(ticker=ticker)}") # Ticker Header (H3 for emphasis)
        
        ticker_analytical_signals = analytical_signals_by_ticker.get(ticker, {})
        ticker_investor_signals = investor_signals_by_ticker.get(ticker, {})
        
        if not ticker_analytical_signals and not ticker_investor_signals:
            st.caption(f"No signals found for {ticker}.")
            st.divider() # Add divider even if no signals
            continue # Move to next ticker

        # --- Display Analytical Signals for this ticker --- 
        if ticker_analytical_signals:
            st.subheader(t["analytical_agents_header"])
            # Special Handling for Quantitative Analyst
            if "quantitative_analyst" in ticker_analytical_signals:
                 quant_data = ticker_analytical_signals["quantitative_analyst"]
                 hist_df = quant_data.get('historical_data')
                 if hist_df is not None and not hist_df.empty:
                     chart_fig = create_indicator_chart(hist_df, ticker)
                     if chart_fig:
                          st.plotly_chart(chart_fig, use_container_width=True)
                     else:
                          st.caption("Could not generate indicator chart.")
                 else:
                      st.caption("Historical data for charting not found.")
                 with st.expander(f"Quantitative Indicators ({ticker}) - Table", expanded=False):
                    if isinstance(quant_data, dict):
                        tech_signals = quant_data.get('technical_signals')
                        if tech_signals and isinstance(tech_signals, dict):
                            indicator_table = []
                            for indicator, value in tech_signals.items():
                                formatted_value = "N/A"
                                if isinstance(value, (int, float)) and not math.isnan(value):
                                    formatted_value = f"{value:.2f}"
                                elif value is not None:
                                    formatted_value = str(value)
                                indicator_table.append([indicator, formatted_value])
                            if indicator_table:
                                df_indicators = pd.DataFrame(indicator_table, columns=["Indicator", "Value"])
                                st.dataframe(df_indicators, hide_index=True, use_container_width=True)
                            else:
                                st.caption("No technical signals data found.")
                        elif quant_data.get("error"):
                            st.error(f"Quant Error: {quant_data.get('error')}")
                        else:
                            st.caption("Technical signals data missing or invalid format.")
                    else:
                        st.json(quant_data) # Display raw if not dict
                 st.markdown("---") # Divider after quant section
                 del ticker_analytical_signals["quantitative_analyst"] # Remove handled quant data
            
            # Display other analytical agents
            if ticker_analytical_signals: # Check if any remain after quant
                for agent_key, signal_data in ticker_analytical_signals.items():
                    details = AGENT_DETAILS.get(agent_key.replace('_agent', ''))
                    display_name = details["display_name"] if details else agent_key
                    st.markdown(f"##### {display_name}")
                    display_single_signal(signal_data, t, lang)
            elif "quantitative_analyst" not in analytical_signals_by_ticker.get(ticker, {}): # If only quant was present initially
                pass # No other analytical signals to show
        else:
             st.caption(f"No analytical signals for {ticker}.") # Show if no analytical signals at all

        # --- Display Investor Signals for this ticker --- 
        if ticker_investor_signals:
            st.subheader(t["investor_agents_header"])
            for agent_key, signal_data in ticker_investor_signals.items():
                details = AGENT_DETAILS.get(agent_key.replace('_agent', ''))
                display_name = details["display_name"] if details else agent_key
                st.markdown(f"##### {display_name}")
                display_single_signal(signal_data, t, lang)
        else:
            st.caption(f"No investor signals for {ticker}.") # Show if no investor signals
            
        st.divider() # Divider after each ticker section

    # --- Remove the old column-based display logic and overall checks --- 
    # st.subheader(t["analytical_agents_header"])
    # analytical_signals_found = False
    # analytical_cols = st.columns(len(selected_tickers) if selected_tickers else 1)
    # for i, ticker in enumerate(selected_tickers):
    #     with analytical_cols[i % len(analytical_cols)]:
            # ... (Old analytical display logic removed)

    # if not analytical_signals_found:
    #      st.caption("No analytical signals were generated for the selected tickers.")
         
    # st.subheader(t["investor_agents_header"])
    # investor_signals_found = False
    # investor_cols = st.columns(len(selected_tickers) if selected_tickers else 1)
    # for i, ticker in enumerate(selected_tickers):
    #     with investor_cols[i % len(investor_cols)]:
            # ... (Old investor display logic removed)
            
    # if not investor_signals_found:
    #      st.caption("No investor signals were generated for the selected tickers.")

# --- Helper Function for Displaying Full Results --- 
def display_results(results, tickers, t, lang):
    """Displays the simulation or backtest results in Streamlit."""
    # Display Final Decisions
    st.subheader(t["final_decisions_header"])
    if results and not results.get("error") and results.get("decisions"):
        decisions_data = results["decisions"]
        if isinstance(decisions_data, dict):
            translated_decisions = {}
            for ticker, data in decisions_data.items():
                translated_data = data.copy()
                translated_data['action'] = translate_text(data.get('action', ''), lang)
                translated_data['reasoning'] = translate_text(data.get('reasoning', ''), lang)
                translated_decisions[ticker] = translated_data
            df_decisions = pd.DataFrame.from_dict(translated_decisions, orient='index')
            df_decisions.index.name = 'Ticker'
            cols_order = ['action', 'quantity', 'confidence', 'reasoning']
            df_decisions = df_decisions[[col for col in cols_order if col in df_decisions.columns]]
            df_decisions.rename(columns={
                'action': t["action_label"], 'quantity': t["quantity_label"],
                'confidence': t["confidence_label"], 'reasoning': t["reasoning_label"]
            }, inplace=True)
            st.dataframe(df_decisions, use_container_width=True)
        else:
            st.json(decisions_data)
    elif results and results.get("error"):
        st.error(t["error_message"].format(error=results['error']))
        if "details" in results:
            st.error(t["error_details"].format(details=results['details']))
    else:
        st.warning(t["no_decisions_warning"])

    # Display Individual Agent Signals using the helper function
    st.subheader(t["agent_signals_header"])
    if results and results.get("analyst_signals"):
         display_agent_signals(results["analyst_signals"], tickers, t, lang)
    elif not (results and results.get("error")):
         st.warning(t["no_signals_warning"])

# --- Set Page Config FIRST --- 
st.set_page_config(page_title="AI Hedge Fund / AI 对冲基金 / AI 對沖基金", layout="wide") # Updated neutral title

# --- Streamlit App Layout ---

# --- Sidebar Definition STARTS HERE ---
with st.sidebar:
    # --- Language selection in the sidebar ---
    def format_language(lang_code):
        if lang_code == "en":
            return "English"
        elif lang_code == "zh":
            return "简体中文"
        elif lang_code == "zh-Hant":
            return "繁體中文" # NEW
        return lang_code # Fallback

    language = st.selectbox(
        "Language / 语言 / 語言",
        options=["en", "zh", "zh-Hant"],
        format_func=format_language,
        key="language_selector"
    )
    TXT = TRANSLATIONS[language] # Get translated text based on selection

    # --- App Description (MOVED INSIDE SIDEBAR) ---
    st.markdown("---") # Add a separator
    st.markdown(TXT["app_description"])
    st.markdown("---") # Add another separator
    # -------------------------------------------

    # --- Ticker Input --- # Ticker input and other config should be OUTSIDE sidebar
    # st.header(TXT["config_header"])
    # tickers_input = st.text_input(...) 
    # ... other config moved to main area ...
# --- Sidebar Definition ENDS HERE ---


# --- Define available agents for selection (Globally needed) ---
# --- FIX: Replace placeholder code with actual definitions ---
ALL_AGENT_KEYS = list(AGENT_DETAILS.keys())
# Define mandatory keys expected by the core function (ensure these match run_hedge_fund_core expectations!)
# These might differ slightly from the keys used in AGENT_DETAILS if suffixes (_agent) are added/removed
MANDATORY_CORE_KEYS = ["technical_analyst_agent", "fundamentals_agent", "sentiment_agent", "valuation_agent", "quantitative_analyst"]
# Define keys for agents that should NOT be user-selectable in the UI
NON_SELECTABLE_UI_KEYS = ["risk_manager", "portfolio_manager"] + [k.replace('_agent', '') for k in MANDATORY_CORE_KEYS]

SELECTABLE_AGENT_KEYS = [
    k for k in ALL_AGENT_KEYS 
    if k not in NON_SELECTABLE_UI_KEYS
]
AGENT_DISPLAY_NAME_MAP = {
    AGENT_DETAILS[k]["display_name"]: k for k in SELECTABLE_AGENT_KEYS
}
SELECTABLE_DISPLAY_NAMES = list(AGENT_DISPLAY_NAME_MAP.keys())
# --- END FIX ---

# --- Main Area Layout Starts Here --- 
st.title(TXT["header_title"])
st.caption(TXT["header_caption"])

# --- Input Section (Now in Main Area) ---
st.header(TXT["config_header"])
tickers_input = st.text_input(
    TXT["tickers_label"],
    value="AAPL,MSFT,NVDA",
    help=TXT["tickers_help"],
    key="tickers"
)

# --- Calculate default dates (NEW) ---
end_default = date.today()
start_default = end_default - relativedelta(months=3)
# ------------------------------------

col1, col2 = st.columns(2)
with col1:
    start_date_input = st.date_input(TXT["start_date_label"], value=start_default, key="start_date")
with col2:
    end_date_input = st.date_input(TXT["end_date_label"], value=end_default, key="end_date")

# --- NEW: Mode Selection ---
mode = st.radio(
    label=TXT["mode_select_label"],
    options=[TXT["mode_simulation"], TXT["mode_backtest"]],
    horizontal=True,
    index=0 # Default to Simulation
)
is_simulation_mode = (mode == TXT["mode_simulation"])
is_backtest_mode = (mode == TXT["mode_backtest"])
# ------------------------

# --- NEW: Mode Descriptions ---
st.markdown("<hr style='height:1px;border:none;color:#333;background-color:#333;' />", unsafe_allow_html=True)
# Use columns for side-by-side description
col_desc1, col_desc2 = st.columns(2)
with col_desc1:
    st.caption(TXT["mode_simulation_desc"])
with col_desc2:
    st.caption(TXT["mode_backtest_desc"])
st.markdown("<hr style='height:1px;border:none;color:#333;background-color:#333;' />", unsafe_allow_html=True)
# ---------------------------

# --- Advanced Options (Optional) ---
with st.expander(TXT["advanced_options_label"]):
    # --- Re-add Show Reasoning Checkbox (Simulation Only) ---
    show_reasoning = st.checkbox(
        TXT["show_reasoning_label"],
        value=False, # Default to False
        help=TXT["show_reasoning_help"],
        disabled=is_backtest_mode, # Disable for backtest
        key="show_reasoning" # Add key for state management
    )
    # ---------------------------------------------------

    # --- Add Agent Selection Multiselect --- 
    selected_agent_display_names = st.multiselect(
        label=TXT.get("select_agents_label", "Select Agents to Run"), # Use translation key
        options=SELECTABLE_DISPLAY_NAMES, # Now only shows selectable (non-mandatory) agents
        default=SELECTABLE_DISPLAY_NAMES, # Default to all selectable agents
        key="selected_agents",
        disabled=is_backtest_mode # <-- ADDED disabled parameter
    )
    # --- Add info text below multiselect --- 
    # st.caption(TXT.get("mandatory_analysts_info", "Note: Technical, Fundamentals, Sentiment, Valuation, and Quantitative Analysts are always included."))
    # --- Show info text conditionally ONLY in backtest mode --- 
    if is_backtest_mode:
        st.info(TXT["backtest_all_analysts_info"])
    else:
        # Show mandatory info only in simulation mode below multiselect
        st.caption(TXT.get("mandatory_analysts_info", "Note: Technical, Fundamentals, Sentiment, Valuation, and Quantitative Analysts are always included."))
    # ----------------------------------------------------------

    # --- Model Selection Dropdown (MODIFIED TO FILTER) ---
    # Define allowed model names
    allowed_model_names = ["deepseek-chat", "gpt-4o"]
    
    # Filter available models
    filtered_models = [
        model for model in AVAILABLE_MODELS
        if model.model_name in allowed_model_names
    ]
    
    # Create choices from filtered models
    model_choices = {model.display_name: model.model_name for model in filtered_models}
    
    # Determine the default selection (prefer gpt-4o)
    default_model_display_name = "No models available"
    default_model_index = 0
    if filtered_models:
        gpt4o_model = next((m for m in filtered_models if m.model_name == "gpt-4o"), None)
        if gpt4o_model:
            default_model_display_name = gpt4o_model.display_name
        else: # Fallback to the first available filtered model
            default_model_display_name = filtered_models[0].display_name
            
        # Find the index of the default display name
        try:
            default_model_index = list(model_choices.keys()).index(default_model_display_name)
        except ValueError:
            default_model_index = 0 # Should not happen if logic is correct
            
    # Get the default model config (still needed for fallback provider/name if selection fails)
    # default_model_config = get_default_model()
    # default_model_display_name = default_model_config.display_name
    # Ensure the default display name is in the choices, otherwise use the first one as fallback
    # if default_model_display_name not in model_choices:
    #     if AVAILABLE_MODELS:
    #         default_model_display_name = AVAILABLE_MODELS[0].display_name
    #     else:
    #         default_model_display_name = "No models available"
            
    selected_model_display_name = st.selectbox(
        label=TXT.get("select_model_label", "Select AI Model (Optional)"),
        options=list(model_choices.keys()), # Use filtered choices
        index=default_model_index, # Set default index based on gpt-4o or first filtered
        help=TXT.get("select_model_help", "Defaults to DeepSeek R1 if not specified."), # Help text might need update later
        key="selected_model"
    )
    # Get the actual model_name and provider based on the selected display name
    selected_model_name = model_choices.get(selected_model_display_name)
    selected_model_info = get_model_info(selected_model_name) if selected_model_name else None
    # Fallback logic for provider/name might need adjustment if default model isn't selectable
    # Get the globally default model info for fallback
    global_default_model_config = get_default_model() 
    selected_model_provider = selected_model_info.provider.value if selected_model_info else global_default_model_config.provider.value
    if selected_model_name is None:
        selected_model_name = global_default_model_config.model_name
    # ------------------------------------

    # ---------------------------------------

# --- RUN BUTTON (Now correctly positioned after config) ---
st.markdown("---") # Separator
run_button_pressed = st.button(TXT["run_button_label"], key="run_unified", use_container_width=True)
# -----------------------------------------------------------

st.markdown("---") # Separator before results

# --- Results Area Definition ---
results_area = st.container()
log_expander_area = st.container()

# --- REMOVE Description from here ---
# st.markdown("---") # Add a separator
# st.markdown(TXT["app_description"])
# st.markdown("---") # Add another separator
# -----------------------------------

# --- Run Button Logic ---
if run_button_pressed:
    tickers_list = [ticker.strip().upper() for ticker in tickers_input.split(",")]
    if tickers_list:
        # --- Common setup for model ---
        default_model_config = get_default_model()
        selected_model_name = selected_model_name if selected_model_name else default_model_config.model_name
        selected_model_provider = selected_model_provider if selected_model_provider else default_model_config.provider.value
        model_info = get_model_info(selected_model_name)
        if model_info:
            selected_model_provider = model_info.provider.value
        # ----------------------------

        if is_simulation_mode:
            # --- Simulation Logic ---
            with results_area:
                st.info(f"Running Simulation for {', '.join(tickers_list)}...") # Simple status
            with st.spinner(TXT["simulation_spinner"].format(tickers=tickers_input)):
                sim_stdout = io.StringIO()
                sim_stderr = io.StringIO()
                results = None
                try:
                    # --- Define Initial Portfolio for Simulation ---
                    initial_portfolio_sim = {
                        "cash": 100000.0,
                        "margin_used": 0.0,
                        "margin_requirement": 0.0,
                        "positions": {t: {"long": 0, "short": 0, "long_cost_basis": 0.0, "short_cost_basis": 0.0, "short_margin_used": 0.0} for t in tickers_list},
                        "realized_gains": {t: {"long": 0.0, "short": 0.0} for t in tickers_list}
                    }
                    # ------------------------------------------------
                    with contextlib.redirect_stdout(sim_stdout), contextlib.redirect_stderr(sim_stderr):
                        # --- Convert selected display names back to keys (user selected only) --- 
                        selected_user_keys_from_ui = [
                            AGENT_DISPLAY_NAME_MAP[name] for name in selected_agent_display_names
                        ]
                        # --- !!! Combine user selected keys with mandatory keys !!! ---
                        # Make sure the keys here match what run_hedge_fund_core expects!
                        final_agent_keys_to_run = list(set(MANDATORY_CORE_KEYS + selected_user_keys_from_ui))
                        # ----------------------------------------------------------------
                        results = run_hedge_fund_core(
                            tickers=tickers_list,
                            start_date=start_date_input.strftime('%Y-%m-%d') if start_date_input else None,
                            end_date=end_date_input.strftime('%Y-%m-%d') if end_date_input else date.today().strftime('%Y-%m-%d'),
                            selected_analysts=final_agent_keys_to_run, # Pass the combined list
                            portfolio=initial_portfolio_sim, # <-- ADDED missing portfolio argument
                            show_reasoning=show_reasoning,
                            model_name=selected_model_name, # Pass selected model name
                            model_provider=selected_model_provider # Pass selected model provider
                        )
                    st.success(TXT["simulation_complete"])
                except Exception as e:
                    results_area.error(TXT["error_unexpected"].format(e=e))
                    # Log display logic here... (similar to below)

                # --- Display Simulation Results ---
                with results_area:
                    # Use existing display_results function
                    display_results(results, tickers_list, TXT, language)

                # --- Log Display for Simulation ---
                stdout_val = sim_stdout.getvalue()
                stderr_val = sim_stderr.getvalue()
                with log_expander_area:
                    if stdout_val:
                        with st.expander(TXT["log_expander_title"]):
                            st.code(stdout_val)
                    if stderr_val:
                        st.warning(TXT["stderr_warning"])
                        st.code(stderr_val)
                # ----------------------------------

        elif is_backtest_mode:
            # --- Backtest Logic ---
            with results_area:
                 st.info(f"Running Backtest for {', '.join(tickers_list)}...") # Simple status
            with st.spinner(TXT["backtest_spinner"].format(tickers=tickers_input)):
                 results = None
                 backtest_stdout = io.StringIO()
                 backtest_stderr = io.StringIO()
                 try:
                     with contextlib.redirect_stdout(backtest_stdout), contextlib.redirect_stderr(backtest_stderr):
                         results = run_backtest_core(
                             tickers=tickers_list,
                             start_date=start_date_input.strftime('%Y-%m-%d') if start_date_input else None,
                             end_date=end_date_input.strftime('%Y-%m-%d') if end_date_input else date.today().strftime('%Y-%m-%d'),
                             model_name=selected_model_name,
                             model_provider=selected_model_provider
                         )
                     st.success(TXT["backtest_complete"])
                 except Exception as e:
                     results_area.error(TXT["backtest_error_unexpected"].format(e=e))
                     # Log display logic here... (similar to simulation)

                 # --- Display Backtest Results ---
                 with results_area:
                     # Moved result display logic inside the button press
                     results_area.subheader(TXT["performance_metrics_header"])
                     # Check if performance_metrics exists and is not None
                     if results and not results.get("error") and results.get("performance_metrics") is not None:
                         # Attempt to display metrics, assuming it's a DICTIONARY of final metrics
                         try:
                             metrics_data = results["performance_metrics"]
                             # --- MODIFIED: Expect a DICT, convert to DataFrame --- 
                             if isinstance(metrics_data, dict):
                                 # Convert dict to DataFrame: index=metric names, column='Value'
                                 metrics_df = pd.DataFrame.from_dict(metrics_data, orient='index', columns=['Value'])
                             elif isinstance(metrics_data, pd.DataFrame): # Keep if backend already returns DF
                                 metrics_df = metrics_data
                                 if metrics_df.shape[1] == 1: # Ensure single value column has standard name
                                     metrics_df.columns = ['Value']
                             else:
                                 results_area.warning("Performance metrics format is not an expected dictionary or DataFrame.")
                                 metrics_df = pd.DataFrame() # Create empty DF
                             # ------------------------------------------------------

                             if not metrics_df.empty:
                                 # Translate index (metric names)
                                 metrics_df.index = [translate_text(idx, language) for idx in metrics_df.index]
                                 # Format numeric values (assuming 'Value' column or similar)
                                 if 'Value' in metrics_df.columns:
                                      # --- REWRITE: Apply formatting with explicit loop ---
                                      def format_value(key, value):
                                          # Handle non-numeric types first
                                          if not isinstance(value, (int, float, np.number)) or pd.isna(value):
                                              return value # Return non-numeric or NaN/NaT as is

                                          # --- Corrected Formatting Logic ---
                                          if 'Date' in key:
                                              return value # Already string
                                          elif key in ['Total Return %', 'Max Drawdown %', 'Win Rate %', 'Avg Win %', 'Avg Loss %']:
                                              # These are already percentages (0-100 range from backend)
                                              return f"{value:.2f}%"
                                          elif 'Ratio' in key:
                                              # Ratios should be decimals
                                              return f"{value:.2f}"
                                          elif 'Capital' in key or 'Value' in key or 'PnL' in key:
                                              # Currency formatting
                                              return f"${value:,.2f}" # Format as currency
                                          elif 'Consecutive' in key:
                                              # Integer formatting for counts
                                              return f"{int(value)}"
                                          else: # Default formatting for other numbers
                                              return f"{value:,.2f}"
                                      
                                      # --- REWRITE: Apply formatting with explicit loop ---
                                      metrics_df['Value'] = [
                                          format_value(idx, metrics_df.loc[idx, 'Value'])
                                          for idx in metrics_df.index # <-- Explicitly included loop
                                      ]
                                      # --- End REWRITE ---
                                 results_area.dataframe(metrics_df, use_container_width=True)
                             else:
                                 results_area.warning(TXT["no_metrics_warning"])

                         except Exception as metrics_e:
                             results_area.error(f"Error displaying performance metrics: {metrics_e}")
                             # Optionally display raw metrics data if available
                             if "performance_metrics" in results:
                                  results_area.write("Raw performance metrics data:")
                                  results_area.json(results["performance_metrics"])

                     results_area.subheader(TXT["trade_log_header"])
                     trade_log_df = results.get("trade_log") if results else None
                     if trade_log_df is not None and not trade_log_df.empty:
                         # Translate trade log columns
                         trade_log_df.columns = [translate_text(col, language) for col in trade_log_df.columns]
                         results_area.dataframe(trade_log_df, use_container_width=True)
                     elif not (results and results.get("error")):
                         results_area.info(TXT["no_trade_log_warning"])

                     # --- NEW: Display Portfolio Value Chart ---
                     results_area.subheader(TXT["portfolio_value_header"]) # Assumes this key exists in translations
                     portfolio_values_data = results.get("portfolio_values") if results else None

                     if portfolio_values_data and isinstance(portfolio_values_data, list) and len(portfolio_values_data) > 0:
                         try:
                             # Convert list of dicts to DataFrame
                             portfolio_df = pd.DataFrame(portfolio_values_data)
                             # Ensure 'Date' and 'Portfolio Value' columns exist
                             if 'Date' in portfolio_df.columns and 'Portfolio Value' in portfolio_df.columns:
                                 # Convert 'Date' to datetime and set as index
                                 portfolio_df['Date'] = pd.to_datetime(portfolio_df['Date'])
                                 portfolio_df = portfolio_df.set_index('Date')
                                 # Ensure 'Portfolio Value' is numeric
                                 portfolio_df['Portfolio Value'] = pd.to_numeric(portfolio_df['Portfolio Value'], errors='coerce')
                                 portfolio_df = portfolio_df.dropna(subset=['Portfolio Value'])

                                 # --- Calculate Y-axis range with padding --- 
                                 min_val = portfolio_df['Portfolio Value'].min()
                                 max_val = portfolio_df['Portfolio Value'].max()
                                 padding = (max_val - min_val) * 0.05 # 5% padding
                                 # Ensure padding is reasonable, handle cases where min=max
                                 if padding < 1: padding = 1 # Minimum padding of 1 unit
                                 if max_val == min_val:
                                      y_range = [min_val - padding, max_val + padding]
                                 else:
                                      y_range = [min_val - padding, max_val + padding]
                                 # -------------------------------------------

                                 # Display the line chart using only the 'Portfolio Value' column
                                 if not portfolio_df.empty:
                                     # Create Plotly figure
                                     fig = go.Figure()
                                     fig.add_trace(go.Scatter(
                                         x=portfolio_df.index,
                                         y=portfolio_df['Portfolio Value'],
                                         mode='lines',
                                         name=TXT.get('Portfolio Value', 'Portfolio Value') # Use get with fallback
                                     ))
                                     fig.update_layout(
                                         # title=TXT["portfolio_value_header"], # Title already set by subheader
                                         xaxis_title="Date",
                                         yaxis_title="Portfolio Value ($)",
                                         yaxis_range=y_range, # <-- SET Y-AXIS RANGE
                                         height=400 # Adjust height as needed
                                     )
                                     results_area.plotly_chart(fig, use_container_width=True)
                                 else:
                                     results_area.info(TXT["no_portfolio_value_data_warning"])
                             else:
                                  results_area.warning(TXT["portfolio_value_missing_columns_warning"])
                         except Exception as chart_e:
                             results_area.error(TXT["portfolio_value_chart_error"].format(e=chart_e))
                     elif not (results and results.get("error")):
                         results_area.info(TXT["no_portfolio_value_data_warning"])
                     # -------------------------------------------

                     # --- Log Display for Backtest ---
                     stdout_val = backtest_stdout.getvalue()
                     stderr_val = backtest_stderr.getvalue()
                     with log_expander_area:
                         if stdout_val:
                             with st.expander(TXT["backtest_log_expander_title"]):
                                 st.code(stdout_val)
                         if stderr_val:
                             st.warning(TXT["backtest_stderr_warning"])
                             st.code(stderr_val)
                     # -----------------------------------------------------
        else:
             with results_area:
                 st.warning(TXT["enter_ticker_warning"])

# --- Disclaimer ---
st.markdown("---")
st.warning(f"**{TXT['disclaimer_header']}**")
st.warning(TXT["disclaimer_text"]) 

# --- Main App Execution --- 
if __name__ == "__main__":
    # This block now only contains setup code if needed directly on run,
    # most logic is triggered by button presses above.
    pass # No direct execution needed here anymore 