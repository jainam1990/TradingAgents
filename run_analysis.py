from tradingagents.graph.trading_graph import TradingAgentsGraph
from tradingagents.default_config import DEFAULT_CONFIG
from dotenv import load_dotenv

load_dotenv()

config = DEFAULT_CONFIG.copy()
config["llm_provider"] = "openrouter"
config["deep_think_llm"] = "deepseek/deepseek-chat"
config["quick_think_llm"] = "deepseek/deepseek-chat"
config["max_debate_rounds"] = 1
config["max_risk_discuss_rounds"] = 1
config["data_vendors"] = {
    "core_stock_apis": "yfinance",
    "technical_indicators": "yfinance",
    "fundamental_data": "yfinance",
    "news_data": "yfinance",
}

ta = TradingAgentsGraph(debug=True, config=config)

_, decision = ta.propagate("SPCX", "2026-06-13")
print("\n" + "="*60)
print("FINAL DECISION")
print("="*60)
print(decision)
