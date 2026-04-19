# 🤖 LLM-Powered Stock Trading Agent

**AI-driven paper trading system using LLM decision-making, news sentiment analysis, and RAG memory**

---

## Overview

An end-to-end AI trading agent built in Python that leverages Large Language Models (Qwen2.5-1.5B-Instruct) to make autonomous stock trading decisions. The agent integrates real-time news sentiment, historical market data, and a retrieval-augmented generation (RAG) memory system to learn from past trades and improve over time.

## Key Features

- **LLM Decision Engine** — Uses Qwen2.5-1.5B-Instruct to analyze market conditions and generate buy/sell/hold decisions
- **Paper Trading via Alpaca** — Executes trades through the Alpaca Markets API in a risk-free paper trading environment
- **News Sentiment Analysis** — Fetches and analyzes financial news via Alpha Vantage to inform trading decisions
- **RAG Memory System** — Stores and retrieves past trade reflections to improve future decision-making
- **MCP-Style Tool Infrastructure** — Modular tool registry and dispatcher connecting the agent to data retrieval functions
- **SFT Fine-Tuning with LoRA** — Fine-tunes the base LLM on backtested trading data using Low-Rank Adaptation
- **Backtesting Engine** — Simulates historical trading days using yfinance data and cached news

## Architecture

```
┌──────────────────────────────────────────────┐
│              LLM Agent Core                  │
│         (Qwen2.5-1.5B-Instruct)              │
├──────────────────────────────────────────────┤
│  Tool Registry & Dispatcher (MCP-style)      │
├────────┬────────┬────────┬───────────────────┤
│ Alpaca │  News  │  RAG   │   Backtesting     │
│Trading │Fetcher │Memory  │   Engine          │
│ Client │(Alpha  │System  │  (yfinance +      │
│  (API) │Vantage)│        │   historical)     │
└────────┴────────┴────────┴───────────────────┘
```

## Tech Stack

| Component | Technology |
|-----------|-----------|
| LLM | Qwen2.5-1.5B-Instruct (HuggingFace Transformers) |
| Trading | Alpaca Markets API |
| News Data | Alpha Vantage NEWS_SENTIMENT API |
| Market Data | yfinance |
| Fine-Tuning | LoRA via PEFT + TRL |
| Framework | PyTorch, Google Colab (T4 GPU) |

## Getting Started

### Prerequisites
- Google Colab account (recommended for GPU access)
- [Alpaca Markets](https://alpaca.markets/) API key (free paper trading)
- [Alpha Vantage](https://www.alphavantage.co/) API key (free tier available)

### Setup
1. Open `Trading_Agent_final.ipynb` in Google Colab
2. Add your API keys via **Colab Secrets** (🔑 icon in left sidebar):
   - `ALPACA_API_KEY`
   - `ALPACA_SECRET_KEY`
   - `ALPHA_VANTAGE_KEY`
3. Run cells sequentially to:
   - Install dependencies & load the LLM
   - Connect to Alpaca paper trading
   - Fetch & cache financial news
   - Implement data retrieval tools
   - Build the RAG memory system
   - Fine-tune with LoRA on backtested data
   - Run the full autonomous trading loop

## Project Structure

```
trading-agent/
├── Trading_Agent_final.ipynb   # Complete trading agent notebook
└── README.md
```

## License

This project is for educational and research purposes only. Not financial advice.
