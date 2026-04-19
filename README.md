# 🤖 LLM-Powered Stock Trading Agent

**Autonomous AI trading system using LLM decision-making, real-time news sentiment analysis, RAG memory, and LoRA fine-tuning for paper trading on Alpaca Markets**

---

## Overview

An end-to-end AI-powered stock trading agent that leverages a **Qwen2.5-1.5B-Instruct** Large Language Model to autonomously analyze market conditions, process financial news sentiment, and execute buy/sell/hold decisions via the Alpaca paper trading API. The system features a **Retrieval-Augmented Generation (RAG) memory** that enables the agent to learn from past trades and a **Supervised Fine-Tuning (SFT) pipeline with LoRA** to improve decision-making from backtested data.

## Key Features

- **LLM Decision Engine** — Qwen2.5-1.5B-Instruct on GPU (FP16) analyzes market data and generates structured JSON trading decisions with reasoning
- **Paper Trading Execution** — Live order execution via Alpaca Markets API (market & limit orders) in a risk-free paper environment
- **Real-Time News Sentiment** — Fetches and analyzes financial news via Alpha Vantage NEWS_SENTIMENT API with Google Drive caching
- **Technical Indicators** — RSI, MACD computation for quantitative signal analysis
- **RAG Memory System** — BM25-based retrieval of past trade reflections to inform future decisions; agent reflects on trade outcomes and stores lessons learned
- **MCP-Style Tool Infrastructure** — Modular tool registry and dispatcher connecting the agent to data retrieval functions (`get_news`, `get_price`, `get_rsi`, `get_macd`, `get_price_history`)
- **SFT Fine-Tuning with LoRA** — Fine-tunes the base LLM on profitable backtested trades using Parameter-Efficient Fine-Tuning (544K trainable params / 1.54B total — 0.035% trainable)
- **Backtesting Engine** — Multi-day historical simulation using yfinance price data and cached news, with PnL tracking per trade
- **Trading Data Collection** — Automated logging of agent episodes to JSONL for training data curation

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                  LLM Agent Core                         │
│           Qwen2.5-1.5B-Instruct (FP16, GPU)            │
│         + LoRA Adapter (SFT Fine-Tuned)                 │
├─────────────────────────────────────────────────────────┤
│         MCP-Style Tool Registry & Dispatcher            │
├──────────┬──────────┬───────────┬───────────────────────┤
│  Alpaca  │  News    │  RAG      │   Backtesting         │
│ Trading  │ Fetcher  │ Memory    │   Engine              │
│  Client  │(Alpha    │(BM25      │  (yfinance +          │
│  (REST   │Vantage   │Retrieval  │  historical news)     │
│   API)   │+ Cache)  │+ Reflect) │                       │
├──────────┴──────────┴───────────┴───────────────────────┤
│     SFT Fine-Tuning Pipeline (LoRA via PEFT + TRL)      │
│     Training Data Collector → Curate → Fine-Tune        │
└─────────────────────────────────────────────────────────┘
```

## Tech Stack

| Category | Technologies |
|----------|-------------|
| **LLM / NLP** | Qwen2.5-1.5B-Instruct, HuggingFace Transformers, Tokenizers |
| **Fine-Tuning** | LoRA (Low-Rank Adaptation), PEFT, TRL (SFTTrainer), HuggingFace Datasets |
| **Deep Learning** | PyTorch (CUDA, FP16 inference), Accelerate |
| **Trading API** | Alpaca Markets (alpaca-py) — Paper Trading, Market/Limit Orders |
| **Market Data** | yfinance (historical OHLCV), Alpha Vantage (news sentiment, RSI, MACD) |
| **Information Retrieval** | BM25 (rank_bm25) for RAG memory retrieval |
| **Data Processing** | Pandas, JSON, Regex-based LLM output parsing |
| **Infrastructure** | Google Colab (T4/L4 GPU), Google Drive (caching & model storage) |
| **languages** | Python |

## Results & Outputs

### Portfolio & Live Trading
- **Paper trading account**: ~$99K equity across 12 stocks
- **Active positions**: Multi-asset portfolio including AAPL, AMZN, GOOG, GS, NVDA
- **Trading loop**: 80 iterations with 10s intervals, autonomous decision-making with tool calls

### Backtesting
- Simulated multi-day backtest on historical data (AAPL, NVDA, MSFT, SPY)
- Tracked per-day equity, trade count, and tool call usage
- Collected 8 profitable (positive PnL) trading episodes for SFT training data

### Fine-Tuning (LoRA)
- **Trainable parameters**: 544,768 / 1,544,259,072 (0.035%)
- **Training data**: 8 curated profitable trade episodes from backtest
- **Epochs**: 2-epoch SFT training
- **Post-fine-tuning**: Re-ran backtest to compare base vs. fine-tuned model performance

### RAG Memory
- 9 stored trading memories with BM25 retrieval
- Automated trade reflection with lesson extraction
- Memory-augmented prompts improve decision quality over time

## Getting Started

### Prerequisites
- Google Colab account (T4/L4 GPU recommended)
- [Alpaca Markets](https://alpaca.markets/) API key (free paper trading)
- [Alpha Vantage](https://www.alphavantage.co/) API key (free tier available)

### Setup
1. Open `Trading_Agent_final.ipynb` in Google Colab
2. Add your API keys via **Colab Secrets** (🔑 icon in sidebar):
   - `ALPACA_API_KEY` / `ALPACA_SECRET_KEY`
   - `ALPHA_VANTAGE_KEY`
3. Run cells sequentially through 5 tasks:
   - **Task 1**: Environment setup, LLM loading, Alpaca connection, news retrieval, tool registry
   - **Task 2**: Implement MCP tool functions (price, RSI, MACD, price history)
   - **Task 3**: RAG memory system with BM25 retrieval and trade reflection
   - **Task 4**: SFT fine-tuning with LoRA on backtest data
   - **Task 5**: Full autonomous trading loop integration

## License

This project is for educational and research purposes only. Not financial advice.
