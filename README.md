# Trading Agent

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python) ![NLP](https://img.shields.io/badge/NLP-Sentiment-blueviolet) ![Finance](https://img.shields.io/badge/Finance-Live%20APIs-success) ![License](https://img.shields.io/badge/license-MIT-green)

> Autonomous trading agent that fuses live market data with NLP sentiment analysis to generate real-time buy/sell signals.

## Overview

Markets move on information, not just price. This agent ingests live equity price feeds alongside streaming financial news, scores article sentiment using NLP, and synthesizes both signals into actionable buy/sell/hold decisions — removing human latency from the decision loop. The system is designed to be extended with additional data sources or strategy layers.

## Tech Stack

| Component | Technology |
|---|---|
| Data Ingestion | Live market APIs, financial news feeds |
| NLP / Sentiment | VADER / Transformer-based sentiment classifier |
| Signal Logic | Rule-based fusion of price momentum + sentiment score |
| Execution Interface | Simulated order output (paper trading) |
| Analysis | pandas, Jupyter Notebook |

## How It Works

- **Market feed** streams real-time OHLCV price data via REST/WebSocket API
- **News feed** polls financial news APIs and extracts headlines per ticker
- **Sentiment scorer** assigns polarity scores (positive / negative / neutral) per article
- **Signal engine** combines sentiment score with price momentum to produce trade signals
- **Output** logs timestamped BUY/SELL/HOLD decisions with confidence scores

## Quick Start

```bash
git clone https://github.com/shahidbaig-shaik/trading-agent
cd trading-agent
pip install jupyter pandas requests
jupyter notebook Trading_Agent_final.ipynb
```
