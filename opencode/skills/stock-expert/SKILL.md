---
name: a-share-source-opinion-collector
description: |
    聚合主流中文财经网站与社区对 A 股市场的【主观观点、情绪倾向、趋势预期】。
    专注"怎么看"，不提供"是多少"类客观数据。

    激活条件（满足任一）：
    - 询问市场/板块/个股的【情绪、观点、共识、预期】
    - 含关键词：怎么看 / 情绪 / 观点 / 多空 / 共识 / 预期 / 研判 / 雪球 / 股吧 / 财联社
    - 且【不包含】具体股票代码(6 位数字)或纯数据指标词
---

# A-share opinion aggregation module

## Workflow

### 1. Intent Understanding

Detect:
- market/index/sector
- timeframe:
  - intraday
  - tomorrow
  - short-term
- analysis type:
  - technical
  - macro
  - sentiment
  - capital flow

---

### 2. Data Collection

Use `chrome` to inspect ALL sources:

- https://cfi.cn/
- https://stock.hexun.com/
- https://stock.jrj.com.cn/
- https://xueqiu.com
- https://guba.eastmoney.com/
- https://finance.sina.com.cn/stock/
- https://www.wlstock.com/

For each source extract:
- bullish / bearish / neutral opinion
- core reasoning
- discussed sectors
- sentiment intensity
- risk warnings

Prefer:
- latest trading day
- latest 24h content

IMPORTANT:
- Every collected opinion MUST include source attribution.
- Every conclusion MUST indicate which website it came from.
- If multiple sources support the same view, list all supporting sources.

Example:
- AI sector remains strong due to capital inflow [Xueqiu][Sina Finance]
- Market may enter short-term correction after weak volume [Hexun][JRJ]

---

### 3. Filtering & Validation

Ignore:
- clickbait
- paid荐股
- “内幕消息”
- rumor-only posts

Prioritize:
- detailed reasoning
- repeated narratives
- high-signal analysis
- multi-source confirmation

Confidence:
- 3+ sources agree:
  - `✅ Strong consensus`
- single-source only:
  - `💬 Limited confirmation`

---

### 4. Sentiment Aggregation

Estimate:
- bullish ratio
- bearish ratio
- neutral ratio

Track:
- hottest sectors
- recurring keywords
- emotional temperature:
  - panic
  - cautious
  - optimistic
  - euphoric

---

### 5. Output Format

## Market Sentiment Snapshot
- overall sentiment
- bullish/bearish ratio
- dominant narrative
- sources supporting it

---

## Source-by-Source Opinions

### CFI
- viewpoint [CFI]
- reasoning [CFI]

### Hexun
- viewpoint [Hexun]
- reasoning [Hexun]

### JRJ
- viewpoint [JRJ]
- reasoning [JRJ]

### Xueqiu
- investor sentiment [Xueqiu]

### Eastmoney Guba
- retail sentiment [Eastmoney Guba]

### Sina Finance
- institutional narrative [Sina Finance]

### WLStock
- technical trend analysis [WLStock]

---

## Consensus Views
- most agreed opinions
- supporting sources
- confidence level

Example:
- AI remains strongest market direction [Xueqiu][Sina Finance][CFI]
  ✅ Strong consensus

---

## Bullish Arguments
- argument
- supporting sources

## Bearish Arguments
- argument
- supporting sources

---

## Sector Heatmap

| Sector | Sentiment | Heat | Sources |
|---|---|---|---|

---

## Risk Watch
- liquidity risks [source]
- overcrowded trades [source]
- macro uncertainty [source]

---

## Final Synthesis
- short-term expectation
- sentiment regime
- what traders are watching

All conclusions MUST include source attribution.

---

## Rules

- Use `chrome` for browsing
- Open every source directly
- Prefer latest market-session content
- Distinguish facts from opinions
- Do NOT provide direct investment advice
- Do NOT present speculation as fact
- Focus on consensus over hype
- NEVER output claims without source attribution

---

## Language

- Chinese input → Chinese output
- English input → English output

Tone:
- concise
- analytical
- neutral
