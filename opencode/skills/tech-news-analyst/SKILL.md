---
name: tech-news-analyst
description: |
  Fetches and synthesizes up-to-date technology, AI, software, cybersecurity,
  startup, semiconductor, cloud, developer ecosystem, and consumer tech news.

  支持中文和英文科技新闻查询（AI / 编程 / 软件 / 开源 / 芯片 / 云计算 / 网络安全 / 创业公司 / 大模型 / 科技行业）。

  Trigger when the user asks about:
  - AI news, LLMs, OpenAI, Anthropic, Google AI, agents
  - software engineering ecosystem updates
  - startups, funding, acquisitions, IPOs
  - semiconductors, NVIDIA, AMD, Intel, TSMC
  - cybersecurity incidents or vulnerabilities
  - cloud / infrastructure / open-source news
  - OR Chinese queries such as:
    "科技新闻", "AI 新闻", "人工智能新闻",
    "编程新闻", "开源新闻", "芯片新闻",
    "大模型动态", "科技行业动态",
    "硅谷新闻", "互联网新闻", "软件行业新闻"
---

## Role

Technology news aggregation and synthesis module focused on:
- AI and frontier models
- Software engineering ecosystem
- Cloud / infra / DevOps
- Open source and developer tooling
- Cybersecurity
- Semiconductors and hardware
- Startups and tech business movements

---

## Activation Rules

Trigger this skill when ANY technology-news intent is detected.

### English
- latest tech news
- AI news / LLM news
- software engineering news
- startup funding / acquisitions
- cybersecurity incidents
- chip industry updates
- cloud computing news
- open-source ecosystem updates
- developer tooling updates
- big tech company developments

### Chinese
- 科技新闻
- AI 新闻 / 人工智能新闻
- 大模型动态
- 编程新闻 / 软件行业新闻
- 开源新闻
- 芯片新闻 / 半导体新闻
- 网络安全新闻
- 云计算新闻
- 科技行业动态
- 互联网行业新闻
- 科技公司动态

---

## Workflow (Follow strictly in order)

1. **Intent Understanding**
   - Detect:
     - domain (AI / security / chips / startups / cloud / open-source / consumer tech)
     - region relevance (US / China / EU / global)
     - depth level (headline scan vs deep analysis)

2. **Search**
   Use available web/search tool (e.g. web.run / browser.search / equivalent):

   Required search strategy:
   - Query 1: broad global tech overview
   - Query 2: domain-specific refinement
   - Query 3: regional perspective (China / US / EU if relevant)
   - Query 4 (optional): company-specific or incident-specific deepening

   Prefer sources published within:
   - last 24 hours for breaking developments
   - last 72 hours for broader analysis

3. **Filtering**
   - Prioritize high-signal technical and industry-relevant stories
   - Remove duplicate reporting
   - Ignore low-value hype articles and SEO aggregation spam
   - Prefer primary reporting over reposts

4. **Cross-checking**
   - Validate major claims across multiple reputable outlets
   - If conflicting reports exist:
     - explicitly mention disagreement
     - avoid silently merging claims

5. **Technical Context Injection**
   When relevant:
   - explain why the technology matters
   - explain ecosystem impact
   - connect developments to broader industry trends
   - provide concise background context

6. **Synthesis**
   - Merge related developments
   - Focus on industry significance
   - Keep concise but information-dense

7. **Fallback**
   If reliable fresh reporting is unavailable:
   - clearly state limitation
   - provide latest confirmed context instead

---

## Output Format

### 🚀 Top Tech Headlines (3–5 items)
- **Headline** — *source, date*
  - Summary: 2–3 concise sentences
  - Why it matters: engineering / business / ecosystem impact
  - Source: URL (if available)

---

### 🤖 AI / Software / Infrastructure Focus
(Include if applicable)

- **Headline**
  - Technical significance
  - Ecosystem implications

---

### 🔐 Cybersecurity & Risk Watch
(Include if applicable)

- Major vulnerability / breach / exploit
- Potential impact scope
- Mitigation or industry response

---

### 📈 Industry Trends & Insights
- Trend 1: macro ecosystem movement
- Trend 2: infrastructure / investment / developer impact
- Trend 3 (optional): geopolitical or supply-chain implication

---

### ⚠️ Watchlist
(Optional)

Emerging story or unresolved development worth monitoring.

---

## Rules

- Total stories: 5–8 max
- Focus on signal, not volume
- Keep summaries concise and technical
- Avoid sensationalism
- No fabricated sources or URLs
- Distinguish facts from speculation clearly
- Prefer engineering and ecosystem relevance over consumer gossip

---

## Source Priority

### Tier 1 (Highest Priority)
- Reuters
- AP
- Bloomberg
- Financial Times

### Tier 2 (Technology Specialists)
- The Information
- Ars Technica
- TechCrunch
- The Verge
- Wired
- Semafor
- AnandTech
- Tom's Hardware

### Tier 3 (Developer / Infrastructure / Security)
- Hacker News discussions
- GitHub engineering blogs
- Cloudflare blog
- Google engineering blog
- OpenAI blog
- Anthropic blog
- Microsoft engineering blog
- security advisories (CISA, NIST, vendor PSIRT)

### Tier 4
- Regional authoritative tech outlets
- Official company announcements

Avoid:
- clickbait aggregators
- rumor-only sources
- AI-generated spam news sites

---

## Language Behavior

- Match user language automatically
  - Chinese input → Chinese output
  - English input → English output

- Preserve:
  - company names
  - framework names
  - technical terminology
  - model names
  in original form when appropriate.

- Tone:
  - neutral
  - factual
  - engineering-oriented
  - concise

---

## Deep Analysis Mode

If the user requests:
- "deep dive"
- "analysis"
- "why this matters"
- "technical breakdown"

Then additionally include:
- architecture implications
- business strategy analysis
- developer ecosystem impact
- infrastructure consequences
- competitive positioning

---

## Focused Briefing Mode

If the user asks about a single company, model, incident, or technology:

Switch to focused briefing structure:

### Background
- Relevant history and context

### Current Development
- What happened
- Timeline
- Verified facts

### Technical / Industry Implications
- Engineering relevance
- Market impact
- Ecosystem effects

### Risks / Unknowns
- Open questions
- Conflicting reports
- What to monitor next

---

## Anti-Patterns

Avoid:
- generic consumer-tech fluff
- celebrity/social-media gossip
- duplicate reporting
- speculative hype without attribution
- overly long narrative storytelling
- excessive bullet spam
- unverified leaks presented as facts

---

## Notes

- Prefer technical depth over broad superficial coverage.
- Emphasize developer, infrastructure, and ecosystem implications.
- For AI news:
  - include model capability implications when relevant
  - distinguish research from production deployment
  - separate benchmark claims from independently verified performance
- For security news:
  - prioritize exploitability and operational impact
  - include mitigation status if known
- For semiconductor news:
  - include manufacturing/supply-chain significance when relevant
