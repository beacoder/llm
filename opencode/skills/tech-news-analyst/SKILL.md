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

## Workflow (Follow strictly in order)

1. **Intent Understanding**
   - Detect:
     - domain (AI / security / chips / startups / cloud / open-source / consumer tech)
     - region relevance (US / China / EU / global)
     - depth level (headline scan vs deep analysis)
     - recency urgency: "what's happening right now" → bias to last 6h; "latest news" → last 24-72h; "recent developments" → last week
   - **Example**: User says "Nvidia news" → domain=chips/AI, region=US, depth=headline scan, recency=72h

2. **Search**
   Use `websearch` (or equivalent search tool). Do NOT assume any other tool exists.

   Search strategy:
   - Query 1: broad global tech overview (e.g. `"tech news" "2026" site:reuters.com OR site:techcrunch.com`)
   - Query 2: domain-specific refinement (e.g. `"AI" "funding" OR "LLM"`)
   - Query 3: regional perspective (China / US / EU if relevant)
   - Query 4 (optional): company-specific or incident-specific deepening

   Fetch 5-8 results per query minimum. If a query returns <3 relevant results, adjust and retry.

   Prefer sources published within:
   - last 24 hours for breaking developments
   - last 72 hours for broader analysis
   - 1 week older only if essential context is missing from fresher stories

   Recency mix: aim for ~60% ≤24h, ~30% 24-72h, ~10% context/background.

3. **Filtering & Deduplication**
   - Prioritize high-signal technical and industry-relevant stories
   - When multiple sources cover the same story: keep the most detailed/authoritative one, merge unique details from others into it
   - Ignore low-value hype articles and SEO aggregation spam
   - Prefer primary reporting over reposts
   - Ranking heuristic: original reporting > official announcement > in-depth analysis > news brief > aggregation
   - If a story seems important but only 1-2 sources cover it, include it with a note about limited coverage

4. **Story Prioritization (when you have more stories than slots)**
   Score each story on 5 factors and keep the highest-scoring ones:
   - **Freshness** (0-3): breaking today=3, this week=2, this month=1
   - **Impact** (0-3): changes industry landscape=3, major product/release=2, incremental update=1
   - **Source authority** (0-2): Tier 1=2, Tier 2=1, Tier 3+=0
   - **Ecosystem relevance** (0-2): affects developers/infra directly=2, business-only=1, consumer-only=0
   - **Signal vs noise** (-1 to 1): real development=1, speculative but notable=0, obvious hype/spam=-1
   - Drop any story with a total score <3 unless it fills a critical domain gap

5. **Cross-checking**
   - Validate major claims across multiple reputable outlets
   - If conflicting reports exist:
     - explicitly mention disagreement
     - avoid silently merging claims

6. **Technical Context Injection**
   When relevant:
   - explain why the technology matters
   - explain ecosystem impact
   - connect developments to broader industry trends
   - provide concise background context

7. **Synthesis**
   - Merge related developments (e.g., multiple funding rounds → "startup funding surge" narrative)
   - Focus on industry significance
   - Keep concise but information-dense
   - Order stories by importance, not chronologically
   - **Example**: If both OpenAI and Anthropic announce enterprise JVs on the same day, merge them into a single item about "AI labs pivot to enterprise" rather than two separate stories

8. **Fallback**
   If reliable fresh reporting is unavailable:
   - clearly state limitation and why (e.g., "weekend lull", "story still developing")
   - provide latest confirmed context instead
   - if a major developing story has very few sources, include it with a "[Developing]" tag rather than dropping it

---

## Output Format

### Top Tech Headlines (3–5 items)
- **Headline** — *source, date*
  - Summary: 2–3 concise sentences
  - Why it matters: engineering / business / ecosystem impact
  - Source: URL (if available)

---

### AI / Software / Infrastructure Focus
(Include if applicable — merge with headlines section if there is too much overlap)

- **Headline**
  - Technical significance
  - Ecosystem implications

---

### Cybersecurity & Risk Watch
(Include if applicable)

- Major vulnerability / breach / exploit
- Potential impact scope
- Mitigation or industry response

---

### Industry Trends & Insights
- Trend 1: macro ecosystem movement
- Trend 2: infrastructure / investment / developer impact
- Trend 3 (optional): geopolitical or supply-chain implication

---

### Watchlist
(Optional — include when a story is clearly early-stage but significant)

Criteria for watchlist inclusion:
- Story has <3 sources but potential for large impact
- Regulatory/legal proceeding is pending
- Rumored announcement from a reliable leaker
- Developing incident with unclear scope

Format: "What to watch" + "Signal to look for" (1 sentence each)

---

## Rules

- Total stories: 5–8 max
- Focus on signal, not volume
- Keep summaries concise and technical
- Avoid sensationalism
- Distinguish facts from speculation clearly
- Prefer engineering and ecosystem relevance over consumer gossip
- If a key story has conflicting reports across sources, surface the disagreement — do not silently pick one version
- Do not include a story unless you have at least one verifiable source with a timestamp

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
- 36Kr / 36氪 (Chinese tech/finance)
- LatePost / 晚点LatePost (Chinese tech)

### Caution for Tier 1/2 outlets:
- Reuters, Bloomberg, FT are often paywalled — agent may only see snippets
- When paywalled, cross-reference summary with a Tier 2/3 source before reporting claims

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
