---
name: universal_media_search_strategy
description: |
  Use this skill when searching for videos, articles, social posts,
  discussions, reviews, tutorials, guides, or other media content
  across multiple platforms.

  Especially useful for:
  - finding high-quality learning resources
  - collecting opinions or discussions
  - discovering tutorials or explainers
  - researching topics across multiple sources
  - aggregating media results from heterogeneous platforms

  Performs:
  - iterative search
  - query expansion
  - semantic filtering
  - deduplication
  - relevance ranking

  Returns normalized, ranked, platform-independent results.
---

arguments:
  - name: topic
  - name: limit
  - name: pages

default values:
  - limit: 5
  - pages: 5

## Step 1 — Primary Search
- Execute initial search using raw query: "{{topic}}" with Chrome (fallback to websearch)
- Collect results across {{pages}} pages (platform-defined pagination)
- Normalize results into a standard schema:
  - title
  - url
  - author/channel/source
  - views/engagement (if available)
  - date
  - snippet/description

## Step 2 — Failure Detection
Trigger fallback if:
- results < {{limit}}
- low relevance density
- too many duplicates or near-duplicates

## Step 3 — Query Expansion
Decompose query into:
- core intent
- subtopics
- synonyms
- alternative phrasing
- content-type transformations:
  - explained
  - tutorial
  - guide
  - overview
  - review
  - "what is"

Generate 3–6 expanded queries.

## Step 4 — Multi-Query Search
- Run searches for all expanded queries
- Merge results
- Deduplicate by canonical URL

## Step 5 — Relevance Filtering
Keep only results that:
- strongly match intent
- are semantically relevant (not keyword-only matches)

Discard:
- clickbait
- unrelated trending content
- duplicates

## Step 6 — Ranking
Rank by:
1. semantic relevance
2. engagement (views/likes/upvotes)
3. recency

## Step 7 — Output
Return top {{limit}} normalized results in a table.
