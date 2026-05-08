---
name: youtube_search
arguments:
  - name: topic
  - name: number_of_videos
  - name: number_of_pages
---

chrome, in https://www.youtube.com/ search "{{topic}}" across {{number_of_pages}} pages of results.

## Step 1 — Primary Search
- Perform the initial search using the raw topic: "{{topic}}"
- Collect video results across {{number_of_pages}} pages
- Rank all results by view count (descending)

## Step 2 — Failure Detection
If ANY of the following occurs:
- fewer than {{number_of_videos}} relevant results are found
- results are too sparse or repetitive
- results appear off-topic or low-confidence relevance

THEN trigger fallback strategy below.

## Step 3 — Query Decomposition & Expansion
Decompose the original topic into:
- core intent keywords (main subject)
- subtopics (related concepts, components, entities)
- alternative phrasing (synonyms, abbreviations, common names)
- category-based reinterpretation (e.g., tutorial, review, explanation, news, compilation)

Generate 3–6 expanded queries, for example:
- "{{topic}} explained"
- "{{topic}} tutorial"
- "{{topic}} guide"
- "what is {{topic}}"
- "{{topic}} overview"
- related entity or concept variants

## Step 4 — Multi-Query Re-Search
- Run YouTube search for each expanded query across {{number_of_pages}} pages
- Merge all results into a unified pool
- Deduplicate by video URL

## Step 5 — Relevance Filtering (STRICT)
Only keep videos that:
- clearly match the original intent of "{{topic}}"
- are semantically related (not just keyword matches)
- are not tangential or unrelated clickbait content

Discard:
- loosely related content
- generic trending videos unrelated to topic intent
- duplicate or near-duplicate uploads

## Step 6 — Ranking
Rank remaining videos by:
1. relevance score to original topic (primary)
2. view count (secondary)
3. recency (tie-breaker: newer preferred)

## Step 7 — Output
Return the top {{number_of_videos}} videos with highest ranked score, including:

- video title
- video URL
- channel name
- view count
- upload date
