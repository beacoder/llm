---
name: youtube_search
arguments:
  - name: topic
  - name: number_of_videos
  - name: number_of_pages
---

Use skill: universal_media_search_strategy

Platform adapter: YouTube

Execution rules:
- Search endpoint: https://www.youtube.com/results?search_query={{topic}}
- Pagination: {{number_of_pages}} pages
- Result extraction: YouTube video cards only
- Engagement metric: view count
- Source field mapping:
  - title → video title
  - url → video link
  - source → channel name
  - views → view count
  - date → upload date

Parameters mapping:
- query = "{{topic}}"
- pages = {{number_of_pages}}
- limit = {{number_of_videos}}

Return output strictly formatted per skill output schema.
