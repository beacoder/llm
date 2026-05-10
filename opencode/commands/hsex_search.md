---
name: hsex_search
arguments:
  - name: topic
  - name: number_of_videos
  - name: number_of_pages
---

Use skill: universal_media_search_strategy

Platform adapter: https://hsex.tv/

Execution rules:
- Pagination: {{number_of_pages}} pages
- Engagement metric: view count
- Source field mapping:
  - title → video title
  - url → video link
  - source → channel name
  - views → view count
  - date → upload date

Parameters mapping:
- topic = "{{topic}}"
- pages = {{number_of_pages}}
- limit = {{number_of_videos}}

Return output strictly formatted per skill output schema.
