---
name: youtube_search
arguments:
  - name: topic
  - name: number_of_videos
  - name: number_of_pages
---

chrome, in https://www.youtube.com/ search "{{topic}}" across {{number_of_pages}} pages of results.

Collect video results and rank them by view count.

Return the top {{number_of_videos}} videos with the highest views, including:
- video title
- video URL
- channel name
- view count
- upload date
