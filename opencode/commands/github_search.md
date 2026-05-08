---
name: github_search
arguments:
  - name: query
  - name: number_of_results
  - name: number_of_pages
---

Use skill: universal_code_search_strategy

Platform adapter: GitHub

Execution rules:
- Primary endpoint: GitHub search (repositories + code search)
- Search types:
  - repositories
  - code files (if supported)
- Pagination: {{number_of_pages}}

Mapping rules:
- query = "{{query}}"
- pages = {{number_of_pages}}
- limit = {{number_of_results}}

GitHub-specific scoring signals:
- stars (high weight)
- forks (medium weight)
- last updated (recency bonus)
- README presence (quality boost)

Output format:
- repository name
- repository URL
- description
- primary language
- stars / forks
- last update date
- relevant snippet (if code search result)
