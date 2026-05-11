---
name: sync_remote
arguments:
  - name: skill_url
  - name: command_url
---

## task
Replace local OpenCode skills and commands with remote versions from {{skill_url}} and {{command_url}}.
Replace ~/.opencode/bin/telegram_bot.py with https://github.com/beacoder/telegram_bot/blob/main/telegram_bot.py
Replace all files in ~/.opencode/bin/bot with remote versions from https://github.com/beacoder/telegram_bot/tree/main/bot

## workflow
- For each file in remote:
  - Download to temp file
  - Compare with local verion
    - If identical → skip
    - If different or missing → replace

Default values:
- skill_url: https://github.com/beacoder/llm/tree/main/opencode/skills
- command_url: https://github.com/beacoder/llm/tree/main/opencode/commands
