---
name: sync_remote
arguments:
  - name: skill_url
  - name: command_url
---

1.Replace local OpenCode skills and commands with remote versions from {{skill_url}} and {{command_url}}.
2.Replace ~/.opencode/bin/telegram_bot.py with https://github.com/beacoder/telegram_bot/blob/main/telegram_bot.py
3.Replace all files in ~/.opencode/bin/bot with remote versions from https://github.com/beacoder/telegram_bot/tree/main/bot

Default values:
- skill_url: https://github.com/beacoder/llm/tree/main/opencode/skills
- command_url: https://github.com/beacoder/llm/tree/main/opencode/commands
