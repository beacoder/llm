# Telegram ↔ OpenCode Agent Bridge

Overview
--------

This project provides a lightweight Telegram bot that connects Telegram directly to an OpenCode agent backend.

It allows you to interact with OpenCode remotely through Telegram while supporting:

- Persistent AI sessions
- File uploads and downloads
- Multiple model switching
- Scheduled tasks
- Proxy environments
- Remote automation workflows

The system is designed for personal AI operations, remote agent access, and lightweight self-hosted automation.

Architecture
------------

```text
Telegram → Python Bot → OpenCode Agent
                          ↓
                    Local Workspace
                          ↓
                 Files + Agent Output
                          ↓
                   Telegram Response
```

Key Features
------------

### Remote AI Agent Access
Use Telegram as a mobile interface for interacting with your OpenCode agent remotely.

Supports:
- Conversational workflows
- Long-running sessions
- Persistent context
- Remote prompting

### Persistent Sessions
The bot maintains session continuity between messages, allowing ongoing conversations and multi-step workflows.

### File Support
Supports uploading files directly from Telegram, including:
- Documents
- Images
- Videos
- Audio

Generated files can also be automatically returned back to Telegram.

### Multiple AI Models
Switch between different configured models directly from Telegram commands.

Useful for:
- Fast responses
- Higher quality reasoning
- Cost optimization

### Built-in Task Scheduler
Includes a lightweight scheduling system for automated agent execution.

Possible use cases:
- Daily summaries
- Periodic research
- Automated reports
- Scheduled monitoring tasks
- Reminder agents

### Proxy Support
Designed to work reliably in restricted or proxied network environments.

### Local Workspace
All agent operations occur inside a local workspace directory, making:
- File handling simple
- Outputs persistent
- Agent workflows transparent

### Safe Single-Task Execution
Includes task locking to prevent overlapping agent executions and session corruption.

Use Cases
---------

This project is suitable for:

- Personal AI assistant hosting
- Remote OpenCode access
- Mobile AI workflows
- Lightweight AI automation
- Self-hosted agent systems
- Proxy-restricted environments
- AI-powered file processing
- Automated scheduled tasks

Typical Workflow
----------------

1. Send a message or file via Telegram
2. Bot forwards request to OpenCode
3. Agent processes the request
4. Response and generated files are returned to Telegram

Example workflows:
- Summarizing uploaded PDFs
- Generating reports
- Running coding agents remotely
- Processing images or documents
- Performing scheduled AI tasks

Requirements
------------

Main requirements:
- Python 3
- OpenCode CLI
- Telegram bot token
- Properly configured OpenCode environment

Optional:
- HTTP/SOCKS proxy support

Platform Notes
--------------

The project is primarily intended for Linux/Unix-like environments.

Best suited for:
- VPS deployments
- Personal servers
- Home lab environments
- Remote Linux machines

Security Notes
--------------

This project is intentionally designed as a single-user personal automation tool.

Important considerations:
- Restrict access to trusted Telegram accounts only
- Never expose your Telegram bot token
- Run in trusted environments
- Avoid running with unrestricted permissions on sensitive systems

Design Philosophy
-----------------

This implementation prioritizes:

- Simplicity
- Reliability
- Minimal infrastructure
- Local-first execution
- Easy deployment
- OpenCode compatibility

Instead of building a heavy distributed system, the project focuses on creating a practical and robust bridge between Telegram and OpenCode.

Limitations
-----------

Current limitations include:
- Single-user design
- Single active task execution
- No real-time streaming output
- Local filesystem dependency
- Lightweight scheduler only

Future Improvements
-------------------

Potential future enhancements:
- Multi-user support
- Streaming responses
- Better task management
- Web dashboard
- Docker isolation
- Enhanced scheduling features
- Session management improvements

Author Intent
-------------

This project is intended as a lightweight personal AI operations bridge for users who want simple, reliable, and remote access to OpenCode through Telegram without requiring complex infrastructure.
