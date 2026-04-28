#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
from telegram import Update
from telegram.ext import (
    ApplicationBuilder,
    ContextTypes,
    MessageHandler,
    filters,
)
import logging
import asyncio
import shutil
import json
from datetime import datetime, timedelta
from calendar import monthrange

# ================= CONFIG =================
TOKEN = 'XXXXXXXXXX'
AUTHORIZED_USER_ID = 123456789

PROXY_URL = "http://127.0.0.1:1080"

TELEGRAM_MAX_LENGTH = 4000
AGENT_SCHEDULE_FILE = os.path.expanduser("/home/huming/agent/schedule.json")
AGENT_LOCK_FILE = os.path.expanduser("/home/huming/agent/.lock")
AGENT_WORK_DIR = os.path.expanduser("/home/huming/agent")
AGENT_MEDIA_DIR = os.path.expanduser("/home/huming/agent/media-file/")
OPENCODE_TIMEOUT = 300
SESSION_MARKER = os.path.join(AGENT_WORK_DIR, ".session_started")
# =========================================

CLEAR_SESSION_COMMAND = "clear"

os.makedirs(AGENT_WORK_DIR, exist_ok=True)
os.makedirs(AGENT_MEDIA_DIR, exist_ok=True)

logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)

def clear_agent_session():
    if os.path.exists(SESSION_MARKER):
        os.remove(SESSION_MARKER)

async def start_agent(prompt: str) -> str:
    use_continue = os.path.exists(SESSION_MARKER)

    cmd = [
        "opencode", "run",
        "--dangerously-skip-permissions",
    ]
    if use_continue:
        cmd.append("--continue")
    cmd.append(prompt)

    env = os.environ.copy()
    env["HTTP_PROXY"] = PROXY_URL
    env["HTTPS_PROXY"] = PROXY_URL

    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        env=env,
        cwd=AGENT_WORK_DIR
    )

    try:
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=OPENCODE_TIMEOUT)
    except asyncio.TimeoutError:
        proc.kill()
        await proc.wait()
        return "❌ Agent timed out."

    open(SESSION_MARKER, "w").close()

    output = stdout.decode().strip()

    if proc.returncode != 0:
        error_output = stderr.decode().strip()
        if error_output:
            logging.error(f"opencode error (rc={proc.returncode}): {error_output}")

    if not output:
        return "⚠️ Agent returned empty response."

    return output

async def send_text(text: str, update: Update = None, app = None):
    if not text or not text.strip():
        return

    chunks = [
        text[i:i + TELEGRAM_MAX_LENGTH]
        for i in range(0, len(text), TELEGRAM_MAX_LENGTH)
    ]

    for chunk in chunks:
        if update:
            await update.message.reply_text(chunk)
        elif app:
            await app.bot.send_message(
                chat_id=AUTHORIZED_USER_ID,
                text=chunk)
        await asyncio.sleep(0.6)

async def send_files(update: Update = None, app = None):
    if not os.path.exists(AGENT_MEDIA_DIR):
        return

    files = sorted(
        [os.path.join(AGENT_MEDIA_DIR, f) for f in os.listdir(AGENT_MEDIA_DIR)],
        key=os.path.getmtime
    )

    for path in files:
        if not os.path.isfile(path):
            continue

        try:
            with open(path, "rb") as f:
                if update:
                    await update.message.reply_document(document=f)
                elif app:
                    await app.bot.send_document(
                        chat_id=AUTHORIZED_USER_ID,
                        document=f)
        except Exception as e:
            await send_text(f"❌ Failed to send file: {path}", update, app)

def cleanup():
    for item in os.listdir(AGENT_MEDIA_DIR):
        item_path = os.path.join(AGENT_MEDIA_DIR, item)
        if os.path.isdir(item_path):
            shutil.rmtree(item_path)
        else:
            os.remove(item_path)

def load_schedule():
    if not os.path.exists(AGENT_SCHEDULE_FILE):
        return []
    try:
        with open(AGENT_SCHEDULE_FILE, "r") as f:
            content = f.read().strip()
            if content:
                return json.loads(content)
            return []
    except Exception as e:
        print(f"[schedule] load failed: {e}")
        return []

def update_schedule(tasks):
    tmp_file = AGENT_SCHEDULE_FILE + ".tmp"
    with open(tmp_file, "w") as f:
        json.dump(tasks, f, indent=2)
    os.replace(tmp_file, AGENT_SCHEDULE_FILE)

def is_due(task):
    if task.get("done"):
        return False
    run_at = datetime.strptime(task["run_at"], "%Y-%m-%d %H:%M")
    return datetime.now() >= run_at

def compute_next_run(task):
    repeat = task.get("repeat")
    if not repeat:
        return None
    current_run = datetime.strptime(task["run_at"], "%Y-%m-%d %H:%M")
    if repeat == "daily":
        next_run = current_run + timedelta(days=1)
    elif repeat.startswith("weekly:"):
        weekday = int(repeat.split(":")[1])
        next_run = current_run + timedelta(days=1)
        while next_run.isoweekday() != weekday:
            next_run += timedelta(days=1)
    elif repeat.startswith("monthly:"):
        day = int(repeat.split(":")[1])
        year, month = current_run.year, current_run.month
        month += 1
        if month > 12:
            month = 1
            year += 1
        max_day = monthrange(year, month)[1]
        next_day = min(day, max_day)
        next_run = current_run.replace(year=year, month=month, day=next_day)
    elif repeat.startswith("interval:"):
        val = repeat.split(":")[1]
        if val.endswith("m"):
            minutes = int(val[:-1])
            next_run = current_run + timedelta(minutes=minutes)
        elif val.endswith("h"):
            hours = int(val[:-1])
            next_run = current_run + timedelta(hours=hours)
        else:
            next_run = None
    else:
        next_run = None
    if next_run:
        return next_run.strftime("%Y-%m-%d %H:%M")
    return None

async def run_task(task, app):
    if os.path.exists(AGENT_LOCK_FILE):
        await send_text("⚠️ Another task running, waiting in queue", None, app)
        return

    open(AGENT_LOCK_FILE, "w").close()

    prompt = task["prompt"]

    try:
        await send_text(f"🚀 Running task: {prompt}", None, app)
        cleanup()
        await send_text("🧠 Thinking...", None, app)
        response = await start_agent(prompt)
        await send_text(response, None, app)
        await send_files(None, app)
        await send_text("✅ Agent finished.", None, app)
    finally:
        if os.path.exists(AGENT_LOCK_FILE):
            os.remove(AGENT_LOCK_FILE)

async def check_and_run_tasks(app):
    tasks = load_schedule()
    updated = []

    for task in tasks:
        if is_due(task):
            await run_task(task, app)
            next_run = compute_next_run(task)
            if next_run:
                task["run_at"] = next_run
                task["done"] = False
            else:
                task["done"] = True
        updated.append(task)

    if updated:
        update_schedule(updated)

async def scheduler_loop(app):
    while True:
        try:
            await check_and_run_tasks(app)
        except Exception as e:
            await send_text(f"❌ Scheduler error: {e}", None, app)
        await asyncio.sleep(30)

# ---------- Telegram handlers ----------
async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if update.message.from_user.id != AUTHORIZED_USER_ID:
        await send_text("❌ Unauthorized.", update)
        return

    if os.path.exists(AGENT_LOCK_FILE):
        await send_text("⚠️ Another task running, skip the queue", update)

    prompt = update.message.text

    if prompt.lower().strip() == CLEAR_SESSION_COMMAND:
        clear_agent_session()
        await send_text("✅ Session cleared. Next message starts fresh.", update)
        return

    open(AGENT_LOCK_FILE, "w").close()

    try:
        cleanup()
        await send_text("🧠 Thinking...", update)
        response = await start_agent(prompt)
        await send_text(response, update)
        await send_files(update, None)
        await send_text("✅ Agent finished.", update)
    finally:
        if os.path.exists(AGENT_LOCK_FILE):
            os.remove(AGENT_LOCK_FILE)

# ---------- Main ----------
def main():
    app = (
        ApplicationBuilder()
        .token(TOKEN)
        .proxy(PROXY_URL)
        .build()
    )

    app.add_handler(MessageHandler(filters.TEXT & (~filters.COMMAND), handle_message))

    async def error_handler(update, context):
        logging.error(f"Exception: {context.error}")
    app.add_error_handler(error_handler)

    async def _post_init(app):
        asyncio.create_task(scheduler_loop(app))
        await send_text("🚀 Agent ready (opencode backend).", None, app)
    app.post_init = _post_init

    app.run_polling()


if __name__ == "__main__":
    main()
