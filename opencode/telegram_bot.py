#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import logging
import asyncio
import shutil
from datetime import datetime, timedelta
from calendar import monthrange

from telegram import Update
from telegram.ext import (
    ApplicationBuilder,
    ContextTypes,
    MessageHandler,
    filters,
)

# ==============================================================================
# CONFIG
# ==============================================================================

TOKEN = 'XXXXXXXXXX' 
AUTHORIZED_USER_ID = 123456789
PROXY_URL = "http://127.0.0.1:10808"
TELEGRAM_MAX_LENGTH = 4000

AGENT_HOME = os.path.expanduser("~/agent")
AGENT_WORK_DIR = AGENT_HOME
AGENT_MEDIA_DIR = os.path.join(AGENT_HOME, "media-file")
AGENT_LOCK_FILE = os.path.join(AGENT_HOME, ".lock")
AGENT_SCHEDULE_FILE = os.path.join(AGENT_HOME, "schedule.json")
SESSION_MARKER = os.path.join(AGENT_WORK_DIR, ".session_started")
OPENCODE_TIMEOUT = 300

MODELS = {
    "free": "opencode/minimax-m2.5-free",
    "flash": "deepseek/deepseek-v4-flash",
    "pro": "deepseek/deepseek-v4-pro",
}
CURRENT_MODEL_KEY = "free"

COMMANDS = {
    "clear": "clear session",
    "free": "use free model",
    "flash": "use flash model",
    "pro": "use pro model",
}

os.makedirs(AGENT_WORK_DIR, exist_ok=True)
os.makedirs(AGENT_MEDIA_DIR, exist_ok=True)

logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)

# ==============================================================================
# STATE
# ==============================================================================

current_model_key = CURRENT_MODEL_KEY


# ==============================================================================
# UTILITIES
# ==============================================================================

def get_model() -> str:
    return MODELS[current_model_key]


def clear_session():
    if os.path.exists(SESSION_MARKER):
        os.remove(SESSION_MARKER)


def has_lock() -> bool:
    return os.path.exists(AGENT_LOCK_FILE)


def acquire_lock():
    open(AGENT_LOCK_FILE, "w").close()


def release_lock():
    if os.path.exists(AGENT_LOCK_FILE):
        os.remove(AGENT_LOCK_FILE)


def cleanup_media():
    for item in os.listdir(AGENT_MEDIA_DIR):
        item_path = os.path.join(AGENT_MEDIA_DIR, item)
        if os.path.isdir(item_path):
            shutil.rmtree(item_path)
        else:
            os.remove(item_path)


# ==============================================================================
# SCHEDULER
# ==============================================================================

def load_tasks() -> list:
    if not os.path.exists(AGENT_SCHEDULE_FILE):
        return []
    try:
        with open(AGENT_SCHEDULE_FILE, "r") as f:
            content = f.read().strip()
            if content:
                return json.loads(content)
            return []
    except Exception as e:
        logging.error(f"[schedule] load failed: {e}")
        return []


def save_tasks(tasks: list):
    tmp_file = AGENT_SCHEDULE_FILE + ".tmp"
    with open(tmp_file, "w") as f:
        json.dump(tasks, f, indent=2)
    os.replace(tmp_file, AGENT_SCHEDULE_FILE)


def is_task_due(task: dict) -> bool:
    if task.get("done"):
        return False
    run_at = datetime.strptime(task["run_at"], "%Y-%m-%d %H:%M")
    return datetime.now() >= run_at


def compute_next_run(task: dict) -> str | None:
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


# ==============================================================================
# TELEGRAM OUTPUT
# ==============================================================================

async def send_text(text: str, update: Update = None, app=None):
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
            await app.bot.send_message(chat_id=AUTHORIZED_USER_ID, text=chunk)
        await asyncio.sleep(0.6)


async def send_files(update: Update = None, app=None):
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
                    await app.bot.send_document(chat_id=AUTHORIZED_USER_ID, document=f)
        except Exception as e:
            await send_text(f"❌ Failed to send file: {path}", update, app)


# ==============================================================================
# AGENT EXECUTOR
# ==============================================================================

async def run_agent(prompt: str) -> str:
    use_continue = os.path.exists(SESSION_MARKER)
    model = get_model()

    cmd = ["opencode", "run", "--model", model, "--dangerously-skip-permissions"]
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


async def execute_task(prompt: str, update: Update = None, app=None, task_info: str = None):
    if app and has_lock():
        await send_text("⚠️ Another task running, wait in queue", update, app)
        return

    acquire_lock()

    try:
        if task_info:
            await send_text(f"🚀 {task_info}", update, app)
        await send_text("🧠 Thinking...", update, app)
        response = await run_agent(prompt)
        await send_text(response, update, app)
        await send_files(update, app)
        await send_text("✅ Agent finished.", update, app)
    finally:
        cleanup_media()
        release_lock()


# ==============================================================================
# SCHEDULER LOOP
# ==============================================================================

async def run_scheduled_tasks(app):
    tasks = load_tasks()
    updated = []

    for task in tasks:
        if is_task_due(task):
            await execute_task(task["prompt"], None, app, f"Running: {task['prompt']}")
            next_run = compute_next_run(task)
            if next_run:
                task["run_at"] = next_run
                task["done"] = False
            else:
                task["done"] = True
        updated.append(task)

    if updated:
        save_tasks(updated)


async def scheduler_loop(app):
    while True:
        try:
            await run_scheduled_tasks(app)
        except Exception as e:
            await send_text(f"❌ Scheduler error: {e}", None, app)
        await asyncio.sleep(30)


# ==============================================================================
# TELEGRAM HANDLERS
# ==============================================================================

async def handle_file(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if update.message.from_user.id != AUTHORIZED_USER_ID:
        await send_text("❌ Unauthorized.", update)
        return

    message = update.message

    file_obj = None
    file_name = None

    if message.document:
        file_obj = message.document
        file_name = file_obj.file_name or f"document_{datetime.now().timestamp()}"
    elif message.photo:
        file_obj = message.photo[-1]
        file_name = f"photo_{datetime.now().timestamp()}"
    elif message.video:
        file_obj = message.video
        file_name = file_obj.file_name or f"video_{datetime.now().timestamp()}"
    elif message.audio:
        file_obj = message.audio
        file_name = file_obj.file_name or f"audio_{datetime.now().timestamp()}"

    if not file_obj or not file_name:
        await send_text("⚠️ Unsupported file type.", update)
        return

    try:
        file = await file_obj.get_file()
        dest_path = os.path.join(AGENT_MEDIA_DIR, file_name)
        await file.download_to_drive(dest_path)
        await send_text(f"✅ File saved: {file_name}", update)
    except Exception as e:
        await send_text(f"❌ Failed to download file: {e}", update)


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if update.message.from_user.id != AUTHORIZED_USER_ID:
        await send_text("❌ Unauthorized.", update)
        return

    prompt = update.message.text
    text = prompt.lower().strip()

    if text == "clear":
        clear_session()
        await send_text("✅ Session cleared. Next message starts fresh.", update)
        return

    global current_model_key

    if text == "flash":
        current_model_key = "flash"
        await send_text(f"✅ Switched to {MODELS['flash']}", update)
        return

    if text == "pro":
        current_model_key = "pro"
        await send_text(f"✅ Switched to {MODELS['pro']}", update)
        return

    if text == "free":
        current_model_key = "free"
        await send_text(f"✅ Switched to {MODELS['free']}", update)
        return

    await execute_task(prompt, update, None)


# ==============================================================================
# MAIN
# ==============================================================================

def main():
    app = (
        ApplicationBuilder()
        .token(TOKEN)
        .proxy(PROXY_URL)
        .connect_timeout(30)
        .read_timeout(30)
        .get_updates_read_timeout(60)
        .build()
    )

    app.add_handler(MessageHandler(filters.TEXT & (~filters.COMMAND), handle_message))
    app.add_handler(MessageHandler(
        filters.Document.ALL | filters.PHOTO | filters.VIDEO | filters.AUDIO,
        handle_file
    ))

    async def error_handler(update, context):
        logging.error(f"Exception: {context.error}")
    app.add_error_handler(error_handler)

    async def _post_init(app):
        asyncio.create_task(scheduler_loop(app))
        await send_text("🚀 Agent ready (opencode backend).", None, app)
    app.post_init = _post_init

    app.run_polling(drop_pending_updates=True)


if __name__ == "__main__":
    main()
