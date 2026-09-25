#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Vibe Agent — бесплатный ИИ-агент для вайбкодинга.

Пишет код за тебя: ты описываешь задачу на человеческом языке,
агент сам читает/создаёт/редактирует файлы и запускает команды.

Бесплатные бэкенды (выбираются автоматически или через --provider):
  * Ollama        — локальные модели, полностью бесплатно, без ключей (http://localhost:11434)
  * OpenRouter    — бесплатные модели вида deepseek/deepseek-chat-v3-...:free (нужен бесплатный ключ)
  * OpenAI-совместимый API — любой совместимый сервер (укажите --base-url и OPENAI_API_KEY)

Использование:
  python3 vibe_agent.py "сделай скрипт, который переименовывает все .jpg в папке по дате"
  python3 vibe_agent.py            # интерактивный режим (REPL)
  python3 vibe_agent.py --provider ollama --model qwen2.5-coder:7b "напиши игру змейка на python"

Переменные окружения:
  VIBE_PROVIDER   ollama | openrouter | openai
  VIBE_MODEL      имя модели
  OPENROUTER_API_KEY / OPENAI_API_KEY   ключи
  OLLAMA_HOST     адрес сервера Ollama (по умолчанию http://localhost:11434)
"""

import argparse
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path

WORKSPACE = Path.cwd().resolve()
MAX_STEPS = int(os.environ.get("VIBE_MAX_STEPS", "30"))
COMMAND_TIMEOUT = int(os.environ.get("VIBE_CMD_TIMEOUT", "60"))

SYSTEM_PROMPT = """\
Ты — Vibe Agent, автономный ИИ-инженер для «вайбкодинга»: пользователь
описывает задачу на естественном языке, а ты реализуешь её — пишешь код,
создаёшь и редактируешь файлы, запускаешь команды и проверяешь результат.

Правила:
1. Работай в текущей директории. Все пути относительные.
2. Действуй через вызовы инструментов. После каждого шага изучай результат
   и решай, что делать дальше. Не останавливайся, пока задача не выполнена.
3. Пиши рабочий, законченный код без заглушек. После записи файла запускай его
   (run_command), чтобы убедиться, что всё работает, при ошибках — исправляй.
4. Чтобы завершить задачу, вызови инструмент done с кратким отчётом.
5. Отвечай на языке пользователя.
"""

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Прочитать файл и вернуть его содержимое.",
            "parameters": {
                "type": "object",
                "properties": {"path": {"type": "string", "description": "Путь к файлу"}},
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": "Создать или полностью перезаписать файл. Промежуточные каталоги создаются автоматически.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Путь к файлу"},
                    "content": {"type": "string", "description": "Полное содержимое файла"},
                },
                "required": ["path", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "edit_file",
            "description": "Точечная замена фрагмента текста внутри файла.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Путь к файлу"},
                    "old_text": {"type": "string", "description": "Точный фрагмент для замены (должен встречаться ровно один раз)"},
                    "new_text": {"type": "string", "description": "Новый текст"},
                },
                "required": ["path", "old_text", "new_text"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_dir",
            "description": "Показать содержимое директории.",
            "parameters": {
                "type": "object",
                "properties": {"path": {"type": "string", "description": "Путь к директории (по умолчанию — текущая)"}},
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_command",
            "description": "Выполнить shell-команду в рабочей директории и вернуть stdout/stderr/exit code.",
            "parameters": {
                "type": "object",
                "properties": {"command": {"type": "string", "description": "Команда для выполнения"}},
                "required": ["command"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "done",
            "description": "Завершить задачу. Вызывать только когда всё сделано и проверено.",
            "parameters": {
                "type": "object",
                "properties": {"summary": {"type": "string", "description": "Краткий итог: что сделано, какие файлы созданы/изменены"}},
                "required": ["summary"],
            },
        },
    },
]


# ---------------------------------------------------------------- utilities

def safe_path(p: str) -> Path:
    path = (WORKSPACE / p).resolve()
    if not str(path).startswith(str(WORKSPACE)):
        raise ValueError(f"Путь вне рабочей директории: {p}")
    return path


def clip(text: str, limit: int = 8000) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + f"\n...[обрезано, всего {len(text)} символов]"


# ---------------------------------------------------------------- LLM backends

class Backend:
    name = "base"

    def chat(self, messages):
        raise NotImplementedError


class OpenAICompat(Backend):
    """OpenAI-совместимый Chat Completions API (работает и с OpenRouter)."""

    def __init__(self, base_url, model, api_key):
        from openai import OpenAI
        self.client = OpenAI(base_url=base_url, api_key=api_key or "none")
        self.model = model

    def chat(self, messages):
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            tools=TOOLS,
            temperature=0.2,
        )
        msg = resp.choices[0].message
        calls = []
        for tc in (msg.tool_calls or []):
            try:
                args = json.loads(tc.function.arguments or "{}")
            except json.JSONDecodeError:
                args = {}
            calls.append({"id": tc.id, "name": tc.function.name, "arguments": args})
        return {"content": msg.content or "", "tool_calls": calls}


class Ollama(Backend):
    """Локальный Ollama — бесплатно, без ключей."""

    def __init__(self, host, model):
        self.url = host.rstrip("/") + "/api/chat"
        self.model = model

    def chat(self, messages):
        import urllib.request
        payload = json.dumps({
            "model": self.model,
            "messages": messages,
            "tools": [t["function"] | {"type": "function"} for t in TOOLS],
            "stream": False,
            "options": {"temperature": 0.2},
        }).encode()
        req = urllib.request.Request(self.url, data=payload,
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=300) as r:
            data = json.loads(r.read().decode())
        msg = data.get("message", {})
        calls = []
        for tc in (msg.get("tool_calls") or []):
            fn = tc.get("function", {})
            args = fn.get("arguments", {})
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except json.JSONDecodeError:
                    args = {}
            calls.append({"id": tc.get("id") or f"call_{len(calls)}",
                          "name": fn.get("name"), "arguments": args})
        return {"content": msg.get("content") or "", "tool_calls": calls}


def pick_backend(provider, model):
    provider = (provider or os.environ.get("VIBE_PROVIDER") or "").lower()

    candidates = []
    if provider:
        candidates = [provider]
    else:  # автовыбор: сначала локальный Ollama, потом OpenRouter, потом OpenAI
        candidates = ["ollama", "openrouter", "openai"]

    for p in candidates:
        if p == "ollama":
            host = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
            m = model or os.environ.get("VIBE_MODEL") or "qwen2.5-coder:7b"
            try:
                import urllib.request
                urllib.request.urlopen(host.rstrip("/") + "/api/tags", timeout=2)
                print(f"[vibe] бэкенд: Ollama (локально, бесплатно), модель: {m}", file=sys.stderr)
                return Ollama(host, m)
            except Exception:
                if provider == "ollama":
                    sys.exit(f"[vibe] Ollama не найден на {host}. Установите: https://ollama.com и выполните `ollama pull {m}`")
                continue
        elif p == "openrouter":
            key = os.environ.get("OPENROUTER_API_KEY")
            if key:
                m = model or os.environ.get("VIBE_MODEL") or "deepseek/deepseek-chat-v3-0324:free"
                print(f"[vibe] бэкенд: OpenRouter (бесплатная модель), модель: {m}", file=sys.stderr)
                return OpenAICompat("https://openrouter.ai/api/v1", m, key)
            elif provider == "openrouter":
                sys.exit("[vibe] Нужен бесплатный ключ: зарегистрируйтесь на https://openrouter.ai и задайте OPENROUTER_API_KEY")
        elif p == "openai":
            key = os.environ.get("OPENAI_API_KEY")
            if key:
                m = model or os.environ.get("VIBE_MODEL") or "gpt-4o-mini"
                base = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")
                print(f"[vibe] бэкенд: OpenAI-совместимый API ({base}), модель: {m}", file=sys.stderr)
                return OpenAICompat(base, m, key)
            elif provider == "openai":
                sys.exit("[vibe] Не задан OPENAI_API_KEY")

    sys.exit(
        "[vibe] Не найден ни один доступный бэкенд.\n"
        "  Бесплатный вариант 1 (рекомендуется): установите Ollama → https://ollama.com\n"
        "     ollama pull qwen2.5-coder:7b   — и запустите агент снова.\n"
        "  Бесплатный вариант 2: получите бесплатный ключ на https://openrouter.ai\n"
        "     export OPENROUTER_API_KEY=sk-or-...\n"
    )


# ---------------------------------------------------------------- tool exec

def execute(tool, args):
    try:
        if tool == "read_file":
            path = safe_path(args["path"])
            return path.read_text(encoding="utf-8", errors="replace")
        if tool == "write_file":
            path = safe_path(args["path"])
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(args["content"], encoding="utf-8")
            return f"OK: записано {len(args['content'])} символов в {args['path']}"
        if tool == "edit_file":
            path = safe_path(args["path"])
            text = path.read_text(encoding="utf-8")
            old, new = args["old_text"], args["new_text"]
            n = text.count(old)
            if n == 0:
                return "ERROR: old_text не найден в файле. Прочитайте файл и используйте точный фрагмент."
            if n > 1:
                return f"ERROR: old_text встречается {n} раз — добавьте больше контекста."
            path.write_text(text.replace(old, new, 1), encoding="utf-8")
            return "OK: фрагмент заменён"
        if tool == "list_dir":
            path = safe_path(args.get("path") or ".")
            entries = []
            for e in sorted(path.iterdir()):
                entries.append(e.name + ("/" if e.is_dir() else f"  ({e.stat().st_size} B)"))
            return "\n".join(entries) or "(пусто)"
        if tool == "run_command":
            cmd = args["command"]
            dangerous = ("rm -rf /", ":(){", "mkfs", "dd if=")
            if any(d in cmd for d in dangerous):
                return "ERROR: команда отклонена как потенциально опасная"
            r = subprocess.run(cmd, shell=True, cwd=WORKSPACE, capture_output=True,
                               text=True, timeout=COMMAND_TIMEOUT)
            out = (r.stdout or "") + (("\n[stderr]\n" + r.stderr) if r.stderr else "")
            return f"exit={r.returncode}\n{clip(out.strip()) or '(нет вывода)'}"
        return f"ERROR: неизвестный инструмент {tool}"
    except subprocess.TimeoutExpired:
        return f"ERROR: команда превысила таймаут {COMMAND_TIMEOUT}s"
    except Exception as e:
        return f"ERROR: {e}"


# ---------------------------------------------------------------- agent loop

C_RESET, C_DIM, C_CYAN, C_GREEN, C_YELLOW = "\033[0m", "\033[2m", "\033[36m", "\033[32m", "\033[33m"


def render_tool_call(name, args):
    if name == "write_file":
        preview = args.get("content", "").splitlines()
        head = " | ".join(preview[:2])[:80]
        return f"{C_CYAN}✍ write_file{C_RESET} {args.get('path')} {C_DIM}({len(args.get('content',''))} симв., начало: {head}){C_RESET}"
    if name == "edit_file":
        return f"{C_CYAN}✂ edit_file{C_RESET} {args.get('path')}"
    if name == "read_file":
        return f"{C_CYAN}📖 read_file{C_RESET} {args.get('path')}"
    if name == "list_dir":
        return f"{C_CYAN}📂 list_dir{C_RESET} {args.get('path', '.')}"
    if name == "run_command":
        return f"{C_YELLOW}▶ run{C_RESET} {C_DIM}{shlex.quote(args.get('command',''))[:120]}{C_RESET}"
    return f"{C_CYAN}• {name}{C_RESET}"


def run_task(backend, task, messages):
    messages.append({"role": "user", "content": task})
    for step in range(1, MAX_STEPS + 1):
        reply = backend.chat(messages)
        if reply["content"]:
            print(f"{C_GREEN}{reply['content']}{C_RESET}")
        if not reply["tool_calls"]:
            # модель ответила текстом без вызова инструментов — поторопим её
            messages.append({"role": "user",
                             "content": "Если задача ещё не выполнена — продолжай работать через инструменты. Если выполнена — вызови done с итогом."})
            if step > MAX_STEPS - 3:
                break
            continue

        assistant_msg = {"role": "assistant", "content": reply["content"] or None,
                         "tool_calls": [{"id": c["id"], "type": "function",
                                         "function": {"name": c["name"],
                                                      "arguments": json.dumps(c["arguments"], ensure_ascii=False)}}
                                        for c in reply["tool_calls"]]}
        messages.append(assistant_msg)

        for call in reply["tool_calls"]:
            name, args = call["name"], call["arguments"]
            if name == "done":
                print(f"\n{C_GREEN}✔ Готово:{C_RESET} {args.get('summary', '')}")
                return True
            print(render_tool_call(name, args))
            result = execute(name, args)
            print(f"  {C_DIM}{clip(result, 400)}{C_RESET}")
            messages.append({"role": "tool", "tool_call_id": call["id"],
                             "name": name, "content": clip(result)})
        # экономим контекст: старые длинные выводы инструментов урезаем
        _trim_context(messages)
    print(f"{C_YELLOW}⚠ Лимит шагов ({MAX_STEPS}) исчерпан — задача не подтверждена как выполненная.{C_RESET}")
    return False


def _trim_context(messages):
    total = sum(len(json.dumps(m, ensure_ascii=False)) for m in messages)
    i = 2  # не трогаем system и первую постановку задачи
    while total > 60000 and i < len(messages) - 4:
        m = messages[i]
        if m.get("role") == "tool" and len(m.get("content", "")) > 800:
            shrink = m["content"][:300] + "\n[старый вывод урезан]"
            total -= len(m["content"]) - len(shrink)
            m["content"] = shrink
        i += 1


# ---------------------------------------------------------------- main

def main():
    global MAX_STEPS
    ap = argparse.ArgumentParser(description="Vibe Agent — бесплатный ИИ-агент для вайбкодинга")
    ap.add_argument("task", nargs="*", help="Задача своими словами (или пустите в REPL)")
    ap.add_argument("--provider", choices=["ollama", "openrouter", "openai"], help="LLM-бэкенд (по умолчанию автовыбор)")
    ap.add_argument("--model", help="Имя модели")
    ap.add_argument("--max-steps", type=int, default=MAX_STEPS)
    args = ap.parse_args()

    MAX_STEPS = args.max_steps

    backend = pick_backend(args.provider, args.model)
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    if args.task:
        run_task(backend, " ".join(args.task), messages)
        return

    print(f"{C_GREEN}Vibe Agent{C_RESET} {C_DIM}(бесплатный ИИ-агент для вайбкодинга; вход — пустая строка или Ctrl+D для выхода){C_RESET}")
    while True:
        try:
            task = input(f"\n🎸 vibe> {C_RESET}").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not task:
            break
        run_task(backend, task, messages)


if __name__ == "__main__":
    main()
