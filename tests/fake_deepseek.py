"""A local server that answers like DeepSeek's chat-completions API, for offline tests (no network, no key).

What it mimics, from https://api-docs.deepseek.com (read 2026-11-28): ``POST /chat/completions`` without a ``/v1``
prefix; a 401 without a bearer key; a 400 for ``response_format`` ``json_schema`` and for JSON mode when no message
says "json"; the alias ``deepseek-chat`` answering as ``deepseek-flash``; keep-alive blank lines before the JSON
body; and a ``usage`` block with ``prompt_cache_hit_tokens`` / ``prompt_cache_miss_tokens`` (and the same numbers
under ``prompt_tokens_details``). A prompt's cached part is every leading message the server has seen before in
that position, so repeating a system prompt hits the cache as DeepSeek's prefix cache does. Tokens are words.

``answer`` is a function of the request body that returns the reply content (or a ``(content, finish_reason)``
pair); ``fail`` queues ``(status, headers)`` errors served before the next answers.
"""
from __future__ import annotations

import json
import threading
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

MODELS = {"deepseek-flash": "deepseek-flash", "deepseek-v4-pro": "deepseek-v4-pro", "deepseek-chat": "deepseek-flash"}


def words(messages) -> int:
    return sum(len(str(m.get("content") or "").split()) for m in messages)


class FakeDeepSeek:
    def __init__(self, answer, key: str):
        self.answer, self.key = answer, key
        self.models = dict(MODELS)  # requested name -> the name the server answers under
        self.failures: deque = deque()
        self.requests: list[dict] = []
        self.seen_prefixes: set[str] = set()
        fake = self

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                fake.requests.append({"path": self.path, "headers": dict(self.headers), "body": body})
                status, headers, payload = fake.respond(self.path, dict(self.headers), body)
                data = payload if isinstance(payload, bytes) else json.dumps(payload).encode()
                self.send_response(status)
                for name, value in {"Content-Type": "application/json", **headers}.items():
                    self.send_header(name, value)
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def log_message(self, *args):
                pass

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.server.daemon_threads = True
        self.base_url = f"http://127.0.0.1:{self.server.server_address[1]}"
        threading.Thread(target=self.server.serve_forever, kwargs={"poll_interval": 0.01}, daemon=True).start()

    def close(self):
        self.server.shutdown()
        self.server.server_close()

    def fail(self, status: int, headers: dict | None = None, times: int = 1):
        for _ in range(times):
            self.failures.append((status, headers or {}))

    @staticmethod
    def error(status: int, message: str, kind: str = "invalid_request_error"):
        return {"error": {"message": message, "type": kind, "param": None, "code": None}}

    def respond(self, path: str, headers: dict, body: dict):
        if path != "/chat/completions":
            return 404, {}, self.error(404, f"no route {path}")
        if headers.get("Authorization") != f"Bearer {self.key}":
            return 401, {}, self.error(401, "Authentication Fails, Your api key is invalid", "authentication_error")
        if self.failures:
            status, extra = self.failures.popleft()
            return status, extra, self.error(status, f"scripted {status}")
        if body.get("model") not in self.models:
            return 400, {}, self.error(400, f"Model Not Exist: {body.get('model')}")
        fmt = (body.get("response_format") or {}).get("type", "text")
        if fmt not in ("text", "json_object"):
            return 400, {}, self.error(400, "This response_format type is unavailable now")
        if fmt == "json_object" and "json" not in json.dumps(body["messages"]).lower():
            return 400, {}, self.error(400, "Prompt must contain the word 'json' in some form to use "
                                            "'response_format' of type 'json_object'.")
        answer = self.answer(body)
        content, finish = answer if isinstance(answer, tuple) else (answer, "stop")
        messages = body["messages"]
        hit = 0
        for n in range(1, len(messages) + 1):
            prefix = json.dumps(messages[:n], sort_keys=True)
            if prefix in self.seen_prefixes:
                hit = words(messages[:n])
            self.seen_prefixes.add(prefix)
        prompt = words(messages)
        completion = len(str(content or "").split())
        payload = {"id": f"ds-{len(self.requests)}", "object": "chat.completion", "created": 1764288000,
                   "model": self.models[body["model"]], "system_fingerprint": "fp_fake",
                   "choices": [{"index": 0, "finish_reason": finish, "logprobs": None,
                                "message": {"role": "assistant", "content": content}}],
                   "usage": {"prompt_tokens": prompt, "completion_tokens": completion,
                             "total_tokens": prompt + completion, "prompt_cache_hit_tokens": hit,
                             "prompt_cache_miss_tokens": prompt - hit,
                             "prompt_tokens_details": {"cached_tokens": hit},
                             "completion_tokens_details": {"reasoning_tokens": 0}}}
        return 200, {}, b"\n\n" + json.dumps(payload).encode()  # keep-alive blank lines come first
