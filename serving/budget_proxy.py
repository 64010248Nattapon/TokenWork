#!/usr/bin/env python
"""
Tiny proxy in front of vLLM that adds a default thinking budget to every request.

vLLM only accepts `thinking_token_budget` per request, so this proxy injects it
(unless the client already set one) and streams everything else through unchanged.

  grader -> :8000 (this proxy) -> :8001 (vllm serve)

Usage:
  python budget_proxy.py                      # budget 2048, listen 8000, vLLM on 8001
  python budget_proxy.py --budget 3072
vLLM must be started with --reasoning-parser qwen3 and --reasoning-config (see serve.sh).
"""
import argparse
import json

import aiohttp
from aiohttp import web

ap = argparse.ArgumentParser()
ap.add_argument("--listen", type=int, default=8000)
ap.add_argument("--backend", default="http://127.0.0.1:8001")
ap.add_argument("--budget", type=int, default=2048)
args = ap.parse_args()

HOP = {"content-length", "transfer-encoding", "connection", "keep-alive", "host", "content-encoding"}
INJECT_PATHS = ("/v1/chat/completions", "/v1/completions")


async def handle(request: web.Request) -> web.StreamResponse:
    body = await request.read()
    if request.method == "POST" and request.path in INJECT_PATHS and body:
        try:
            data = json.loads(body)
            data.setdefault("thinking_token_budget", args.budget)
            body = json.dumps(data).encode()
        except ValueError:
            pass  # not JSON: forward as-is

    headers = {k: v for k, v in request.headers.items() if k.lower() not in HOP}
    session: aiohttp.ClientSession = request.app["session"]
    async with session.request(request.method, args.backend + request.path_qs,
                               data=body or None, headers=headers) as up:
        resp = web.StreamResponse(status=up.status, headers={
            k: v for k, v in up.headers.items() if k.lower() not in HOP})
        await resp.prepare(request)
        async for chunk in up.content.iter_any():  # pass tokens through immediately
            await resp.write(chunk)
        await resp.write_eof()
        return resp


async def start(app):
    app["session"] = aiohttp.ClientSession(
        timeout=aiohttp.ClientTimeout(total=None),
        connector=aiohttp.TCPConnector(limit=0))


async def stop(app):
    await app["session"].close()


app = web.Application(client_max_size=256 * 1024 * 1024)
app.router.add_route("*", "/{tail:.*}", handle)
app.on_startup.append(start)
app.on_cleanup.append(stop)
print(f"proxy :{args.listen} -> {args.backend}  (thinking_token_budget={args.budget})")
web.run_app(app, host="0.0.0.0", port=args.listen, access_log=None)
