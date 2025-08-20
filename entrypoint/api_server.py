import argparse
import asyncio
import logging
import resource
import signal
import ssl
import time
from argparse import Namespace
from typing import Optional, Any

import psutil
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import Response

from ollama import AsyncClient

app = FastAPI()
client = None
model = ""
start_time = 0.0

TIMEOUT_KEEP_ALIVE = 5.0


@app.post("/generate_benchmark")
async def generate_text(prompt, request_id):
    """
    Generate text using the specified model and prompt.
    """
    arrival_time = time.time()
    assert client is not None, "Client is not initialized"
    try:
        client = AsyncClient(host=client.host, port=client.port)
        results_generator = await client.generate(model, prompt, stream=True)
        final_output = None
        per_token_latency = []
        start = time.time()
        try:
            async for request_output in results_generator:
                now = time.time()
                per_token_latency.append([now, (now - start) * 1000])
                start = now
                final_output = request_output
        except asyncio.CancelledError:
            print("Cancelled request for request_id: {}".format(request_id))
            return Response(status_code=499)
        response = {
            "request_id": request_id,
            "model": model,
            "prompt": prompt,
            "output": final_output,
            "arrival_time": arrival_time,
            "ttft": per_token_latency[0][1] * 1000 if per_token_latency else 0,
            "e2e_latency": (time.time() - arrival_time) * 1000
        }
        return Response(content=response)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


def build_app(args: Namespace, ollama_client: AsyncClient) -> FastAPI:
    global app, start_time, client, model
    app.root_path = args.root_path
    start_time = time.time()
    client = ollama_client
    model = args.model
    return app


async def serve_http(**uvicorn_kwargs: Any):
    for route in app.routes:
        methods = getattr(route, "methods", None)
        path = getattr(route, "path", None)
        if methods is None or path is None:
            continue

    config = uvicorn.Config(app, **uvicorn_kwargs)
    server = uvicorn.Server(config)
    loop = asyncio.get_running_loop()
    server_task = loop.create_task(server.serve())

    def signal_handler() -> None:
        # prevents the uvicorn signal handler to exit early
        server_task.cancel()

    async def dummy_shutdown() -> None:
        pass

    def find_process_using_port(http_port) -> Optional[psutil.Process]:
        for conn in psutil.net_connections():
            if conn.laddr.port == http_port:
                try:
                    return psutil.Process(conn.pid)
                except psutil.NoSuchProcess:
                    return None
        return None

    loop.add_signal_handler(signal.SIGINT, signal_handler)
    loop.add_signal_handler(signal.SIGTERM, signal_handler)
    try:
        await server_task
        return dummy_shutdown()
    except asyncio.CancelledError:
        port = uvicorn_kwargs["port"]
        process = find_process_using_port(port)
        if process is not None:
            logging.debug(
                "port %s is used by process %s launched with command:\n%s",
                port, process, " ".join(process.cmdline()))
        logging.info("Shutting down FastAPI HTTP server.")
        return server.shutdown()


async def run_server(args: Namespace,
                     **uvicorn_kwargs: Any) -> None:
    global start_time, app
    start_time = time.time()
    if args.ollama_host is None:
        ollama_client = AsyncClient()
    else:
        ollama_client = AsyncClient(host=args.host, port=args.port)
    app = build_app(args, ollama_client)
    shutdown_task = await serve_http(
        host=args.host,
        port=args.port,
        timeout_keep_alive=TIMEOUT_KEEP_ALIVE,
        ssl_keyfile=args.ssl_keyfile,
        ssl_certfile=args.ssl_certfile,
        ssl_ca_certs=args.ssl_ca_certs,
        ssl_cert_reqs=args.ssl_cert_reqs,
        workers=args.workers,
        **uvicorn_kwargs,
    )

    await shutdown_task


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default=None)
    parser.add_argument("--port", type=int, default=8100)
    parser.add_argument("--ssl-keyfile", type=str, default=None)
    parser.add_argument("--ssl-certfile", type=str, default=None)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--ssl-ca-certs",
                        type=str,
                        default=None,
                        help="The CA certificates file")
    parser.add_argument(
        "--ssl-cert-reqs",
        type=int,
        default=int(ssl.CERT_NONE),
        help="Whether client certificate is required (see stdlib ssl module's)"
    )
    parser.add_argument(
        "--root-path",
        type=str,
        default=None,
        help="FastAPI root_path when app is behind a path based routing proxy")
    parser.add_argument("--model", type=str, default="gpt-oss:20b")
    parser.add_argument("--ollama-host", type=str, default=None, help="host address running ollama server")
    parser.add_argument("--ollama-port", type=int, default=11434, help="port running ollama server")
    logging.log(logging.INFO, "Starting server with args: %s", str(parser.parse_args()))
    parse_args = parser.parse_args()
    resource.setrlimit(resource.RLIMIT_NOFILE, (65536, 65536))
    asyncio.run(run_server(parse_args))
