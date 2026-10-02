"""
Serves engine executions over TCP, so the fuzzer can run on a different machine (or
VM) to the engine. This is used on Apple Silicon: d8 is a Linux x64 binary that has
to run in a Docker container under Rosetta, where PyTorch is very slow, so the
fuzzer runs natively (with the GPU) and sends programs to this server.

Each connection gets its own REPRL engine process. Messages are a 4 byte length
followed by a JSON header, then a 4 byte length followed by a binary payload:
  request:  {"lib": path or null}, script bytes
  response: {"error", "crashed", "out", "err", "num_edges"}, coverage bitmap bytes
"""

from __future__ import annotations

import json
import socket
import struct
from typing import Any, Optional


# Enough for any error report, while keeping messages small
MAX_OUTPUT = 64 * 1024


def send_message(sock: socket.socket, header: dict[str, Any], payload: bytes):
    header_bytes = json.dumps(header).encode("utf-8")
    sock.sendall(
        struct.pack("<I", len(header_bytes))
        + header_bytes
        + struct.pack("<I", len(payload))
        + payload
    )


def _recv_exact(sock: socket.socket, n: int) -> bytes:
    data = bytearray()
    while len(data) < n:
        chunk = sock.recv(n - len(data))
        if not chunk:
            raise ConnectionError("connection closed")
        data += chunk
    return bytes(data)


def recv_message(sock: socket.socket) -> tuple[dict[str, Any], bytes]:
    (header_len,) = struct.unpack("<I", _recv_exact(sock, 4))
    header = json.loads(_recv_exact(sock, header_len))
    (payload_len,) = struct.unpack("<I", _recv_exact(sock, 4))
    return header, _recv_exact(sock, payload_len)


def serve(
    host: str,
    port: int,
    name: str,
    executable: str,
    timeout: float,
    extra_args: list[str],
):
    import logging
    from pathlib import Path
    import socketserver

    from utils.js_engine import Engine

    class Handler(socketserver.BaseRequestHandler):
        def handle(self):
            sock: socket.socket = self.request
            sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            engine = Engine.get_engine(name, Path(executable), "reprl", timeout, extra_args)
            logging.info(f"Client connected from {self.client_address}")
            try:
                while True:
                    try:
                        header, script = recv_message(sock)
                    except ConnectionError:
                        return

                    lib: Optional[str] = header.get("lib")
                    code = script.decode("utf-8", errors="replace")
                    data = engine.execute_text(code, Path(lib) if lib else None)

                    if data is None:
                        send_message(sock, {"none": True}, b"")
                        continue

                    send_message(
                        sock,
                        {
                            "error": data.error.value,
                            "crashed": data.crashed,
                            "out": data.out[-MAX_OUTPUT:],
                            "err": data.err[-MAX_OUTPUT:],
                            "num_edges": data.coverage.num_edges,
                        },
                        data.coverage.edges.tobytes(),
                    )
            finally:
                engine.close()
                logging.info(f"Client {self.client_address} disconnected")

    class Server(socketserver.ThreadingTCPServer):
        allow_reuse_address = True
        daemon_threads = True

    with Server((host, port), Handler) as server:
        logging.info(f"Serving {executable} on {host}:{port}")
        server.serve_forever()


if __name__ == "__main__":
    import argparse
    import logging
    import sys

    from utils.js_engine import TIMEOUT

    parser = argparse.ArgumentParser(description="Serve engine executions over TCP")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=9000)
    parser.add_argument("--engine-name", default="v8")
    parser.add_argument(
        "--engine-executable", default="engines/v8/v8/out/fuzzbuild/d8"
    )
    parser.add_argument("--engine-arg", action="append", default=[])
    parser.add_argument("--timeout", type=float, default=TIMEOUT)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, stream=sys.stdout)
    serve(
        args.host,
        args.port,
        args.engine_name,
        args.engine_executable,
        args.timeout,
        args.engine_arg,
    )
