# Здесь запускаем на моём компьютере прокси для туннеля Википедии: сервер ходит в Википедию через него.
"""Прокси туннеля Википедии (решение 10.11.2026, по согласованию с администратором сервера).

Работает на моём компьютере и слушает только 127.0.0.1. Сервер достаёт до него через обратный SSH-проброс
(``tunnel.ps1``): на сервере открывается ``127.0.0.1:18080``, ведущий сюда. Прокси понимает только CONNECT
и пропускает только API русской и английской Википедии (порт 443) — всё остальное отклоняет. TLS сквозной:
прокси лишь перекладывает байты, сертификат Википедии проверяет клиент на сервере.

Только стандартная библиотека Python 3.10+::

    python proxy.py --port 18080
"""
from __future__ import annotations

import argparse
import asyncio
import logging
import re
import time

ALLOW = r"(ru|en)\.wikipedia\.org"
PORTS = (443,)

log = logging.getLogger("wiki-proxy")


def parse_connect(line: bytes) -> tuple[str, int] | None:
    """«CONNECT host:port HTTP/1.1» → (host, port); иначе None."""
    parts = line.decode("latin-1").strip().split()
    if len(parts) != 3 or parts[0].upper() != "CONNECT":
        return None
    host, _, port = parts[1].rpartition(":")
    if not host or not port.isdigit():
        return None
    return host.strip("[]").lower(), int(port)


def allowed(host: str, port: int, pattern: re.Pattern, ports: tuple[int, ...] = PORTS) -> bool:
    return port in ports and pattern.fullmatch(host) is not None


async def _pipe(reader: asyncio.StreamReader, writer: asyncio.StreamWriter, counter: list[int]) -> None:
    try:
        while data := await reader.read(65536):
            counter[0] += len(data)
            writer.write(data)
            await writer.drain()
    except (ConnectionError, asyncio.IncompleteReadError):
        pass
    finally:
        try:
            writer.close()
        except Exception:
            pass


async def handle(reader, writer, pattern: re.Pattern, ports: tuple[int, ...]) -> None:
    start = time.monotonic()
    try:
        line = await asyncio.wait_for(reader.readline(), 30)
        while (header := await asyncio.wait_for(reader.readline(), 30)) not in (b"\r\n", b"\n", b""):
            pass  # заголовки запроса не нужны
    except asyncio.TimeoutError:
        writer.close()
        return
    target = parse_connect(line)
    if target is None:
        writer.write(b"HTTP/1.1 405 Method Not Allowed\r\nContent-Length: 0\r\n\r\n")
        await writer.drain()
        writer.close()
        log.warning("отклонено (не CONNECT): %r", line[:100])
        return
    host, port = target
    if not allowed(host, port, pattern, ports):
        writer.write(b"HTTP/1.1 403 Forbidden\r\nContent-Length: 0\r\n\r\n")
        await writer.drain()
        writer.close()
        log.warning("отклонено (не Википедия): %s:%s", host, port)
        return
    try:
        up_reader, up_writer = await asyncio.wait_for(asyncio.open_connection(host, port), 15)
    except (OSError, asyncio.TimeoutError) as error:
        writer.write(b"HTTP/1.1 502 Bad Gateway\r\nContent-Length: 0\r\n\r\n")
        await writer.drain()
        writer.close()
        log.warning("нет связи с %s:%s: %s", host, port, error)
        return
    writer.write(b"HTTP/1.1 200 Connection established\r\n\r\n")
    await writer.drain()
    sent, received = [0], [0]
    await asyncio.gather(_pipe(reader, up_writer, sent), _pipe(up_reader, writer, received))
    log.info("%s:%s — отправлено %d Б, получено %d Б, %.1f с", host, port, sent[0], received[0],
             time.monotonic() - start)


async def serve(port: int, allow: str = ALLOW, ports: tuple[int, ...] = PORTS, host: str = "127.0.0.1"):
    pattern = re.compile(allow)
    return await asyncio.start_server(lambda r, w: handle(r, w, pattern, ports), host, port)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--port", type=int, default=18080)
    parser.add_argument("--allow", default=ALLOW, help="регулярное выражение для разрешённых хостов")
    parser.add_argument("--log", default="wiki_proxy.log")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s",
                        handlers=[logging.StreamHandler(), logging.FileHandler(args.log, encoding="utf-8")])

    async def run():
        server = await serve(args.port, args.allow)
        log.info("прокси Википедии слушает 127.0.0.1:%d, разрешено: %s", args.port, args.allow)
        async with server:
            await server.serve_forever()

    asyncio.run(run())


if __name__ == "__main__":
    main()
