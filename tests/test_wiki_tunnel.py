# Здесь проверяем прокси туннеля Википедии: разбор CONNECT, список разрешённых хостов, перекладку байтов.
import asyncio
import importlib.util
import re
from pathlib import Path

spec = importlib.util.spec_from_file_location(
    "wiki_tunnel_proxy", Path(__file__).resolve().parents[1] / "scripts" / "wiki_tunnel" / "proxy.py")
proxy = importlib.util.module_from_spec(spec)
spec.loader.exec_module(proxy)


def test_parse_connect_and_allow_list():
    assert proxy.parse_connect(b"CONNECT ru.wikipedia.org:443 HTTP/1.1\r\n") == ("ru.wikipedia.org", 443)
    assert proxy.parse_connect(b"GET http://example.com/ HTTP/1.1\r\n") is None
    pattern = re.compile(proxy.ALLOW)
    assert proxy.allowed("ru.wikipedia.org", 443, pattern) and proxy.allowed("en.wikipedia.org", 443, pattern)
    assert not proxy.allowed("ru.wikipedia.org", 80, pattern)
    assert not proxy.allowed("evil.ru.wikipedia.org.example.com", 443, pattern)
    assert not proxy.allowed("api.deepseek.com", 443, pattern) and not proxy.allowed("de.wikipedia.org", 443, pattern)


def test_tunnel_relays_bytes_and_refuses_other_hosts():
    async def scenario():
        async def echo(reader, writer):  # «Википедия»: отвечает тем же
            data = await reader.read(100)
            writer.write(b"echo:" + data)
            await writer.drain()
            writer.close()

        upstream = await asyncio.start_server(echo, "127.0.0.1", 0)
        up_port = upstream.sockets[0].getsockname()[1]
        server = await proxy.serve(0, allow=r"127\.0\.0\.1", ports=(up_port,))
        port = server.sockets[0].getsockname()[1]

        reader, writer = await asyncio.open_connection("127.0.0.1", port)
        writer.write(f"CONNECT 127.0.0.1:{up_port} HTTP/1.1\r\nHost: x\r\n\r\n".encode())
        await writer.drain()
        assert (await reader.readline()).startswith(b"HTTP/1.1 200")
        await reader.readline()
        writer.write(b"hello")
        await writer.drain()
        assert await reader.read(100) == b"echo:hello"
        writer.close()

        reader, writer = await asyncio.open_connection("127.0.0.1", port)
        writer.write(b"CONNECT example.com:443 HTTP/1.1\r\n\r\n")
        await writer.drain()
        assert (await reader.readline()).startswith(b"HTTP/1.1 403")
        writer.close()
        server.close()
        upstream.close()

    asyncio.run(scenario())
