"""Bounded token reader for large historical mask JSON; no scientific changes."""
import json
import re


def json_tokens(path, chunk_size=1024*1024):
    decoder = json.JSONDecoder()
    whitespace = re.compile(r"\s*")
    with open(path, encoding="utf-8") as stream:
        buffer, position, eof = "", 0, False
        while True:
            if position >= len(buffer):
                if eof:
                    return
                buffer, position = stream.read(chunk_size), 0
                eof = len(buffer) < chunk_size
                if not buffer:
                    return
            position = whitespace.match(buffer, position).end()
            if position == len(buffer):
                continue
            first = buffer[position]
            if first in "{}[]:,":
                position += 1
                yield first
                continue
            try:
                value, end = decoder.raw_decode(buffer, position)
                # A number at a buffer boundary may continue in the next chunk.
                if end == len(buffer) and not eof:
                    raise json.JSONDecodeError("incomplete boundary token", buffer, position)
            except json.JSONDecodeError:
                if eof:
                    raise
                tail = stream.read(chunk_size)
                buffer, position = buffer[position:] + tail, 0
                eof = len(tail) < chunk_size
                if len(buffer) > max(2 * chunk_size, 4096):
                    raise ValueError("mask JSON token exceeds bounded token size")
                continue
            position = end
            yield ("string" if isinstance(value, str) else "atom", value)
