"""Minimal deterministic reader for the pinned TFDS Speech Commands shards.

Only the protobuf wire types used by ``tf.train.Example`` are implemented.
This keeps dataset discovery independent of TensorFlow while the actual MFCC
preprocessing remains the pinned TensorFlow signal pipeline.
"""

from __future__ import annotations

from itertools import islice
import struct
from pathlib import Path
from typing import Iterable, Iterator

import numpy as np


def _varint(payload: bytes, position: int) -> tuple[int, int]:
    shift = value = 0
    while True:
        byte = payload[position]
        position += 1
        value |= (byte & 0x7F) << shift
        if not byte & 0x80:
            return value, position
        shift += 7


def _fields(payload: bytes, position: int = 0, end: int | None = None):
    end = len(payload) if end is None else end
    while position < end:
        key, position = _varint(payload, position)
        field, wire = key >> 3, key & 7
        if wire == 0:
            value, position = _varint(payload, position)
        elif wire == 2:
            length, position = _varint(payload, position)
            value, position = (position, position + length), position + length
        elif wire == 5:
            value, position = (position, position + 4), position + 4
        elif wire == 1:
            value, position = (position, position + 8), position + 8
        else:
            raise ValueError(f"unsupported protobuf wire type {wire}")
        yield field, wire, value


def _feature(payload: bytes, span: tuple[int, int]):
    for field, _, value in _fields(payload, *span):
        start, stop = value
        if field == 1:
            return [payload[x:y] for item, _, (x, y) in _fields(payload, start, stop) if item == 1]
        if field == 2:
            body = payload[start:stop]
            return np.frombuffer(body[2:] if body and body[0] == 0x0A else body, dtype="<f4").copy()
        if field == 3:
            result = []
            for item, wire, encoded in _fields(payload, start, stop):
                if item != 1:
                    continue
                if wire == 2:
                    position, end = encoded
                    while position < end:
                        number, position = _varint(payload, position)
                        result.append(number - (1 << 64) if number >= 1 << 63 else number)
                else:
                    result.append(encoded - (1 << 64) if encoded >= 1 << 63 else encoded)
            return np.asarray(result, dtype=np.int64)
    return None


def iter_examples(path: str | Path, only: Iterable[str] | None = None) -> Iterator[dict]:
    wanted = None if only is None else set(only)
    with Path(path).open("rb") as stream:
        while True:
            header = stream.read(12)
            if not header:
                return
            if len(header) != 12:
                raise ValueError(f"{path}: truncated TFRecord header")
            (length,) = struct.unpack("<Q", header[:8])
            record = stream.read(length)
            if len(record) != length or len(stream.read(4)) != 4:
                raise ValueError(f"{path}: truncated TFRecord payload")
            example: dict[str, object] = {}
            for field, _, features in _fields(record):
                if field != 1:
                    continue
                for item, _, entry in _fields(record, *features):
                    if item != 1:
                        continue
                    name = None
                    value = None
                    for attribute, _, span in _fields(record, *entry):
                        if attribute == 1:
                            name = record[span[0] : span[1]].decode()
                        elif attribute == 2 and (wanted is None or name in wanted):
                            value = _feature(record, span)
                    if name is not None and (wanted is None or name in wanted):
                        example[name] = value
            yield example


def interleaved_examples(
    shards: Iterable[str | Path], *, block_size: int = 16
) -> Iterator[dict]:
    """Yield TFDS ``shuffle_files=False`` order across sharded records."""

    iterators = [iter(iter_examples(path, only=("audio", "label"))) for path in shards]
    active = list(iterators)
    while active:
        following = []
        for iterator in active:
            block = tuple(islice(iterator, block_size))
            yield from block
            if len(block) == block_size:
                following.append(iterator)
        active = following
