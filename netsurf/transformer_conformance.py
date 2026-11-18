"""Prespecified encoded-word conformance cases for real transformer storage.

Exceptional-value cases are deliberately forced engineering schedules, never
misrepresented as draws from a Bernoulli field.  Their full event lists and
before/after bytes are suitable for a bounded backend conformance receipt.
"""

from __future__ import annotations

from .identities import BitIdentity
from .storage_bit_codec import StorageBitCodec


def conformance_cases(parameter: str, flat_index: int, raw_word: bytes,
                      codec: StorageBitCodec) -> dict[str, tuple[BitIdentity, ...]]:
    if flat_index < 0 or len(raw_word) != codec.byte_width:
        raise ValueError("one valid stored word is required")
    fields = codec.fields(raw_word)
    if fields["category"] not in {"normal", "subnormal", "zero"}:
        raise ValueError("conformance source word must be finite")
    width = codec.width
    mantissa_width = 23 if codec.precision == "FP32" else 7
    original = int.from_bytes(raw_word, "little")
    sign_mask = 1 << (width - 1)
    exponent_mask = 0xff << mantissa_width
    mantissa_mask = (1 << mantissa_width) - 1
    infinity = (original & sign_mask) | exponent_mask
    quiet_nan = infinity | (1 << (mantissa_width - 1))

    def bits_for_difference(target: int) -> tuple[BitIdentity, ...]:
        difference = original ^ target
        return tuple(BitIdentity(parameter, flat_index, bit)
                     for bit in range(width) if difference & (1 << bit))

    cases = {
        "sign": (BitIdentity(parameter, flat_index, width - 1),),
        "exponent": (BitIdentity(parameter, flat_index, mantissa_width),),
        "significand": (BitIdentity(parameter, flat_index, mantissa_width - 1),),
        "infinity": bits_for_difference(infinity),
        "nan": bits_for_difference(quiet_nan),
    }
    if not cases["infinity"] or not cases["nan"]:
        raise RuntimeError("exceptional conformance schedule is empty")
    for name, events in cases.items():
        changed = raw_word
        for event in events:
            changed = codec.xor(changed, 0, event.bit_index)
        observed = codec.fields(changed)["category"]
        if name in {"infinity", "nan"} and observed != name:
            raise RuntimeError(f"{name} conformance bits encoded {observed}")
        if name in {"sign", "exponent", "significand"} and len(events) != 1:
            raise RuntimeError(f"{name} conformance is not a single-bit event")
    return cases
