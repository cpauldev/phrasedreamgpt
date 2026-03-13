from __future__ import annotations

import base64
import hashlib
import math
from collections.abc import Iterable
from dataclasses import dataclass

from .config import Dataset, fail

SOURCE_FILTER_KIND = "bloom"
SOURCE_FILTER_VERSION = 1
DEFAULT_SOURCE_FILTER_FALSE_POSITIVE_RATE = 1e-4
DEFAULT_SOURCE_FILTER_MAX_RETRIES = 40
_FALLBACK_HASH_STEP = 0x9E3779B97F4A7C15


def normalize_source_text(text: str) -> str:
    return text.strip()


def iter_dataset_documents(dataset: Dataset) -> Iterable[str]:
    chars: list[str] = []
    for token in dataset.data.detach().to(device="cpu").tolist():
        if token == dataset.bos_id:
            if chars:
                yield "".join(chars)
                chars = []
            continue
        chars.append(dataset.id_to_char[token])
    if chars:
        yield "".join(chars)


@dataclass(frozen=True)
class BloomSourceFilter:
    bit_count: int
    hash_count: int
    bits: bytes
    false_positive_rate: float
    item_count: int

    def __post_init__(self) -> None:
        expected_bytes = math.ceil(self.bit_count / 8)
        if self.bit_count <= 0:
            fail("Source filter bit count must be greater than 0.")
        if self.hash_count <= 0:
            fail("Source filter hash count must be greater than 0.")
        if len(self.bits) != expected_bytes:
            fail(
                "Source filter bit payload has the wrong size.",
                f"Expected {expected_bytes} bytes but received {len(self.bits)}.",
            )
        if not 0 < self.false_positive_rate < 1:
            fail(
                "Source filter false-positive rate must be between 0 and 1.",
                "Use a value such as 1e-4.",
            )
        if self.item_count < 0:
            fail("Source filter item count cannot be negative.")

    def matches(self, text: str) -> bool:
        normalized = normalize_source_text(text)
        if not normalized:
            return False
        bit_view = memoryview(self.bits)
        for index in iter_hash_indices(normalized, self.bit_count, self.hash_count):
            byte_index, bit_offset = divmod(index, 8)
            if not (bit_view[byte_index] & (1 << bit_offset)):
                return False
        return True

    def to_artifact_dict(self) -> dict[str, object]:
        return {
            "kind": SOURCE_FILTER_KIND,
            "version": SOURCE_FILTER_VERSION,
            "bit_count": self.bit_count,
            "hash_count": self.hash_count,
            "bits": self.bits,
            "false_positive_rate": self.false_positive_rate,
            "item_count": self.item_count,
        }

    def to_json_dict(self) -> dict[str, object]:
        return {
            "kind": SOURCE_FILTER_KIND,
            "version": SOURCE_FILTER_VERSION,
            "bit_count": self.bit_count,
            "hash_count": self.hash_count,
            "bits_base64": base64.b64encode(self.bits).decode("ascii"),
            "false_positive_rate": self.false_positive_rate,
            "item_count": self.item_count,
        }

    @property
    def byte_count(self) -> int:
        return len(self.bits)


def _require_positive_int(value: object, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        fail(
            f"Source filter field '{field_name}' must be a positive integer.",
            "Regenerate the artifact to rebuild the source filter payload.",
        )
    return value


def _require_non_negative_int(value: object, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        fail(
            f"Source filter field '{field_name}' must be a non-negative integer.",
            "Regenerate the artifact to rebuild the source filter payload.",
        )
    return value


def _require_rate(value: object) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        fail(
            "Source filter false-positive rate must be numeric.",
            "Regenerate the artifact to rebuild the source filter payload.",
        )
    rate = float(value)
    if not 0 < rate < 1:
        fail(
            "Source filter false-positive rate must be between 0 and 1.",
            "Regenerate the artifact to rebuild the source filter payload.",
        )
    return rate


def bloom_filter_from_mapping(mapping: object) -> BloomSourceFilter | None:
    if mapping is None:
        return None
    if not isinstance(mapping, dict):
        fail(
            "Source filter payload must be a mapping.",
            "Regenerate the artifact to rebuild the source filter payload.",
        )

    kind = mapping.get("kind")
    version = mapping.get("version")
    if kind != SOURCE_FILTER_KIND or version != SOURCE_FILTER_VERSION:
        fail(
            "Unsupported source filter payload.",
            "Regenerate the artifact with a supported DreamPhraseGPT version.",
        )

    bit_count = _require_positive_int(mapping.get("bit_count"), "bit_count")
    hash_count = _require_positive_int(mapping.get("hash_count"), "hash_count")
    item_count = _require_non_negative_int(mapping.get("item_count"), "item_count")
    false_positive_rate = _require_rate(mapping.get("false_positive_rate"))

    bits_value = mapping.get("bits")
    if not isinstance(bits_value, (bytes, bytearray)):
        fail(
            "Source filter bit payload is missing or invalid.",
            "Regenerate the artifact to rebuild the source filter payload.",
        )
    bits = bytes(bits_value)

    return BloomSourceFilter(
        bit_count=bit_count,
        hash_count=hash_count,
        bits=bits,
        false_positive_rate=false_positive_rate,
        item_count=item_count,
    )


def _estimate_bit_count(item_count: int, false_positive_rate: float) -> int:
    bits = -item_count * math.log(false_positive_rate) / (math.log(2) ** 2)
    return max(8, math.ceil(bits))


def _estimate_hash_count(bit_count: int, item_count: int) -> int:
    hashes = (bit_count / item_count) * math.log(2)
    return max(1, round(hashes))


def iter_hash_indices(text: str, bit_count: int, hash_count: int) -> Iterable[int]:
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    first = int.from_bytes(digest[:8], "little")
    second = int.from_bytes(digest[8:16], "little") or _FALLBACK_HASH_STEP
    for offset in range(hash_count):
        yield (first + offset * second) % bit_count


def build_bloom_source_filter(
    texts: Iterable[str],
    *,
    false_positive_rate: float = DEFAULT_SOURCE_FILTER_FALSE_POSITIVE_RATE,
) -> BloomSourceFilter:
    normalized = sorted(
        {normalized_text for text in texts if (normalized_text := normalize_source_text(text))}
    )
    if not normalized:
        fail(
            "Cannot build a source filter from an empty text set.",
            "Provide at least one non-empty line in the dataset.",
        )

    bit_count = _estimate_bit_count(len(normalized), false_positive_rate)
    hash_count = _estimate_hash_count(bit_count, len(normalized))
    bits = bytearray(math.ceil(bit_count / 8))

    for text in normalized:
        for index in iter_hash_indices(text, bit_count, hash_count):
            byte_index, bit_offset = divmod(index, 8)
            bits[byte_index] |= 1 << bit_offset

    return BloomSourceFilter(
        bit_count=bit_count,
        hash_count=hash_count,
        bits=bytes(bits),
        false_positive_rate=false_positive_rate,
        item_count=len(normalized),
    )


def build_dataset_source_filter(
    dataset: Dataset,
    *,
    false_positive_rate: float = DEFAULT_SOURCE_FILTER_FALSE_POSITIVE_RATE,
) -> BloomSourceFilter:
    return build_bloom_source_filter(
        iter_dataset_documents(dataset),
        false_positive_rate=false_positive_rate,
    )


def resolve_source_filter(
    dataset: Dataset,
    source_filter: BloomSourceFilter | None = None,
) -> BloomSourceFilter:
    return source_filter or build_dataset_source_filter(dataset)
