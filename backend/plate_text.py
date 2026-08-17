"""Indian number plate string parsing, normalisation and validation.

The previous implementation forced characters into positions with guesses such
as "RR -> RJ" and "NM -> MH", which silently rewrote legitimate plates. This
module instead generates candidate readings by applying ambiguity swaps only
where the plate *grammar* requires a letter or a digit, then keeps the
candidate that actually validates. Nothing is invented: if no candidate is a
legal Indian registration, the cleaned raw reading is returned and flagged
as unvalidated.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from itertools import product
from typing import Iterable

#: Standard format: 2 letter state, 1-2 digit RTO, 0-3 letter series, 1-4 digit number.
#: Covers MH12DE1433, DL8CAV1234, KA03MG5678 and older MH121234 style plates.
#: RTO numbers never exceed two digits, which keeps the split unambiguous.
STANDARD_RE = re.compile(r"^([A-Z]{2})(\d{1,2})([A-Z]{0,3})(\d{1,4})$")

#: Bharat series: 2 digit year, BH, 4 digit number, 1-2 letter series (22BH1234AA).
BH_RE = re.compile(r"^(\d{2})(BH)(\d{4})([A-Z]{1,2})$")

#: Military / defence plates such as 09B123456A are intentionally out of scope
#: for RTO mapping but are still recognised so they are not mangled.
DEFENCE_RE = re.compile(r"^(\d{2})([A-Z])(\d{6})([A-Z])$")

#: Characters an OCR engine most commonly swaps. Used only at positions where
#: the plate grammar already tells us whether a letter or digit belongs.
DIGIT_FOR_LETTER = {"O": "0", "Q": "0", "D": "0", "I": "1", "L": "1", "J": "1",
                    "T": "1", "Z": "2", "A": "4", "S": "5", "G": "6", "B": "8"}
LETTER_FOR_DIGIT = {"0": "O", "1": "I", "2": "Z", "4": "A", "5": "S", "6": "G", "8": "B"}

#: Tokens printed on Indian plates that are not part of the registration.
NOISE_TOKENS = ("IND", "INDIA", "BHARAT")

MAX_CANDIDATES = 512

#: Shortest string accepted as a complete registration. Without this, a partial
#: read such as "MH12" satisfies the grammar (MH + 1 + "" + 2) and would be
#: reported as a whole plate.
MIN_PLATE_LENGTH = 6

#: Populated from the RTO dataset at startup by :mod:`backend.rto_lookup`.
#: A reading whose state code is not a real Indian state code is rejected, which
#: is what stops OCR noise like "MM72M0" from passing as a plate.
_KNOWN_STATE_CODES: set[str] = set()


def set_known_state_codes(codes: Iterable[str]) -> None:
    """Register the state codes that exist in the RTO dataset."""
    global _KNOWN_STATE_CODES
    _KNOWN_STATE_CODES = {c.strip().upper() for c in codes if c and c.strip()}


def known_state_codes() -> frozenset[str]:
    return frozenset(_KNOWN_STATE_CODES)


def _state_code_allowed(code: str) -> bool:
    if not _KNOWN_STATE_CODES:
        return True
    return code in _KNOWN_STATE_CODES


@dataclass
class PlateReading:
    """Result of interpreting one OCR string."""

    text: str
    """Best available reading. Normalised when valid, cleaned raw otherwise."""

    is_valid: bool = False
    """True when ``text`` matches a known Indian registration grammar."""

    plate_format: str = "unrecognised"
    """One of ``standard``, ``bharat``, ``defence`` or ``unrecognised``."""

    state_code: str = ""
    rto_number: str = ""
    series: str = ""
    number: str = ""
    corrections: int = 0
    """How many ambiguous characters had to be swapped to reach a valid form."""

    raw: str = ""

    def as_dict(self) -> dict:
        return {
            "text": self.text,
            "is_valid": self.is_valid,
            "plate_format": self.plate_format,
            "state_code": self.state_code,
            "rto_number": self.rto_number,
            "series": self.series,
            "number": self.number,
            "corrections": self.corrections,
            "raw": self.raw,
        }


def clean(text: str) -> str:
    """Upper-case and strip everything that cannot appear in a registration."""
    if not text:
        return ""
    cleaned = re.sub(r"[^A-Z0-9]", "", text.upper())
    for token in NOISE_TOKENS:
        # Only strip the country badge when something plate-like remains.
        if cleaned.startswith(token) and len(cleaned) - len(token) >= 6:
            cleaned = cleaned[len(token):]
    return cleaned


def _match_format(candidate: str):
    if len(candidate) >= MIN_PLATE_LENGTH:
        m = STANDARD_RE.match(candidate)
        if m and _state_code_allowed(m.group(1)):
            return "standard", m
    m = BH_RE.match(candidate)
    if m:
        return "bharat", m
    m = DEFENCE_RE.match(candidate)
    if m:
        return "defence", m
    return None, None


def _swap_options(char: str, want: str) -> tuple[str, ...]:
    """Return the possible readings of ``char`` when a letter/digit is wanted."""
    if want == "digit":
        if char.isdigit():
            return (char,)
        mapped = DIGIT_FOR_LETTER.get(char)
        return (mapped,) if mapped else ()
    if want == "letter":
        if char.isalpha():
            return (char,)
        mapped = LETTER_FOR_DIGIT.get(char)
        return (mapped,) if mapped else ()
    return (char,)


def _layouts(length: int) -> Iterable[tuple[str, ...]]:
    """Yield plausible letter/digit layouts for a string of ``length`` chars.

    A standard plate is LL + D{1,3} + L{0,3} + D{1,4}. Enumerating the layouts
    lets us repair a reading without assuming a fixed plate shape.
    """
    for rto_len in (2, 1):
        for series_len in (2, 1, 3, 0):
            num_len = length - 2 - rto_len - series_len
            if 1 <= num_len <= 4:
                yield tuple(
                    ["letter"] * 2 + ["digit"] * rto_len + ["letter"] * series_len + ["digit"] * num_len
                )
    # Bharat series layout: DD BH DDDD L(L)
    for series_len in (2, 1):
        if length == 2 + 2 + 4 + series_len:
            yield tuple(["digit"] * 2 + ["letter"] * 2 + ["digit"] * 4 + ["letter"] * series_len)


def _candidates(cleaned: str) -> Iterable[tuple[str, int]]:
    """Yield (candidate, correction_count) repairs of ``cleaned``."""
    yield cleaned, 0
    seen = {cleaned}
    for layout in _layouts(len(cleaned)):
        per_position = []
        for char, want in zip(cleaned, layout):
            options = _swap_options(char, want)
            if not options:
                per_position = None
                break
            per_position.append(options)
        if per_position is None:
            continue
        produced = 0
        for combo in product(*per_position):
            candidate = "".join(combo)
            if candidate in seen:
                continue
            seen.add(candidate)
            corrections = sum(1 for a, b in zip(candidate, cleaned) if a != b)
            yield candidate, corrections
            produced += 1
            if produced >= MAX_CANDIDATES:
                break


def parse(text: str) -> PlateReading:
    """Interpret one OCR string as an Indian registration number."""
    cleaned = clean(text)
    if not cleaned:
        return PlateReading(text="", raw="")

    best: PlateReading | None = None
    for candidate, corrections in _candidates(cleaned):
        fmt, match = _match_format(candidate)
        if not fmt:
            continue
        if fmt == "standard":
            state, rto, series, number = match.groups()
            reading = PlateReading(
                text=candidate, is_valid=True, plate_format=fmt, state_code=state,
                rto_number=rto, series=series, number=number, corrections=corrections,
                raw=cleaned,
            )
        elif fmt == "bharat":
            year, _, number, series = match.groups()
            reading = PlateReading(
                text=candidate, is_valid=True, plate_format=fmt, state_code="BH",
                rto_number=year, series=series, number=number, corrections=corrections,
                raw=cleaned,
            )
        else:
            reading = PlateReading(
                text=candidate, is_valid=True, plate_format=fmt,
                corrections=corrections, raw=cleaned,
            )
        if best is None or reading.corrections < best.corrections:
            best = reading
        if best.corrections == 0:
            break

    if best is not None:
        return best
    return PlateReading(text=cleaned, is_valid=False, raw=cleaned)


MAX_FRAGMENT_OVERLAP = 4


def _merge(left: str, right: str) -> str:
    """Join two fragments, collapsing any repeated characters where they meet.

    OCR engines frequently return two boxes covering the same plate, so the tail
    of one fragment repeats the head of the next. Concatenating blindly turns
    "MH12DE" + "E1433" into "MH12DEE1433"; collapsing the shared "E" gives the
    correct "MH12DE1433".
    """
    limit = min(len(left), len(right), MAX_FRAGMENT_OVERLAP)
    for size in range(limit, 0, -1):
        if left[-size:] == right[:size]:
            return left + right[size:]
    return left + right


def extract_best(fragments: Iterable[str]) -> PlateReading:
    """Pick the best plate reading from OCR fragments.

    Candidates are each fragment on its own, neighbouring fragments merged, and
    all fragments merged. Merging is always overlap-aware: fragments are never
    concatenated blindly, because an engine that returns two boxes for one plate
    would otherwise duplicate the shared characters.

    Ranking is by fewest repairs, then the longest reading, then the simplest
    construction. Fewest repairs first means a clean complete reading beats a
    partial one that only validated after a character was swapped; longest next
    means a truncated fragment cannot beat the assembled plate.
    """
    fragments = [f for f in (clean(f) for f in fragments) if f]
    if not fragments:
        return PlateReading(text="", raw="")

    attempts: list[tuple[int, str]] = [(0, f) for f in fragments]

    if len(fragments) > 1:
        attempts += [(1, _merge(fragments[i], fragments[i + 1]))
                     for i in range(len(fragments) - 1)]

        merged_all = fragments[0]
        for fragment in fragments[1:]:
            merged_all = _merge(merged_all, fragment)
        attempts.append((2, merged_all))

    readings = [(tier, parse(text)) for tier, text in attempts]

    plate_like = [(tier, r) for tier, r in readings
                  if r.is_valid and r.plate_format in ("standard", "bharat")]
    if plate_like:
        plate_like.sort(key=lambda item: (item[1].corrections, -len(item[1].text), item[0]))
        return plate_like[0][1]

    other_valid = [(tier, r) for tier, r in readings if r.is_valid]
    if other_valid:
        other_valid.sort(key=lambda item: (item[1].corrections, -len(item[1].text), item[0]))
        return other_valid[0][1]

    # Nothing validated; return the longest reading so the caller can report what
    # was actually seen, flagged as unverified.
    readings.sort(key=lambda item: -len(item[1].text))
    return readings[0][1]
