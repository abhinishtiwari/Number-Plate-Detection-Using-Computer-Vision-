"""RTO lookup backed by the India_RTO_Registration_Dataset_New.csv master file.

Design notes
------------
* The CSV is the single source of truth. There is no second copy and no
  hard-coded state table to drift out of sync; the state-code to state-name map
  is derived from the dataset itself.
* Nothing is invented. When a prefix is not in the dataset the response says so
  (``match_level`` of ``state`` or ``none``) instead of returning a made-up city
  such as "Regional RTO".
* A malformed row is skipped with a warning rather than poisoning the map.
"""
from __future__ import annotations

import csv
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from . import plate_text
from .config import RTO_DATASET_CANDIDATES

logger = logging.getLogger(__name__)

REQUIRED_COLUMNS = ("state_code", "state_name", "rto_code", "full_rto_code",
                    "registration_prefix", "city")

NOT_DETECTED = "Not detected"


@dataclass(frozen=True)
class RTORecord:
    state_code: str
    state_name: str
    rto_code: str
    full_rto_code: str
    registration_prefix: str
    city: str

    def as_dict(self) -> dict:
        return {
            "state_code": self.state_code,
            "state_name": self.state_name,
            "rto_code": self.rto_code,
            "full_rto_code": self.full_rto_code,
            "registration_prefix": self.registration_prefix,
            "city": self.city,
        }


class DatasetNotFound(RuntimeError):
    """Raised when no RTO dataset CSV can be located."""


def _resolve_dataset_path(candidates: Iterable[Path]) -> Path:
    tried = []
    for candidate in candidates:
        tried.append(str(candidate))
        if candidate.is_file():
            return candidate
    raise DatasetNotFound(
        "RTO dataset CSV not found. Looked in: " + ", ".join(tried)
        + ". Set RTO_DATASET_PATH to point at India_RTO_Registration_Dataset_New.csv."
    )


class RTOLookupEngine:
    """Prefix -> (state, RTO code, city) resolver."""

    def __init__(self, dataset_path: Path | str | None = None):
        candidates = (Path(dataset_path),) if dataset_path else RTO_DATASET_CANDIDATES
        self.dataset_path = _resolve_dataset_path(candidates)
        self._records: list[RTORecord] = []
        self._by_prefix: dict[str, RTORecord] = {}
        self._state_names: dict[str, str] = {}
        self._load()
        # Let the plate parser reject readings with impossible state codes.
        plate_text.set_known_state_codes(self._state_names.keys())

    # ------------------------------------------------------------ loading

    def _load(self) -> None:
        with self.dataset_path.open(encoding="utf-8-sig", newline="") as fh:
            reader = csv.DictReader(fh)
            missing = [c for c in REQUIRED_COLUMNS if c not in (reader.fieldnames or [])]
            if missing:
                raise ValueError(
                    f"{self.dataset_path.name} is missing required column(s): {', '.join(missing)}"
                )

            skipped = 0
            for line_no, row in enumerate(reader, start=2):
                record = self._build_record(row, line_no)
                if record is None:
                    skipped += 1
                    continue
                if record.registration_prefix in self._by_prefix:
                    logger.warning(
                        "Duplicate registration_prefix %s on line %d ignored.",
                        record.registration_prefix, line_no,
                    )
                    skipped += 1
                    continue
                self._by_prefix[record.registration_prefix] = record
                self._records.append(record)
                self._state_names.setdefault(record.state_code, record.state_name)

        if not self._records:
            raise ValueError(f"{self.dataset_path.name} contains no usable rows.")

        logger.info(
            "Loaded %d RTO records covering %d states/UTs from %s (%d row(s) skipped).",
            len(self._records), len(self._state_names), self.dataset_path.name, skipped,
        )

    @staticmethod
    def _build_record(row: dict, line_no: int) -> RTORecord | None:
        def cell(key: str) -> str:
            return (row.get(key) or "").strip()

        state_code = cell("state_code").upper()
        prefix = cell("registration_prefix").upper()
        rto_code = cell("rto_code")

        if not state_code or not prefix:
            logger.warning("Skipping line %d: missing state_code/registration_prefix.", line_no)
            return None
        if not rto_code.isdigit():
            logger.warning("Skipping line %d: non-numeric rto_code %r.", line_no, rto_code)
            return None

        rto_code = f"{int(rto_code):02d}"
        expected_prefix = f"{state_code}{rto_code}"
        if prefix != expected_prefix:
            logger.warning(
                "Line %d: registration_prefix %r does not match %r; using the derived value.",
                line_no, prefix, expected_prefix,
            )
            prefix = expected_prefix

        return RTORecord(
            state_code=state_code,
            state_name=cell("state_name") or state_code,
            rto_code=rto_code,
            full_rto_code=cell("full_rto_code") or f"{state_code}-{rto_code}",
            registration_prefix=prefix,
            city=cell("city"),
        )

    # ------------------------------------------------------------- queries

    @property
    def record_count(self) -> int:
        return len(self._records)

    @property
    def state_count(self) -> int:
        return len(self._state_names)

    def get_all_records(self) -> list[dict]:
        return [r.as_dict() for r in self._records]

    def state_name(self, state_code: str) -> str | None:
        return self._state_names.get(state_code.upper())

    def lookup(self, reading: "plate_text.PlateReading | str | None") -> dict:
        """Resolve a plate reading to its registering authority.

        ``match_level`` tells the caller exactly how much was resolved:
        ``exact`` (prefix found), ``state`` (state code known, RTO not in the
        dataset), ``national`` (Bharat series), or ``none``.
        """
        if reading is None:
            return self._empty(NOT_DETECTED)

        if isinstance(reading, str):
            if not reading or reading == NOT_DETECTED:
                return self._empty(NOT_DETECTED)
            reading = plate_text.parse(reading)

        if not reading.text:
            return self._empty(NOT_DETECTED)

        if reading.plate_format == "bharat":
            return {
                "state_name": "Bharat Series (valid nationwide)",
                "state_code": "BH",
                "full_rto_code": "BH",
                "city": None,
                "match_level": "national",
                "registration_prefix": "BH",
            }

        if reading.plate_format == "defence":
            return {
                "state_name": "Defence / Military registration",
                "state_code": "",
                "full_rto_code": None,
                "city": None,
                "match_level": "national",
                "registration_prefix": None,
            }

        state_code = reading.state_code
        if not state_code:
            return self._empty("Unknown")

        if reading.rto_number:
            prefix = f"{state_code}{int(reading.rto_number):02d}"
            record = self._by_prefix.get(prefix)
            if record is not None:
                result = record.as_dict()
                result["match_level"] = "exact"
                return result

        state_name = self.state_name(state_code)
        if state_name:
            rto_display = (
                f"{state_code}-{int(reading.rto_number):02d}" if reading.rto_number.isdigit() else state_code
            )
            return {
                "state_name": state_name,
                "state_code": state_code,
                "full_rto_code": rto_display,
                "city": None,
                "match_level": "state",
                "registration_prefix": None,
            }

        return self._empty("Unknown")

    @staticmethod
    def _empty(label: str) -> dict:
        return {
            "state_name": label,
            "state_code": "",
            "full_rto_code": None,
            "city": None,
            "match_level": "none",
            "registration_prefix": None,
        }


#: Process-wide instance. Constructed eagerly so a missing/broken dataset fails
#: at import time instead of silently returning "Unknown" for every request.
rto_engine = RTOLookupEngine()
