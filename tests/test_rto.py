"""RTO dataset integrity and lookup behaviour."""
from __future__ import annotations

import csv

import pytest

from backend.rto_lookup import REQUIRED_COLUMNS, RTOLookupEngine, rto_engine


@pytest.fixture(scope="module")
def raw_rows():
    with rto_engine.dataset_path.open(encoding="utf-8-sig", newline="") as fh:
        return list(csv.DictReader(fh))


# --------------------------------------------------------------- dataset shape


def test_dataset_file_is_the_master_csv():
    assert rto_engine.dataset_path.name == "India_RTO_Registration_Dataset_New.csv"


def test_dataset_has_required_columns(raw_rows):
    assert set(REQUIRED_COLUMNS).issubset(raw_rows[0].keys())


def test_dataset_has_no_blank_fields(raw_rows):
    blank = [r["registration_prefix"] for r in raw_rows
             if not all(str(v).strip() for v in r.values())]
    assert blank == [], f"rows with empty fields: {blank}"


def test_dataset_has_no_duplicate_prefixes(raw_rows):
    prefixes = [r["registration_prefix"].strip().upper() for r in raw_rows]
    duplicates = {p for p in prefixes if prefixes.count(p) > 1}
    assert not duplicates, f"duplicate prefixes: {sorted(duplicates)}"


def test_prefix_matches_state_and_rto_code(raw_rows):
    mismatched = [
        r["registration_prefix"] for r in raw_rows
        if r["registration_prefix"].strip() != r["state_code"].strip() + r["rto_code"].strip()
        or r["full_rto_code"].strip() != f"{r['state_code'].strip()}-{r['rto_code'].strip()}"
    ]
    assert mismatched == [], f"inconsistent rows: {mismatched[:10]}"


def test_one_state_name_per_state_code(raw_rows):
    names: dict[str, set[str]] = {}
    for row in raw_rows:
        names.setdefault(row["state_code"].strip(), set()).add(row["state_name"].strip())
    conflicts = {k: v for k, v in names.items() if len(v) > 1}
    assert not conflicts, f"conflicting state names: {conflicts}"


@pytest.mark.parametrize("state_code", ["TS", "TG", "LA", "DL", "MH", "AP", "OD", "UK", "CG"])
def test_states_previously_missing_are_present(state_code):
    """Telangana and Ladakh were absent from the dataset entirely."""
    assert rto_engine.state_name(state_code) is not None


def test_dataset_covers_every_state_and_ut():
    # 28 states + 8 UTs, plus TS and TG both mapping to Telangana.
    assert rto_engine.state_count >= 36
    assert rto_engine.record_count >= 1100


# -------------------------------------------------------------------- lookups


@pytest.mark.parametrize("plate,state,rto,city_fragment", [
    ("MH12DE1433", "Maharashtra", "MH-12", "Pune"),
    ("MP09AB1234", "Madhya Pradesh", "MP-09", "Indore"),
    ("RJ14CV0002", "Rajasthan", "RJ-14", "Jaipur"),
    ("UP32KJ9012", "Uttar Pradesh", "UP-32", "Lucknow"),
    ("KA03MG5678", "Karnataka", "KA-03", "Bangalore"),
    ("TS09EA5678", "Telangana", "TS-09", "Hyderabad"),
    ("TG09EA5678", "Telangana", "TG-09", "Hyderabad"),
    ("LA01A1234", "Ladakh", "LA-01", "Leh"),
])
def test_exact_prefix_lookups(plate, state, rto, city_fragment):
    result = rto_engine.lookup(plate)
    assert result["match_level"] == "exact"
    assert result["state_name"] == state
    assert result["full_rto_code"] == rto
    assert city_fragment in result["city"]


def test_single_digit_rto_is_zero_padded():
    """DL8CAV1234 must resolve through the DL08 prefix."""
    result = rto_engine.lookup("DL8CAV1234")
    assert result["state_name"] == "Delhi"
    assert result["full_rto_code"] == "DL-08"
    assert result["match_level"] == "exact"


def test_bharat_series_is_national_not_a_state():
    result = rto_engine.lookup("22BH1234AA")
    assert result["match_level"] == "national"
    assert result["state_code"] == "BH"
    assert result["city"] is None


def test_unknown_rto_number_resolves_state_only_without_inventing_a_city():
    result = rto_engine.lookup("MH97XX1234")
    assert result["match_level"] == "state"
    assert result["state_name"] == "Maharashtra"
    assert result["city"] is None, "must not fabricate a city such as 'Regional RTO'"


def test_ocr_noise_is_not_mapped_to_a_state():
    """'MM72M0' fits the plate grammar but MM is not an Indian state code."""
    result = rto_engine.lookup("MM72M0")
    assert result["match_level"] == "none"
    assert result["state_name"] == "Unknown"


@pytest.mark.parametrize("value", ["", None, "Not detected"])
def test_missing_text_reports_not_detected(value):
    result = rto_engine.lookup(value)
    assert result["state_name"] == "Not detected"
    assert result["city"] is None


def test_missing_dataset_raises_instead_of_returning_unknown(tmp_path):
    from backend.rto_lookup import DatasetNotFound

    with pytest.raises(DatasetNotFound):
        RTOLookupEngine(dataset_path=tmp_path / "nope.csv")


def test_malformed_rows_are_skipped(tmp_path):
    path = tmp_path / "rto.csv"
    path.write_text(
        "state_code,state_name,rto_code,full_rto_code,registration_prefix,city\n"
        "MH,Maharashtra,12,MH-12,MH12,Pune\n"
        "XX,Bad,notanumber,XX-1,XX1,Nowhere\n"
        ",Missing,01,-01,,Nowhere\n",
        encoding="utf-8",
    )
    engine = RTOLookupEngine(dataset_path=path)
    assert engine.record_count == 1
    assert engine.lookup("MH12AB1234")["city"] == "Pune"
