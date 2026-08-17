"""Plate string normalisation, repair and validation."""
from __future__ import annotations

import pytest

from backend import plate_text
from backend.rto_lookup import rto_engine  # noqa: F401 - registers valid state codes


@pytest.mark.parametrize("raw,expected", [
    ("MH12DE1433", "MH12DE1433"),
    ("  mh-12 de 1433 ", "MH12DE1433"),
    ("MH.12*DE#1433", "MH12DE1433"),
    ("IND MH12DE1433", "MH12DE1433"),
    ("INDMH12DE1433", "MH12DE1433"),
])
def test_cleaning_and_badge_removal(raw, expected):
    assert plate_text.parse(raw).text == expected


@pytest.mark.parametrize("raw", [
    "MH12DE1433", "DL8CAV1234", "KA03MG5678", "TS09EA5678",
    "MH121234", "GJ01AB1", "UP32KJ9012",
])
def test_valid_standard_plates(raw):
    reading = plate_text.parse(raw)
    assert reading.is_valid
    assert reading.plate_format == "standard"


def test_rto_number_split_is_not_greedy():
    """MH121234 is MH-12 / 1234, not MH-121 / 234."""
    reading = plate_text.parse("MH121234")
    assert (reading.state_code, reading.rto_number, reading.number) == ("MH", "12", "1234")


def test_bharat_series_recognised():
    reading = plate_text.parse("22BH1234AA")
    assert reading.is_valid
    assert reading.plate_format == "bharat"
    assert reading.state_code == "BH"


@pytest.mark.parametrize("raw,expected", [
    ("MHI2DE1433", "MH12DE1433"),   # I misread for 1 in the RTO number
    ("MHO9DE1433", "MH09DE1433"),   # O misread for 0 in the RTO number
    ("MH12DE14O0", "MH12DE1400"),   # O misread for 0 in the number
    ("MH12DE143S", "MH12DE1435"),   # S misread for 5 in the number
])
def test_ambiguous_characters_are_repaired_where_the_format_demands_it(raw, expected):
    reading = plate_text.parse(raw)
    assert reading.text == expected
    assert reading.is_valid
    assert reading.corrections > 0


def test_a_reading_that_is_already_valid_is_left_alone():
    """MH1ZDE1433 is MH-1 with series ZDE, so it must not be rewritten to MH12.

    Repairs are only applied when the reading cannot otherwise be a legal plate;
    guessing on top of a valid reading is how the old normaliser corrupted data.
    """
    reading = plate_text.parse("MH1ZDE1433")
    assert reading.is_valid
    assert reading.corrections == 0
    assert (reading.state_code, reading.rto_number, reading.series) == ("MH", "1", "ZDE")


def test_repair_prefers_the_fewest_edits():
    """MHIZDE1433 needs one fix as MH-1/ZDE, or two as MH-12/DE.

    The single-edit reading wins. Preferring minimum edits is what keeps the
    parser from confidently rewriting plates into a different RTO.
    """
    reading = plate_text.parse("MHIZDE1433")
    assert reading.text == "MH1ZDE1433"
    assert reading.corrections == 1


def test_repairs_are_counted_so_callers_can_judge_trust():
    assert plate_text.parse("MH12DE1433").corrections == 0
    assert plate_text.parse("MHI2DE1433").corrections == 1


def test_state_code_is_never_invented():
    """The old normaliser rewrote RR->RJ and NM->MH, changing real plates."""
    reading = plate_text.parse("RR12AB1234")
    assert reading.is_valid is False
    assert reading.text == "RR12AB1234", "must not silently become RJ12AB1234"


@pytest.mark.parametrize("noise", ["MM72M0", "W10UWMWMWMV", "GARBAGE", "AAAA", "12"])
def test_ocr_noise_is_rejected(noise):
    assert plate_text.parse(noise).is_valid is False


def test_empty_input_is_safe():
    reading = plate_text.parse("")
    assert reading.text == ""
    assert reading.is_valid is False


def test_fragments_are_joined_for_two_line_plates():
    reading = plate_text.extract_best(["MH12", "DE1433"])
    assert reading.text == "MH12DE1433"
    assert reading.is_valid


@pytest.mark.parametrize("fragments", [
    ["MH12DE", "E1433"],       # boxes overlap on the series letter
    ["MH12D", "DE1433"],       # overlap on D
    ["MH12DE1", "1433"],       # overlap on the first digit
])
def test_overlapping_ocr_boxes_do_not_duplicate_characters(fragments):
    """Two OCR boxes covering one plate used to concatenate into MH12DEE1433."""
    assert plate_text.extract_best(fragments).text == "MH12DE1433"


def test_many_small_fragments_are_assembled():
    assert plate_text.extract_best(["MH", "12", "DE", "1433"]).text == "MH12DE1433"


@pytest.mark.parametrize("partial", ["MH12", "MH1", "DL8"])
def test_partial_reads_are_not_accepted_as_whole_plates(partial):
    """'MH12' satisfies the bare grammar as MH + 1 + '2', but is not a plate."""
    assert plate_text.parse(partial).is_valid is False


@pytest.mark.parametrize("junk", [
    "FORD", "YPLANETFOR", "HOSTEOONAOPYRIEHTPASNEOTIVEBHPOWNERE",
    "PLANETFORD", "TEAMBHP",
])
def test_badges_stickers_and_watermarks_are_not_plates(junk):
    assert plate_text.extract_best([junk]).is_valid is False


def test_fragments_ignore_country_badge_token():
    reading = plate_text.extract_best(["IND", "TS09", "EA", "5678"])
    assert reading.text == "TS09EA5678"


def test_best_fragment_wins_over_partial_reads():
    reading = plate_text.extract_best(["XY", "MH12DE1433"])
    assert reading.text == "MH12DE1433"
