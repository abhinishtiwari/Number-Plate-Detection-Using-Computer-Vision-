"""Plate localisation behaviour."""
from __future__ import annotations

import cv2
import numpy as np
import pytest

from backend.detector import PlateCandidate, PlateDetector
from backend.ocr_engine import ocr_engine
from conftest import render_plate_image


@pytest.fixture(scope="module")
def detector():
    return PlateDetector(ocr=ocr_engine)


def _covers(candidates, box, tolerance=0.5):
    """True when some candidate overlaps ``box`` by more than ``tolerance``."""
    bx, by, bw, bh = box
    for c in candidates:
        inter_w = max(0, min(c.x + c.w, bx + bw) - max(c.x, bx))
        inter_h = max(0, min(c.y + c.h, by + bh) - max(c.y, by))
        if inter_w * inter_h / (bw * bh) > tolerance:
            return True
    return False


def test_empty_input_returns_no_candidates(detector):
    assert detector.detect(None) == []
    assert detector.detect(np.zeros((0, 0, 3), np.uint8)) == []


def test_flat_image_produces_no_plates(detector):
    """A plain background must not yield a fabricated box."""
    assert detector.detect(np.full((400, 600, 3), 120, np.uint8)) == []


def test_finds_the_plate_region(detector):
    box = (250, 300, 380, 105)
    image = render_plate_image("MH12DE1433", plate_box=box)
    assert _covers(detector.detect(image), box)


@pytest.mark.parametrize("scale", [0.6, 1.0, 2.0, 3.0])
def test_detection_is_resolution_independent(detector, scale):
    """Absolute pixel-area limits used to reject plates in high-res photos."""
    box = (250, 300, 380, 105)
    image = render_plate_image("MH12DE1433", plate_box=box)
    scaled = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    expected = tuple(int(v * scale) for v in box)
    assert _covers(detector.detect(scaled), expected), f"missed the plate at scale {scale}"


def test_boxes_stay_inside_the_image(detector):
    image = render_plate_image("MH12DE1433")
    h, w = image.shape[:2]
    for c in detector.detect(image):
        assert 0 <= c.x < w and 0 <= c.y < h
        assert c.x + c.w <= w and c.y + c.h <= h


def test_one_candidate_per_plate(detector):
    """A tight text box nested in a plate-border box is the same plate."""
    image = render_plate_image("MH12DE1433", plate_box=(250, 300, 380, 105))
    assert len(detector.detect(image)) == 1


def test_two_plates_are_reported_separately(detector):
    image = np.full((900, 1500, 3), 90, np.uint8)
    for (x, y) in ((120, 200), (820, 600)):
        cv2.rectangle(image, (x, y), (x + 420, y + 110), (245, 245, 245), -1)
        cv2.rectangle(image, (x, y), (x + 420, y + 110), (20, 20, 20), 3)
        cv2.putText(image, "MP09AB1234", (x + 10, y + 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.7, (15, 15, 15), 4, cv2.LINE_AA)
    candidates = detector.detect(image)
    assert len(candidates) >= 2
    assert _covers(candidates, (120, 200, 420, 110))
    assert _covers(candidates, (820, 600, 420, 110))


def test_geometric_candidates_carry_no_fake_confidence(detector):
    """Contour/Haar proposals have no calibrated score, so it must be None."""
    image = render_plate_image("MH12DE1433")
    for candidate in detector.detect(image):
        if candidate.source in ("contour", "haar", "text"):
            assert candidate.detector_confidence is None


def test_active_sources_are_reported_honestly(detector):
    sources = detector.active_sources
    assert "contour" in sources
    # No weights are shipped, so YOLO must not be advertised.
    assert "yolo" not in sources


def test_nested_boxes_are_merged():
    outer = PlateCandidate(x=100, y=100, w=400, h=120, source="haar")
    inner = PlateCandidate(x=140, y=130, w=300, h=55, source="text", text_hint="MH12DE1433")
    kept = PlateDetector._non_max_suppression(PlateDetector(ocr=None), [outer, inner])
    assert len(kept) == 1
    # The surviving box inherits the text that was already read.
    assert kept[0].text_hint == "MH12DE1433"
