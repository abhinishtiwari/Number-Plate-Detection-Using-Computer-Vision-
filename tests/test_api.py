"""End-to-end API behaviour: upload -> detection -> OCR -> RTO -> JSON."""
from __future__ import annotations

import os
import tempfile

import cv2
import numpy as np
import pytest

from conftest import encode_jpeg, encode_webp, render_plate_image

PLATE = "MH12DE1433"


# ------------------------------------------------------------------- /health


def test_health_reports_dataset_and_engines(client):
    body = client.get("/health").json()
    assert body["status"] == "ok"
    assert body["rto_dataset"]["records"] >= 1100
    assert body["rto_dataset"]["file"] == "India_RTO_Registration_Dataset_New.csv"
    # The UI depends on these to avoid claiming a capability that is absent.
    assert isinstance(body["ocr"]["available"], list)
    assert "detection_sources" in body


def test_rto_dataset_endpoint_returns_records(client):
    body = client.get("/rto-dataset").json()
    assert body["count"] == len(body["records"]) >= 1100
    assert {"registration_prefix", "state_name", "city"} <= body["records"][0].keys()


# ------------------------------------------------------------------- /detect


def test_rejects_empty_file(client):
    response = client.post("/detect", files={"file": ("a.jpg", b"", "image/jpeg")})
    assert response.status_code == 400


def test_rejects_non_media_file(client):
    response = client.post("/detect", files={"file": ("a.txt", b"hello world", "text/plain")})
    assert response.status_code == 400


def test_rejects_corrupt_image(client):
    response = client.post("/detect", files={"file": ("a.jpg", b"not an image" * 40, "image/jpeg")})
    assert response.status_code == 400
    assert "corrupt" in response.json()["detail"].lower()


def test_rejects_oversized_upload(client):
    from backend.config import MAX_UPLOAD_BYTES

    payload = b"x" * (MAX_UPLOAD_BYTES + 1024)
    response = client.post("/detect", files={"file": ("big.jpg", payload, "image/jpeg")})
    assert response.status_code == 413


def test_undecodable_video_is_an_error_not_an_empty_success(client):
    """This used to return 200 with an empty list, hiding the failure."""
    response = client.post("/detect", files={"file": ("a.mp4", b"nope" * 200, "video/mp4")})
    assert response.status_code == 400


def test_missing_content_type_falls_back_to_the_extension(client):
    image = render_plate_image(PLATE)
    response = client.post("/detect", files={"file": ("car.jpg", encode_jpeg(image), None)})
    assert response.status_code == 200


def test_webp_upload_uses_safe_decoder(client):
    """WebP decoding must not crash the worker (observed as Render HTTP 502)."""
    image = render_plate_image(PLATE)
    response = client.post(
        "/detect", files={"file": ("car.webp", encode_webp(image), "image/webp")}
    )
    assert response.status_code == 200
    assert response.json()["media_type"] == "image"


def test_image_without_a_plate_returns_no_results(client):
    blank = np.full((420, 620, 3), 120, np.uint8)
    body = client.post("/detect", files={"file": ("f.jpg", encode_jpeg(blank), "image/jpeg")}).json()
    assert body["plates"] == []
    assert body["plate_count"] == 0


def test_plate_is_read_and_mapped_end_to_end(client, neural_ocr_available):
    if not neural_ocr_available:
        pytest.skip("no neural OCR engine installed")

    image = render_plate_image(PLATE)
    body = client.post("/detect", files={"file": ("car.jpg", encode_jpeg(image), "image/jpeg")}).json()

    assert body["media_type"] == "image"
    assert body["plate_count"] == 1
    plate = body["plates"][0]

    assert plate["text"] == PLATE
    assert plate["is_valid_format"] is True
    assert plate["state_name"] == "Maharashtra"
    assert plate["full_rto_code"] == "MH-12"
    assert plate["city"] == "Pune"
    assert plate["rto_match_level"] == "exact"
    # Confidence must come from the OCR engine, not a synthesised constant.
    assert plate["confidence_basis"].startswith("ocr:")
    assert 0 < plate["confidence"] <= 100


def test_reported_box_contains_the_plate(client, neural_ocr_available):
    """The drawn box must be the region the text came from."""
    if not neural_ocr_available:
        pytest.skip("no neural OCR engine installed")

    box = (250, 300, 380, 105)
    image = render_plate_image(PLATE, plate_box=box)
    body = client.post("/detect", files={"file": ("car.jpg", encode_jpeg(image), "image/jpeg")}).json()

    x, y, w, h = body["plates"][0]["box"]
    px, py, pw, ph = box
    inter_w = max(0, min(x + w, px + pw) - max(x, px))
    inter_h = max(0, min(y + h, py + ph) - max(y, py))
    assert inter_w * inter_h / (pw * ph) > 0.6


def test_no_result_reuse_between_uploads(client, neural_ocr_available):
    """A second image must not inherit the first image's plate."""
    if not neural_ocr_available:
        pytest.skip("no neural OCR engine installed")

    first = client.post(
        "/detect", files={"file": ("a.jpg", encode_jpeg(render_plate_image("MP09AB1234")), "image/jpeg")}
    ).json()
    second = client.post(
        "/detect", files={"file": ("b.jpg", encode_jpeg(render_plate_image("RJ14CV0002")), "image/jpeg")}
    ).json()
    blank = client.post(
        "/detect", files={"file": ("c.jpg", encode_jpeg(np.full((420, 620, 3), 120, np.uint8)), "image/jpeg")}
    ).json()

    assert first["plates"][0]["text"] == "MP09AB1234"
    assert second["plates"][0]["text"] == "RJ14CV0002"
    assert blank["plates"] == []


def test_multiple_plates_are_all_returned(client, neural_ocr_available):
    if not neural_ocr_available:
        pytest.skip("no neural OCR engine installed")

    image = np.full((900, 1500, 3), 90, np.uint8)
    for (x, y, text) in ((120, 200, "MP09AB1234"), (820, 600, "TS09EA5678")):
        cv2.rectangle(image, (x, y), (x + 420, y + 110), (245, 245, 245), -1)
        cv2.rectangle(image, (x, y), (x + 420, y + 110), (20, 20, 20), 3)
        cv2.putText(image, text, (x + 10, y + 80), cv2.FONT_HERSHEY_SIMPLEX,
                    1.7, (15, 15, 15), 4, cv2.LINE_AA)

    body = client.post("/detect", files={"file": ("two.jpg", encode_jpeg(image), "image/jpeg")}).json()
    found = {p["text"] for p in body["plates"]}
    assert {"MP09AB1234", "TS09EA5678"} <= found

    states = {p["text"]: p["state_name"] for p in body["plates"]}
    assert states["MP09AB1234"] == "Madhya Pradesh"
    assert states["TS09EA5678"] == "Telangana"


def test_high_resolution_upload_is_handled(client, neural_ocr_available):
    """A 3x upscaled photo used to fail the hard-coded pixel-area filter."""
    if not neural_ocr_available:
        pytest.skip("no neural OCR engine installed")

    image = cv2.resize(render_plate_image(PLATE), None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
    body = client.post("/detect", files={"file": ("big.jpg", encode_jpeg(image), "image/jpeg")}).json()
    assert body["plates"], "no plate found in the high-resolution image"
    assert body["plates"][0]["text"] == PLATE
    # Boxes must be reported in original-image coordinates.
    x, y, w, h = body["plates"][0]["box"]
    assert w > 380 and x > 380


def test_video_results_come_from_real_frames(client, neural_ocr_available):
    if not neural_ocr_available:
        pytest.skip("no neural OCR engine installed")

    path = os.path.join(tempfile.gettempdir(), "npai_pytest_clip.mp4")
    writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), 24, (880, 540))
    try:
        for i in range(72):
            writer.write(render_plate_image("RJ14CV0002", plate_box=(120 + i * 4, 300, 380, 105)))
        writer.release()

        with open(path, "rb") as fh:
            body = client.post("/detect", files={"file": ("clip.mp4", fh.read(), "video/mp4")}).json()
    finally:
        if os.path.exists(path):
            os.remove(path)

    assert body["media_type"] == "video"
    assert body["frames_scanned"] > 0
    assert body["fps"] == pytest.approx(24.0, abs=1.0)

    plate = body["plates"][0]
    assert plate["text"] == "RJ14CV0002"
    assert plate["city"] == "Jaipur"
    # Frame provenance lets the UI seek to the sighting.
    assert isinstance(plate["frame_index"], int)
    assert plate["timestamp_seconds"] >= 0
    assert plate["times_seen"] >= 1


def test_only_the_number_plate_is_returned(client, neural_ocr_available):
    """A real photo also contains a badge, a dealer sticker and a watermark.

    Those used to come back as extra "Vehicle #N" entries with unverified text,
    alongside a dozen unreadable chrome-trim boxes.
    """
    if not neural_ocr_available:
        pytest.skip("no neural OCR engine installed")

    image = np.full((760, 1020, 3), 150, np.uint8)
    # Manufacturer badge
    cv2.putText(image, "FORD", (430, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (20, 20, 60), 4, cv2.LINE_AA)
    # Chrome trim: a long high-contrast strip that yields contour candidates
    cv2.rectangle(image, (60, 210), (960, 250), (30, 30, 30), -1)
    # The actual plate
    cv2.rectangle(image, (180, 290), (860, 460), (250, 250, 250), -1)
    cv2.rectangle(image, (180, 290), (860, 460), (15, 15, 15), 5)
    cv2.putText(image, "MH12DE1433", (200, 410), cv2.FONT_HERSHEY_SIMPLEX, 2.6, (10, 10, 10), 8, cv2.LINE_AA)
    # Dealer sticker and site watermark
    cv2.putText(image, "Planet Ford", (740, 620), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (40, 90, 40), 3, cv2.LINE_AA)
    cv2.putText(image, "Team-BHP.com", (40, 720), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (60, 20, 20), 3, cv2.LINE_AA)

    body = client.post("/detect", files={"file": ("car.jpg", encode_jpeg(image), "image/jpeg")}).json()

    assert [p["text"] for p in body["plates"]] == ["MH12DE1433"]
    plate = body["plates"][0]
    assert plate["state_name"] == "Maharashtra"
    assert plate["city"] == "Pune"
    # No unverified or unreadable entries survive.
    assert all(p["is_valid_format"] for p in body["plates"])


def test_response_has_no_placeholder_strings(client, neural_ocr_available):
    """Unresolved fields must be null, not invented text."""
    if not neural_ocr_available:
        pytest.skip("no neural OCR engine installed")

    image = render_plate_image(PLATE)
    body = client.post("/detect", files={"file": ("car.jpg", encode_jpeg(image), "image/jpeg")}).json()
    forbidden = {"Regional RTO", "Regional Center", "DETECTED_PLATE", "N/A", "Unknown"}
    for plate in body["plates"]:
        for key in ("state_name", "city", "full_rto_code", "text"):
            assert plate[key] not in forbidden
