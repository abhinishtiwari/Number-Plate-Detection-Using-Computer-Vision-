"""Shared fixtures. Also puts the repository root on sys.path so the tests can
import the `backend` package whichever directory pytest is invoked from."""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def render_plate_image(text: str, width: int = 880, height: int = 540,
                       plate_box: tuple[int, int, int, int] = (250, 300, 380, 105)) -> np.ndarray:
    """Synthesise a vehicle-like frame containing one white plate."""
    image = np.full((height, width, 3), 70, np.uint8)
    x, y, w, h = plate_box
    cv2.rectangle(image, (x, y), (x + w, y + h), (245, 245, 245), -1)
    cv2.rectangle(image, (x, y), (x + w, y + h), (20, 20, 20), 3)
    scale = (w / len(text)) / 24.0
    cv2.putText(image, text, (x + 10, y + int(h * 0.72)),
                cv2.FONT_HERSHEY_SIMPLEX, scale, (15, 15, 15), 4, cv2.LINE_AA)
    return image


def encode_jpeg(image: np.ndarray) -> bytes:
    ok, buffer = cv2.imencode(".jpg", image)
    assert ok, "failed to encode the test image"
    return buffer.tobytes()


@pytest.fixture(scope="session")
def client():
    from fastapi.testclient import TestClient

    from backend.main import app

    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture(scope="session")
def neural_ocr_available() -> bool:
    from backend.ocr_engine import ocr_engine

    return ocr_engine.status()["neural_engine_available"]
