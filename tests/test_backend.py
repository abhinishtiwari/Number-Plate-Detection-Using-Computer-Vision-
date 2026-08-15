import unittest
import sys
from pathlib import Path
from fastapi.testclient import TestClient

# Add project root to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from backend.main import app

class TestBackendAPI(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)

    def test_health_check_endpoint(self):
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"status": "ok"})

    def test_detect_rejects_non_image(self):
        response = self.client.post("/detect", files={"file": ("test.txt", b"hello world", "text/plain")})
        self.assertEqual(response.status_code, 400)

if __name__ == '__main__':
    unittest.main()
