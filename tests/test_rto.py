import unittest
from backend.rto_lookup import rto_engine

class TestRTOLookup(unittest.TestCase):
    def test_mh12_pune(self):
        res = rto_engine.lookup("MH12DE1433")
        self.assertEqual(res["state_name"], "Maharashtra")
        self.assertEqual(res["full_rto_code"], "MH-12")
        self.assertIn("Pune", res["city"])

    def test_mp09_indore(self):
        res = rto_engine.lookup("MP09AB1234")
        self.assertEqual(res["state_name"], "Madhya Pradesh")
        self.assertEqual(res["full_rto_code"], "MP-09")
        self.assertIn("Indore", res["city"])

    def test_rj14_jaipur(self):
        res = rto_engine.lookup("RJ14CV0002")
        self.assertEqual(res["state_name"], "Rajasthan")
        self.assertEqual(res["full_rto_code"], "RJ-14")
        self.assertIn("Jaipur", res["city"])

    def test_dl08_delhi(self):
        res = rto_engine.lookup("DL8CAV1234")
        self.assertEqual(res["state_name"], "Delhi")
        self.assertEqual(res["full_rto_code"], "DL-08")

    def test_up32_lucknow(self):
        res = rto_engine.lookup("UP32KJ9012")
        self.assertEqual(res["state_name"], "Uttar Pradesh")
        self.assertEqual(res["full_rto_code"], "UP-32")
        self.assertIn("Lucknow", res["city"])

    def test_dataset_size(self):
        all_recs = rto_engine.get_all_records()
        self.assertGreaterEqual(len(all_recs), 1000)

    def test_not_detected(self):
        res = rto_engine.lookup("Not detected")
        self.assertEqual(res["state_name"], "Not detected")
        self.assertEqual(res["full_rto_code"], "Not detected")
        self.assertEqual(res["city"], "Not detected")

if __name__ == "__main__":
    unittest.main()
