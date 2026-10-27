"""Automated tests for the API. Run from the project root:

    python -m unittest discover -s backend -p "test_*.py" -v

They use Flask's built-in test client, so no server needs to be running.
CAR_MODEL=sklearn makes the tests use the scikit-learn MLP even if TensorFlow and a .keras file exist,
so the results are the same on every computer.
"""
import os
import sys
import unittest

os.environ["CAR_MODEL"] = "sklearn"
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from app import app  # noqa: E402

VALID = {
    "brand": "Maruti", "model_segment": "Hatchback", "year": 2020, "km_driven": 45000,
    "fuel": "Petrol", "transmission": "Manual", "owners": 1, "city": "Pune",
    "seller_type": "Individual", "mileage_kmpl": 21.0, "engine_cc": 1200,
}
ELECTRIC = {**VALID, "brand": "Tata", "model_segment": "SUV", "year": 2023, "fuel": "Electric",
            "transmission": "Automatic", "mileage_kmpl": "", "engine_cc": ""}


class UsedCarApiTests(unittest.TestCase):
    def setUp(self):
        self.client = app.test_client()

    def post(self, payload):
        return self.client.post("/api/predict", json=payload)

    # ---- health and options ----
    def test_health_reports_fallback(self):
        r = self.client.get("/api/health")
        self.assertEqual(r.status_code, 200)
        body = r.get_json()
        self.assertEqual(body["status"], "ok")
        self.assertTrue(body["using_fallback"])
        self.assertIn("MLP", body["ann_model"])

    def test_options(self):
        data = self.client.get("/api/options").get_json()
        self.assertIn("Maruti", data["categories"]["brand"])
        self.assertIn("Electric", data["categories"]["fuel"])
        self.assertNotIn("Bangalore", data["categories"]["city"])   # cleaned in the notebook
        self.assertIn("year", data["numeric_ranges"])

    # ---- happy paths ----
    def test_valid_prediction(self):
        r = self.post(VALID)
        self.assertEqual(r.status_code, 200)
        body = r.get_json()
        price = body["predicted_price_lakhs"]
        self.assertGreater(price, 0)
        low, high = body["likely_range_lakhs"]
        self.assertLess(low, price)
        self.assertGreater(high, price)
        self.assertGreater(body["comparison"]["predicted_price_lakhs"], 0)
        print("\n  sample prediction:", body)

    def test_electric_without_mileage(self):
        r = self.post(ELECTRIC)
        self.assertEqual(r.status_code, 200, r.get_json())
        print("\n  electric prediction:", r.get_json()["predicted_price_lakhs"])

    def test_engine_size_is_optional(self):
        payload = dict(VALID)
        del payload["engine_cc"]
        self.assertEqual(self.post(payload).status_code, 200)

    def test_numbers_sent_as_text_and_city_alias(self):
        r = self.post({**VALID, "km_driven": "45000", "year": "2020", "city": "Bangalore"})
        self.assertEqual(r.status_code, 200)

    def test_newer_car_costs_more(self):
        old = self.post({**VALID, "year": 2012, "km_driven": 120000}).get_json()["predicted_price_lakhs"]
        new = self.post({**VALID, "year": 2023, "km_driven": 15000}).get_json()["predicted_price_lakhs"]
        self.assertGreater(new, old)

    # ---- validation errors ----
    def test_missing_field(self):
        payload = dict(VALID)
        del payload["brand"]
        r = self.post(payload)
        self.assertEqual(r.status_code, 422)
        self.assertIn("brand", r.get_json()["fields"])

    def test_mileage_required_for_petrol(self):
        r = self.post({**VALID, "mileage_kmpl": ""})
        self.assertEqual(r.status_code, 422)
        self.assertEqual(r.get_json()["fields"]["mileage_kmpl"], "This field is required.")

    def test_electric_must_not_send_mileage(self):
        r = self.post({**ELECTRIC, "mileage_kmpl": 18})
        self.assertEqual(r.status_code, 422)
        self.assertIn("Electric", r.get_json()["fields"]["mileage_kmpl"])

    def test_electric_too_old(self):
        r = self.post({**ELECTRIC, "year": 2012})
        self.assertEqual(r.status_code, 422)
        self.assertIn("year", r.get_json()["fields"])

    def test_not_a_number(self):
        r = self.post({**VALID, "km_driven": "a lot"})
        self.assertEqual(r.status_code, 422)
        self.assertEqual(r.get_json()["fields"]["km_driven"], "Must be a number.")

    def test_km_out_of_range(self):
        r = self.post({**VALID, "km_driven": 4_500_000})
        self.assertEqual(r.status_code, 422)
        self.assertIn("between", r.get_json()["fields"]["km_driven"])

    def test_year_out_of_range(self):
        r = self.post({**VALID, "year": 1995})
        self.assertEqual(r.status_code, 422)
        self.assertIn("between", r.get_json()["fields"]["year"])

    def test_owners_whole_number(self):
        r = self.post({**VALID, "owners": 1.5})
        self.assertEqual(r.status_code, 422)
        self.assertEqual(r.get_json()["fields"]["owners"], "Must be a whole number.")

    def test_unknown_category(self):
        r = self.post({**VALID, "brand": "Ferrari", "fuel": "Hydrogen"})
        self.assertEqual(r.status_code, 422)
        fields = r.get_json()["fields"]
        self.assertIn("Unknown value", fields["brand"])
        self.assertIn("fuel", fields)

    def test_body_not_json(self):
        r = self.client.post("/api/predict", data="hello", content_type="text/plain")
        self.assertEqual(r.status_code, 400)

    def test_json_but_not_an_object(self):
        r = self.post([1, 2, 3])
        self.assertEqual(r.status_code, 422)

    def test_cors_headers_present(self):
        r = self.client.get("/api/health")
        self.assertEqual(r.headers["Access-Control-Allow-Origin"], "http://localhost:5173")
        pre = self.client.options("/api/predict")
        self.assertEqual(pre.status_code, 204)


if __name__ == "__main__":
    unittest.main()
