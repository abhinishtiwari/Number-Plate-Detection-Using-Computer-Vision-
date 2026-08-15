import csv
import json
import re
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
RTO_DB_JSON = BASE_DIR / "data" / "rto_database.json"
RTO_DB_CSV = BASE_DIR / "data" / "rto_database.csv"

# State Code to State Name Fallback Lookup
STATE_NAMES = {
  "AN": "Andaman and Nicobar Islands", "AP": "Andhra Pradesh", "AR": "Arunachal Pradesh",
  "AS": "Assam", "BR": "Bihar", "CG": "Chhattisgarh", "CH": "Chandigarh",
  "DD": "Daman and Diu", "DL": "Delhi", "DN": "Dadra and Nagar Haveli",
  "GA": "Goa", "GJ": "Gujarat", "HR": "Haryana", "HP": "Himachal Pradesh",
  "JH": "Jharkhand", "JK": "Jammu and Kashmir", "KA": "Karnataka", "KL": "Kerala",
  "LA": "Ladakh", "LD": "Lakshadweep", "MH": "Maharashtra", "ML": "Meghalaya",
  "MN": "Manipur", "MP": "Madhya Pradesh", "MZ": "Mizoram", "NL": "Nagaland",
  "OD": "Odisha", "PB": "Punjab", "PY": "Puducherry", "RJ": "Rajasthan",
  "SK": "Sikkim", "TN": "Tamil Nadu", "TR": "Tripura", "TS": "Telangana",
  "UK": "Uttarakhand", "UP": "Uttar Pradesh", "WB": "West Bengal"
}

class RTOLookupEngine:
    def __init__(self, db_path=None):
        self.db_path = Path(db_path) if db_path else RTO_DB_JSON
        self.rto_map = {}
        self._load_database()

    def _load_database(self):
        """Loads 1000+ RTO dataset records from JSON or CSV into lookup map."""
        if RTO_DB_JSON.exists():
            try:
                with open(RTO_DB_JSON, 'r', encoding='utf-8') as f:
                    records = json.load(f)
                    for item in records:
                        prefix = item.get("registration_prefix", "").upper()
                        if prefix:
                            self.rto_map[prefix] = item
                return
            except Exception as e:
                print(f"[RTOLookup] Error loading JSON database: {e}")

        if RTO_DB_CSV.exists():
            try:
                with open(RTO_DB_CSV, 'r', encoding='utf-8') as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        prefix = row.get("registration_prefix", "").strip().upper()
                        if prefix:
                            self.rto_map[prefix] = {
                                "state_code": row.get("state_code", "").strip(),
                                "state_name": row.get("state_name", "").strip(),
                                "rto_code": row.get("rto_code", "").strip(),
                                "full_rto_code": row.get("full_rto_code", "").strip(),
                                "registration_prefix": prefix,
                                "city": row.get("city", "").strip()
                            }
            except Exception as e:
                print(f"[RTOLookup] Error loading CSV database: {e}")

    def lookup(self, plate_text: str):
        """
        Parses OCR plate string to extract state, RTO code, and city from local dataset.
        Example: MP09AB1234 -> registration_prefix: MP09 -> State: Madhya Pradesh, RTO: MP-09, City: Indore
        """
        if not plate_text or plate_text in ["Not detected", "Unknown"]:
            return {
                "state_name": "Not detected",
                "full_rto_code": "Not detected",
                "city": "Not detected",
                "state_code": "Unknown"
            }

        clean = re.sub(r'[^A-Z0-9]', '', plate_text.upper())
        if len(clean) < 3:
            return {
                "state_name": "Unknown",
                "full_rto_code": "Unknown",
                "city": "Unknown",
                "state_code": "Unknown"
            }

        # 1. Direct 4-char prefix lookup (e.g. MP09, RJ14, DL08, DL8C, MH12, UP32, KA03)
        prefix4 = clean[:4]
        if prefix4 in self.rto_map:
            match = self.rto_map[prefix4]
            return {
                "state_name": match.get("state_name", "Unknown"),
                "full_rto_code": match.get("full_rto_code", "Unknown"),
                "city": match.get("city", "Unknown"),
                "state_code": match.get("state_code", "Unknown")
            }

        # 2. Extract State code + 2-digit RTO number via regex e.g. MP09 from MP09AB1234
        regex_match = re.match(r'^([A-Z]{2}\d{1,2})', clean)
        if regex_match:
            code = regex_match.group(1)
            state_st = code[:2]
            num_st = code[2:]
            padded_code = f"{state_st}{int(num_st):02d}" if num_st.isdigit() else code

            if padded_code in self.rto_map:
                match = self.rto_map[padded_code]
                return {
                    "state_name": match.get("state_name", "Unknown"),
                    "full_rto_code": match.get("full_rto_code", "Unknown"),
                    "city": match.get("city", "Unknown"),
                    "state_code": match.get("state_code", "Unknown")
                }

            if state_st in STATE_NAMES:
                rto_str = f"{state_st}-{int(num_st):02d}" if num_st.isdigit() else f"{state_st}-{num_st}"
                return {
                    "state_name": STATE_NAMES[state_st],
                    "full_rto_code": rto_str,
                    "city": "Regional RTO",
                    "state_code": state_st
                }

        # 3. State prefix fallback e.g. MP, RJ, DL, MH
        state_prefix = clean[:2]
        if state_prefix in STATE_NAMES:
            return {
                "state_name": STATE_NAMES[state_prefix],
                "full_rto_code": f"{state_prefix}-RTO",
                "city": "Regional Center",
                "state_code": state_prefix
            }

        return {
            "state_name": "Unknown",
            "full_rto_code": "Unknown",
            "city": "Unknown",
            "state_code": "Unknown"
        }

# Global singleton instance
rto_engine = RTOLookupEngine()
