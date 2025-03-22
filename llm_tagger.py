import os
import json
import time
import hashlib
import requests
import pandas as pd
from tqdm import tqdm
from typing import Dict
from core.config import (
    DEFAULT_OUTPUT_DIR,
    LLM_API_KEY,
    LLM_API_ENDPOINT,
    MODEL_NAME,
)
from core.loader import DataLoader
from core.profiler import DataProfiler
from core.cleaner import Cleaner


class LLMTagger:
    def __init__(self, df_clean: pd.DataFrame):
        self.df_clean = df_clean
        self.llm_cache = {}

    def _generate_llm_cache_key(self, prompt: str) -> str:
        return hashlib.md5(prompt.encode()).hexdigest()

    def _call_llm(self, prompt: str, temperature: float = 0.2) -> Dict[str, str]:
        cache_key = self._generate_llm_cache_key(prompt)
        if cache_key in self.llm_cache:
            return self.llm_cache[cache_key]

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {LLM_API_KEY}",
        }

        payload = {
            "model": MODEL_NAME,
            "messages": [
                {
                    "role": "system",
                    "content": "You are an expert in vehicle repair classification.",
                },
                {"role": "user", "content": prompt},
            ],
            "temperature": temperature,
        }

        try:
            response = requests.post(LLM_API_ENDPOINT, headers=headers, json=payload)
            response.raise_for_status()
            content = response.json()["choices"][0]["message"]["content"]
            json_str = (
                content.split("```json")[-1].split("```")[0].strip()
                if "```json" in content
                else content
            )
            result = json.loads(json_str)
        except Exception:
            result = {"issues": [], "components": [], "actions": []}

        for key in ["issues", "components", "actions"]:
            if key not in result:
                result[key] = []

        self.llm_cache[cache_key] = result
        return result

    def extract_tags(self) -> pd.DataFrame:
        print("Running LLM Tagging...")

        issues_col, components_col, actions_col = [], [], []

        for _, row in tqdm(self.df_clean.iterrows(), total=len(self.df_clean)):
            correction = str(row.get("CORRECTION_VERBATIM", "")).strip()
            customer = str(row.get("CUSTOMER_VERBATIM", "")).strip()

            if not correction and not customer:
                issues_col.append("")
                components_col.append("")
                actions_col.append("")
                continue

            taxonomy = {
                "issues": [
                    "Material problems (peeling, delaminating, bubbling, sticky)",
                    "Stitching failures",
                    "Heating malfunctions",
                    "Super Cruise/driver assistance failures",
                    "Noise (clicking, rubbing)",
                    "Horn malfunctions",
                    "Electrical faults/DTCs",
                    "Loose/protruding components",
                    "Broken internal circuits",
                    "Off-center alignment",
                ],
                "components": [
                    "Steering wheel (main component)",
                    "Heated module",
                    "Super Cruise/driver assistance module",
                    "Leather/material covering",
                    "Trim/bezel",
                    "Electrical harness/wiring",
                    "Control buttons/switches",
                    "Horn mechanism",
                    "Airbag assembly",
                    "BCM (Body Control Module)",
                ],
                "actions": [
                    "Part replacement",
                    "Pre-authorization (PRA)",
                    "Circuit testing/diagnosis",
                    "Module programming",
                    "DTC clearing",
                    "Torquing/alignment",
                    "Road testing",
                    "Technical assistance consultation",
                    "Disassembly/reassembly",
                    "System verification",
                ],
            }

            taxonomy_str = (
                f"Issue options: {', '.join(taxonomy['issues'])}\n"
                f"Component options: {', '.join(taxonomy['components'])}\n"
                f"Action options: {', '.join(taxonomy['actions'])}"
            )

            prompt = f"""
You are an expert in vehicle repair classification.

CLASSIFY ONLY USING THESE TAXONOMY OPTIONS:
{taxonomy_str}

Analyze the following service record and extract:
- issues
- components
- actions

Text 1 (Customer Verbatim): {customer}
Text 2 (Correction Verbatim): {correction}

Return strictly in this JSON format:
{{
"issues": ["..."],
"components": ["..."],
"actions": ["..."]
}}

Rules:
- Use only the provided taxonomy options (exact or closest match).
- At least 1 issue, 1 component, and 1 action should be returned if identifiable.
- If nothing is found, return an empty list [] for that field.
"""

            result = self._call_llm(prompt)
            issues_col.append(", ".join(result["issues"]) if result["issues"] else "")
            components_col.append(
                ", ".join(result["components"]) if result["components"] else ""
            )
            actions_col.append(
                ", ".join(result["actions"]) if result["actions"] else ""
            )
            time.sleep(0.2)

        self.df_clean["ISSUES"] = issues_col
        self.df_clean["COMPONENTS"] = components_col
        self.df_clean["ACTIONS"] = actions_col

        print("LLM Tagging complete.")
        return self.df_clean

    def extract_advanced_tags(self) -> pd.DataFrame:
        print("Extracting advanced logic-based tags...")

        self.df_clean["REPAIR_COMPLEXITY"] = ""
        self.df_clean["VEHICLE_SYSTEM"] = ""
        self.df_clean["FAILURE_MODE"] = ""
        self.df_clean["REPAIR_URGENCY"] = ""
        self.df_clean["DIAGNOSTIC_METHOD"] = ""

        for idx, row in tqdm(self.df_clean.iterrows(), total=len(self.df_clean)):
            correction = str(row.get("CORRECTION_VERBATIM", "")).strip()
            customer = str(row.get("CUSTOMER_VERBATIM", "")).strip()
            repair_age = row.get("REPAIR_AGE", 0)
            kilometers = row.get("KM", 0)
            total_cost = row.get("TOTALCOST", 0)
            labor_cost = row.get("LBRCOST", 0)
            global_labor_desc = str(
                row.get("GLOBAL_LABOR_CODE_DESCRIPTION", "")
            ).strip()

            if isinstance(kilometers, str):
                kilometers = float(
                    "".join(c for c in kilometers if c.isdigit() or c == ".") or 0
                )
            if isinstance(total_cost, str):
                total_cost = float(
                    "".join(c for c in total_cost if c.isdigit() or c == ".") or 0
                )
            if isinstance(labor_cost, str):
                labor_cost = float(
                    "".join(c for c in labor_cost if c.isdigit() or c == ".") or 0
                )

            parts_cost = (
                total_cost - labor_cost
                if isinstance(total_cost, (int, float))
                and isinstance(labor_cost, (int, float))
                else 0
            )
            combined_text = f"{correction} {customer} {global_labor_desc}".lower()

            # REPAIR_COMPLEXITY
            if (
                parts_cost > 1000
                or "replaced" in combined_text
                and ("module" in combined_text or "programming" in combined_text)
            ):
                complexity = "High"
            elif parts_cost > 300 or "replaced" in combined_text:
                complexity = "Medium"
            elif any(
                term in combined_text for term in ["adjusted", "cleaned", "checked"]
            ):
                complexity = "Low"
            else:
                complexity = "Medium"
            self.df_clean.at[idx, "REPAIR_COMPLEXITY"] = complexity

            # VEHICLE_SYSTEM
            if "super cruise" in combined_text or "driver assist" in combined_text:
                system = "Driver Assistance"
            elif "heated" in combined_text and "wheel" in combined_text:
                system = "Comfort System"
            elif "horn" in combined_text:
                system = "Electrical System"
            elif "wheel" in combined_text and any(
                x in combined_text for x in ["trim", "cover", "plastic"]
            ):
                system = "Interior Trim"
            elif "wheel" in combined_text:
                system = "Steering System"
            else:
                system = "Other"
            self.df_clean.at[idx, "VEHICLE_SYSTEM"] = system

            # FAILURE_MODE
            if any(
                x in combined_text
                for x in ["coming apart", "peeling", "fraying", "bubbling"]
            ):
                failure = "Material Degradation"
            elif any(x in combined_text for x in ["loose", "protruding"]):
                failure = "Assembly Issue"
            elif any(
                x in combined_text for x in ["inop", "not working", "malfunction"]
            ):
                failure = "Functional Failure"
            elif any(x in combined_text for x in ["dtc", "code", "light", "message"]):
                failure = "Electronic/Software Issue"
            elif any(x in combined_text for x in ["noise", "clicking", "rubbing"]):
                failure = "NVH Issue"
            else:
                failure = "Unknown"
            self.df_clean.at[idx, "FAILURE_MODE"] = failure

            # REPAIR_URGENCY
            if repair_age == 0:
                urgency = "Immediate"
            elif "safety" in combined_text or "airbag" in combined_text:
                urgency = "Safety Critical"
            elif kilometers and kilometers < 20000 and repair_age < 6:
                urgency = "Early Failure"
            elif repair_age > 24:
                urgency = "Long-term Issue"
            else:
                urgency = "Normal"
            self.df_clean.at[idx, "REPAIR_URGENCY"] = urgency

            # DIAGNOSTIC_METHOD
            if "tac case" in combined_text or "technical assistance" in combined_text:
                method = "Technical Assistance"
            elif "dtc" in combined_text or "scan" in combined_text:
                method = "Diagnostic Scan"
            elif "test" in combined_text and "circuit" in combined_text:
                method = "Circuit Testing"
            elif "prog" in combined_text:
                method = "Software Reprogramming"
            elif "pra" in combined_text:
                method = "Pre-authorization Required"
            elif "replace" in combined_text:
                method = "Direct Replacement"
            else:
                method = "Visual Inspection"
            self.df_clean.at[idx, "DIAGNOSTIC_METHOD"] = method

        print("Advanced tagging completed.")
        return self.df_clean


if __name__ == "__main__":
    loader = DataLoader()
    df_raw = loader.load_data()

    profiler = DataProfiler(df_raw)
    profiles = profiler.profile_columns()

    cleaner = Cleaner(df_raw, profiles)
    df_cleaned, _ = cleaner.clean_data()

    tagger = LLMTagger(df_cleaned)
    df_tagged = tagger.extract_tags()
    df_tagged = tagger.extract_advanced_tags()

    os.makedirs(DEFAULT_OUTPUT_DIR, exist_ok=True)
    output_path = os.path.join(DEFAULT_OUTPUT_DIR, "tagged_repair_data.xlsx")
    df_tagged.to_excel(output_path, index=False)

    print(f"\nOutput saved to: {output_path}")
