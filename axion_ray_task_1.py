import pandas as pd
import json
import requests
from tqdm import tqdm
import time

# Config
LLM_API_ENDPOINT = "https://openrouter.ai/api/v1/chat/completions"
LLM_API_KEY = (
    "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"
)
MODEL_NAME = "mistralai/mistral-small-3.1-24b-instruct:free"

# File Paths
input_file = r"C:\Users\tirtn\Downloads\DA - Task 1.xlsx"
output_file = r"C:\Users\tirtn\Downloads\tagged_output.csv"

# Load Data
df_task = pd.read_excel(input_file, sheet_name="Task")
df_taxonomy = pd.read_excel(input_file, sheet_name="Taxonomy")

df_taxonomy.columns = df_taxonomy.columns.str.strip()
taxonomy = {
    "Root Cause": df_taxonomy["Root Cause"].dropna().tolist(),
    "Symptom Condition": df_taxonomy["Symptom Condition"].dropna().tolist(),
    "Symptom Component": df_taxonomy["Symptom Component"].dropna().tolist(),
    "Fix Condition": df_taxonomy["Fix Condition"].dropna().tolist(),
    "Fix Component": df_taxonomy["Fix Component"].dropna().tolist(),
}

# Initialize required columns with NaN instead of "Not Mentioned"
required_columns = [
    "Root Cause",
    "Symptom Condition 1",
    "Symptom Condition 2",
    "Symptom Condition 3",
    "Symptom Component 1",
    "Symptom Component 2",
    "Symptom Component 3",
    "Fix Condition 1",
    "Fix Condition 2",
    "Fix Condition 3",
    "Fix Component 1",
    "Fix Component 2",
    "Fix Component 3",
    "Confidence",
]

for col in required_columns:
    if col not in df_task.columns:
        df_task[col] = pd.NA


def create_llm_prompt(complaint, cause, correction, taxonomy):
    taxonomy_str = "\n".join(
        [
            f"Root Cause options: {', '.join(taxonomy['Root Cause'])}",
            f"Symptom Condition options: {', '.join(taxonomy['Symptom Condition'])}",
            f"Symptom Component options: {', '.join(taxonomy['Symptom Component'])}",
            f"Fix Condition options: {', '.join(taxonomy['Fix Condition'])}",
            f"Fix Component options: {', '.join(taxonomy['Fix Component'])}",
        ]
    )

    return f"""You are an expert in industrial maintenance classification.

ANALYZE THE FOLLOWING MAINTENANCE RECORD:
Complaint: {complaint}
Cause: {cause}
Correction: {correction}

CLASSIFY USING ONLY THESE PREDEFINED CATEGORIES:
{taxonomy_str}

CLASSIFICATION RULES:
1. Extract Root Cause from the Cause field
2. Extract Symptom Conditions (up to 3) from Complaint field
3. Extract Symptom Components (up to 3) from Complaint field
4. Extract Fix Conditions (up to 3) from Correction field
5. Extract Fix Components (up to 3) from Correction field
6. Use "Not Mentioned" if nothing matches
7. Use ONLY terms from the provided options
8. At least ONE Symptom Condition, ONE Symptom Component, ONE Fix Condition, and ONE Fix Component must be identified if possible

RETURN JSON FORMAT:
{{
  "root_cause": "one option from Root Cause list",
  "symptom_condition": ["option1", "option2", "option3"],
  "symptom_component": ["option1", "option2", "option3"],
  "fix_condition": ["option1", "option2", "option3"],
  "fix_component": ["option1", "option2", "option3"],
  "confidence": 0.95
}}
"""


def call_llm_api(prompt):
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {LLM_API_KEY}",
    }
    payload = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "system", "content": "You are a classification assistant."},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.1,
    }

    try:
        print("Calling API...")
        response = requests.post(LLM_API_ENDPOINT, headers=headers, json=payload)
        response.raise_for_status()
        content = response.json()["choices"][0]["message"]["content"]
        print(f"API response received: {content[:100]}...")

        # Extract JSON from code block if present
        if "```json" in content and "```" in content.split("```json", 1)[1]:
            json_str = content.split("```json", 1)[1].split("```", 1)[0].strip()
        else:
            # If not in code block, try to find JSON directly
            json_str = content

        result = json.loads(json_str)
        return result
    except Exception as e:
        print(f"API error: {str(e)}")
        # Return default structure with empty lists for multiple values
        return {
            "root_cause": "Not Mentioned",
            "symptom_condition": ["Not Mentioned", "", ""],
            "symptom_component": ["Not Mentioned", "", ""],
            "fix_condition": ["Not Mentioned", "", ""],
            "fix_component": ["Not Mentioned", "", ""],
            "confidence": 0.0,
        }


def validate_llm_output(llm_output, taxonomy):
    validated = llm_output.copy()

    # Handle root_cause
    if (
        validated.get("root_cause") not in taxonomy["Root Cause"]
        and validated.get("root_cause") != "Not Mentioned"
    ):
        validated["root_cause"] = "Not Mentioned"
        validated["confidence"] *= 0.8

    # Handle list fields
    list_fields = {
        "symptom_condition": "Symptom Condition",
        "symptom_component": "Symptom Component",
        "fix_condition": "Fix Condition",
        "fix_component": "Fix Component",
    }

    for field_key, taxonomy_key in list_fields.items():
        # Ensure the field exists and is a list
        if field_key not in validated or not isinstance(validated[field_key], list):
            validated[field_key] = ["Not Mentioned", "", ""]
            continue

        # Validate each item in the list
        valid_list = []
        for item in validated[field_key]:
            if item in taxonomy[taxonomy_key] or item == "Not Mentioned" or item == "":
                valid_list.append(item)
            else:
                valid_list.append("Not Mentioned")
                validated["confidence"] *= 0.9

        # Ensure we have exactly 3 items (padding with empty strings if needed)
        while len(valid_list) < 3:
            valid_list.append("")

        # Trim to 3 items if more were provided
        validated[field_key] = valid_list[:3]

    return validated


def process_row(row, taxonomy):
    complaint = str(row["Complaint"]) if pd.notna(row["Complaint"]) else ""
    cause = str(row["Cause"]) if pd.notna(row["Cause"]) else ""
    correction = str(row["Correction"]) if pd.notna(row["Correction"]) else ""

    try:
        prompt = create_llm_prompt(complaint, cause, correction, taxonomy)
        llm_result = call_llm_api(prompt)
        return validate_llm_output(llm_result, taxonomy)
    except Exception as e:
        print(f"Error processing row: {e}")
        return {
            "root_cause": "Not Mentioned",
            "symptom_condition": ["Not Mentioned", "", ""],
            "symptom_component": ["Not Mentioned", "", ""],
            "fix_condition": ["Not Mentioned", "", ""],
            "fix_component": ["Not Mentioned", "", ""],
            "confidence": 0.0,
        }


# Initialize confidence to 0 so we know which rows haven't been processed
df_task["Confidence"] = 0.0

# Process each row
for idx, row in tqdm(df_task.iterrows(), total=len(df_task), desc="Processing rows"):
    # Process the row if it hasn't been fully classified
    result = process_row(row, taxonomy)

    # Update the dataframe with classification results
    df_task.at[idx, "Root Cause"] = result["root_cause"]

    # Update Symptom Conditions
    for i, val in enumerate(result["symptom_condition"][:3]):
        df_task.at[idx, f"Symptom Condition {i+1}"] = val if val else "Not Mentioned"

    # Update Symptom Components
    for i, val in enumerate(result["symptom_component"][:3]):
        df_task.at[idx, f"Symptom Component {i+1}"] = val if val else "Not Mentioned"

    # Update Fix Conditions
    for i, val in enumerate(result["fix_condition"][:3]):
        df_task.at[idx, f"Fix Condition {i+1}"] = val if val else "Not Mentioned"

    # Update Fix Components
    for i, val in enumerate(result["fix_component"][:3]):
        df_task.at[idx, f"Fix Component {i+1}"] = val if val else "Not Mentioned"

    # Update confidence
    df_task.at[idx, "Confidence"] = result["confidence"]

    # Add a small delay to avoid rate limiting
    time.sleep(0.5)


df_task.to_csv(output_file, index=False)

# Report on low confidence classifications
low_confidence_rows = df_task[df_task["Confidence"] < 0.7]
if len(low_confidence_rows) > 0:
    print(f"\nWARNING: {len(low_confidence_rows)} rows need manual review.")

print(f"Done. Output saved to {output_file}")
