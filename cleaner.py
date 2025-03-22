# core/cleaner.py

import pandas as pd
from typing import Dict, Any, Tuple
from core.loader import DataLoader
from core.profiler import DataProfiler


class Cleaner:
    def __init__(
        self, df_raw: pd.DataFrame, column_profiles: Dict[str, Dict[str, Any]]
    ):
        self.df_clean = df_raw.copy()
        self.column_profiles = column_profiles
        self.insights = []

    def clean_data(self) -> Tuple[pd.DataFrame, list]:
        print("Cleaning data...")

        for col, profile in self.column_profiles.items():
            if "column_type" not in profile:
                self.insights.append(f"Skipping '{col}' - no column_type detected")
                continue

            if profile["missing_count"] > 0:
                if profile["column_type"] == "numerical":
                    median_val = profile.get("median", self.df_clean[col].median())
                    self.df_clean[col] = self.df_clean[col].fillna(median_val)
                    self.insights.append(
                        f"Filled {profile['missing_count']} missing in '{col}' with median: {median_val}"
                    )

                elif profile["column_type"] == "categorical":
                    mode_val = self.df_clean[col].mode().iloc[0]
                    self.df_clean[col] = self.df_clean[col].fillna(mode_val)
                    self.insights.append(
                        f"Filled {profile['missing_count']} missing in '{col}' with mode: {mode_val}"
                    )

                elif profile["column_type"] == "text":
                    self.df_clean[col] = self.df_clean[col].fillna("")
                    self.insights.append(
                        f"Filled {profile['missing_count']} missing in '{col}' with empty string"
                    )

                elif profile["column_type"] == "date":
                    self.df_clean[f"{col}_MISSING"] = self.df_clean[col].isna()
                    median_date = pd.to_datetime(self.df_clean[col]).median()
                    self.df_clean[col] = self.df_clean[col].fillna(median_date)
                    self.insights.append(
                        f"Filled {profile['missing_count']} missing in '{col}' with median date: {median_date}"
                    )

            if (
                profile["column_type"] == "categorical"
                and "Inconsistent capitalization or spacing"
                in profile["potential_issues"]
            ):
                self.df_clean[col] = (
                    self.df_clean[col].astype(str).str.strip().str.title()
                )
                self.insights.append(f"Standardized capitalization/spacing in '{col}'")

            if profile["column_type"] == "date":
                try:
                    self.df_clean[col] = pd.to_datetime(self.df_clean[col])
                except Exception:
                    self.insights.append(f"Failed to convert '{col}' to datetime")

            if (
                profile["column_type"] == "numerical"
                and profile.get("outlier_count", 0) > 0
            ):
                Q1 = self.df_clean[col].quantile(0.25)
                Q3 = self.df_clean[col].quantile(0.75)
                IQR = Q3 - Q1
                outlier_mask = (self.df_clean[col] < (Q1 - 1.5 * IQR)) | (
                    self.df_clean[col] > (Q3 + 1.5 * IQR)
                )
                self.df_clean[f"{col}_OUTLIER"] = outlier_mask
                self.insights.append(
                    f"Flagged outliers in '{col}' using IQR: {outlier_mask.sum()} rows"
                )

        print("Cleaning completed.")
        return self.df_clean, self.insights


if __name__ == "__main__":
    loader = DataLoader()
    df_raw = loader.load_data()

    profiler = DataProfiler(df_raw)
    profiles = profiler.profile_columns()

    cleaner = Cleaner(df_raw, profiles)
    df_cleaned, insights = cleaner.clean_data()

    print("\nSample cleaned data:")
    print(df_cleaned.head())

    print("\nCleaning insights:")
    for insight in insights:
        print(f"- {insight}")
