import pandas as pd
import numpy as np
from collections import Counter
from scipy import stats
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from typing import Dict

import warnings

warnings.filterwarnings("ignore")

import nltk

nltk.download("punkt", quiet=True)
nltk.download("stopwords", quiet=True)


class DataProfiler:
    def __init__(self, df_raw: pd.DataFrame):
        self.df_raw = df_raw
        self.column_profiles = {}
        self.text_columns = []
        self.numerical_columns = []
        self.categorical_columns = []
        self.date_columns = []
        self.outliers = {}

    def profile_columns(self) -> Dict:
        print("Profiling columns...")
        for col in self.df_raw.columns:
            col_data = self.df_raw[col]
            profile = {
                "name": col,
                "dtype": str(col_data.dtype),
                "missing_count": col_data.isna().sum(),
                "missing_percentage": (col_data.isna().sum() / len(col_data)) * 100,
                "unique_count": col_data.nunique(),
                "unique_percentage": (col_data.nunique() / len(col_data)) * 100,
                "sample_values": (
                    col_data.dropna().sample(min(5, len(col_data.dropna()))).tolist()
                    if len(col_data.dropna()) > 0
                    else []
                ),
                "potential_issues": [],
            }

            if pd.api.types.is_numeric_dtype(col_data):
                profile["column_type"] = "numerical"
                self.numerical_columns.append(col)

                profile.update(
                    {
                        "min": col_data.min(),
                        "max": col_data.max(),
                        "mean": col_data.mean(),
                        "median": col_data.median(),
                        "std": col_data.std(),
                    }
                )

                Q1 = col_data.quantile(0.25)
                Q3 = col_data.quantile(0.75)
                IQR = Q3 - Q1
                mask = (col_data < (Q1 - 1.5 * IQR)) | (col_data > (Q3 + 1.5 * IQR))
                outlier_count = mask.sum()
                profile["outlier_count"] = outlier_count
                profile["outlier_percentage"] = (outlier_count / len(col_data)) * 100

                if profile["outlier_percentage"] > 5:
                    profile["potential_issues"].append("High outlier percentage")

                self.outliers[col] = col_data[mask].tolist()

            elif col_data.dtype == "object":
                try:
                    numeric_data = pd.to_numeric(col_data.dropna())
                    profile["column_type"] = "numerical"
                    self.numerical_columns.append(col)
                    profile.update(
                        {
                            "min": numeric_data.min(),
                            "max": numeric_data.max(),
                            "mean": numeric_data.mean(),
                            "median": numeric_data.median(),
                            "std": numeric_data.std(),
                        }
                    )

                    Q1 = numeric_data.quantile(0.25)
                    Q3 = numeric_data.quantile(0.75)
                    IQR = Q3 - Q1
                    mask = (numeric_data < (Q1 - 1.5 * IQR)) | (
                        numeric_data > (Q3 + 1.5 * IQR)
                    )
                    outlier_count = mask.sum()
                    profile["outlier_count"] = outlier_count
                    profile["outlier_percentage"] = (
                        outlier_count / len(numeric_data)
                    ) * 100

                    if profile["outlier_percentage"] > 5:
                        profile["potential_issues"].append("High outlier percentage")

                    self.outliers[col] = numeric_data[mask].tolist()
                    self.column_profiles[col] = profile
                    continue  # Skip remaining checks for this col
                except:
                    pass

            if pd.api.types.is_string_dtype(col_data):
                try:
                    pd.to_datetime(col_data.dropna(), errors="raise")
                    profile["column_type"] = "date"
                    self.date_columns.append(col)
                except:
                    if col_data.str.len().mean() > 20:
                        profile["column_type"] = "text"
                        self.text_columns.append(col)

                        all_text = " ".join(col_data.dropna().astype(str))
                        words = [
                            w.lower()
                            for w in word_tokenize(all_text)
                            if w.isalpha()
                            and w.lower() not in stopwords.words("english")
                        ]
                        profile["top_words"] = Counter(words).most_common(10)
                        profile["avg_length"] = col_data.str.len().mean()
                        profile["max_length"] = col_data.str.len().max()
                    else:
                        profile["column_type"] = "categorical"
                        self.categorical_columns.append(col)
                        value_counts = col_data.value_counts()
                        profile["top_categories"] = value_counts.head(5).to_dict()
                        profile["category_entropy"] = stats.entropy(value_counts.values)

                        clean = col_data.dropna().astype(str).str.strip().str.lower()
                        if clean.nunique() < col_data.dropna().nunique():
                            profile["potential_issues"].append(
                                "Inconsistent capitalization or spacing"
                            )

            elif pd.api.types.is_datetime64_any_dtype(col_data):
                profile["column_type"] = "date"
                self.date_columns.append(col)
                profile["min_date"] = col_data.min()
                profile["max_date"] = col_data.max()
                profile["range_days"] = (profile["max_date"] - profile["min_date"]).days

            # Check for potential primary key
            if (
                profile["unique_count"] == len(col_data)
                and profile["missing_count"] == 0
            ):
                profile["potential_primary_key"] = True
            else:
                profile["potential_primary_key"] = False

            self.column_profiles[col] = profile

        print("Profiling completed.")
        return self.column_profiles


if __name__ == "__main__":
    from core.loader import DataLoader

    loader = DataLoader()
    df = loader.load_data()

    profiler = DataProfiler(df)
    profiles = profiler.profile_columns()

    print("\nSample column profile:")
    for k, v in list(profiles.items()):
        print(f"\nColumn: {k}")
        for key, value in v.items():
            print(f"  {key}: {value}")
