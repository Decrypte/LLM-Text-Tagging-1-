import os
import pandas as pd
from typing import Optional, Tuple
from core.config import DEFAULT_INPUT_FILE


class DataLoader:
    def __init__(self, input_file: Optional[str] = None):
        self.input_file = input_file or DEFAULT_INPUT_FILE
        self.df_raw = None
        self.taxonomy = None

    def load_data(self) -> pd.DataFrame:
        print("Loading main data...")
        if not os.path.exists(self.input_file):
            raise FileNotFoundError(f"Input file not found: {self.input_file}")

        self.df_raw = pd.read_excel(self.input_file)
        print(f"Main data loaded with shape {self.df_raw.shape}")
        return self.df_raw


if __name__ == "__main__":
    loader = DataLoader()
    df = loader.load_data()
    print("\nSample data preview:")
    print(df.head())
