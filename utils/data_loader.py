import pandas as pd

class DataLoader:
    def load_csv(self, file_path):
        try:
            return pd.read_csv(file_path)
        except FileNotFoundError as e:
            raise FileNotFoundError(f"{file_path} not found.") from e

    def load_text(self, file_path):
        try:
            with open(file_path, 'r') as file:
                return file.read()
        except FileNotFoundError as e:
            raise FileNotFoundError(f"{file_path} not found.") from e
