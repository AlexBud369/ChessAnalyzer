import kagglehub
import shutil
import os

path = kagglehub.dataset_download("ronakbadhe/chess-evaluations")
print(f"Dataset downloaded to: {path}")

os.makedirs("data/raw", exist_ok=True)
for file in os.listdir(path):
    if file.endswith(".csv"):
        shutil.copy(os.path.join(path, file), f"data/raw/{file}")
        print(f"Copied {file}")