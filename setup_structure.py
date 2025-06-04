import os

folders = [
    "data/raw", "data/processed",
    "models/timegan", "models/diffusion",
    "experiments",
    "evaluation/plots",
    "backend/src",
    "frontend/assets",
    "docs/diagrams", "docs/draft",
    "outputs/timegan", "outputs/diffusion"
]

for folder in folders:
    os.makedirs(folder, exist_ok=True)

for folder in set(f.split('/')[0] for f in folders):
    with open(f"{folder}/README.md", 'w') as f:
        f.write(f"# {folder.capitalize()}\n\nDescription of {folder}/ folder.")



