from pathlib import Path

def get_model_path(model: str) -> Path:
    """
    Returns the path to the model directory.
    """
    ROOT = Path.cwd()
    if not (ROOT / "models").exists():
        # We're probably in notebooks/, go one level up
        ROOT = ROOT.parent

    print("Project root assumed as:", ROOT)
    print("Models dir exists:", (ROOT / "models").exists())

    model_path = ROOT / "models" / model
    return model_path