import subprocess
import sys
from pathlib import Path

if __name__ == "__main__":
    for tutorial in sorted(Path(__file__).parent.iterdir()):
        if (tutorial / "generate_models.py").exists():
            subprocess.call([sys.executable, str(tutorial / "generate_models.py")])
