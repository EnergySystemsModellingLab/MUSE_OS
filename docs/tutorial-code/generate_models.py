import subprocess
import sys
from pathlib import Path

parent_path = Path(__file__).parent
for tutorial in parent_path.iterdir():
    if (tutorial / "generate_models.py").exists():
        subprocess.call([sys.executable, str(tutorial / "generate_models.py")])
