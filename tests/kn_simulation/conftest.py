import sys
from pathlib import Path

SRC = Path(__file__).resolve().parents[2] / "kn_simulation" / "src"
sys.path.insert(0, str(SRC))
