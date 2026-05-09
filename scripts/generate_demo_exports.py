from pathlib import Path

from mlb_draft_dashboard.config import EXPORTS_DIR
from mlb_draft_dashboard.sample_data import write_demo_exports


if __name__ == "__main__":
    write_demo_exports(Path(EXPORTS_DIR))
    print(f"Demo exports written to {EXPORTS_DIR}")
