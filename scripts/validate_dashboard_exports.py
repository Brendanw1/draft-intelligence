from pathlib import Path
import sys

from mlb_draft_dashboard.config import EXPORTS_DIR
from mlb_draft_dashboard.export_validation import validate_exports_dir, validation_report_text


def main() -> int:
    exports_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(EXPORTS_DIR)
    issues = validate_exports_dir(exports_dir)
    print(validation_report_text(issues))
    return 1 if any(issue.level == "error" for issue in issues) else 0


if __name__ == "__main__":
    raise SystemExit(main())
