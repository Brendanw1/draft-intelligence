.PHONY: verify validate honesty-report backtest evidence all clean

PYTHON := .venv/bin/python
PYTHONPATH := src

all: verify validate honesty-report backtest evidence

verify:
	@echo "=== Running comprehensive verification ==="
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) scripts/verify_all.py

validate:
	@echo "=== Running Tier 5 validation gates ==="
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) scripts/validate_tier5.py

honesty-report:
	@echo "=== Generating honesty report ==="
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) scripts/tier5_honesty_report.py

backtest:
	@echo "=== Running Tier 5 E2E backtest ==="
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) scripts/backtest_tier5.py --report

evidence:
	@echo "=== Generating Tier 5 evidence bundle ==="
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) scripts/tier5_evidence_bundle.py

clean:
	@echo "=== Cleaning generated artifacts ==="
	rm -f docs/tier5_honesty_report.md
	@echo "Done"
