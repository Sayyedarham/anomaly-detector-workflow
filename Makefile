.PHONY: install test lint benchmark app
install:
	python -m pip install -r requirements.txt
test:
	python -m pytest -q
lint:
	python -m ruff check src tests
benchmark:
	python -m anomaly_bench run
app:
	streamlit run src/anomaly_bench/app.py
