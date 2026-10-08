# Reproduce the whole analysis:  make env  ->  make data  ->  make test  ->  make notebooks
PY ?= python

.PHONY: data test notebooks export service-test all

data:        ## download (if needed), verify, extract REFIT and build the 1-minute cache
	$(PY) scripts/prepare_data.py

test:        ## unit tests for the library
	$(PY) -m pytest -q

notebooks:   ## execute notebooks 01-06 in order (writes artifacts/)
	$(PY) scripts/run_notebooks.py

export:      ## export the selected model to ONNX for the service
	$(PY) scripts/export_onnx.py --checkpoint $(CKPT) --name $(NAME) --card-extra artifacts/metrics/model_card_eval.json

service-test:
	cd service && $(PY) -m pytest -q tests

all: data test notebooks
