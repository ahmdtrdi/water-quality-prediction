install:
	pip install -r requirements.txt
	pip install -e .

train:
	python entrypoint/run_train.py

test:
	pytest tests/

clean:
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete