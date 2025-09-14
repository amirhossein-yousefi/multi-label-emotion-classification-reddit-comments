.PHONY: train infer lint test

train:
	python -m emoclass.train --config configs/base.yaml

infer:
	python -m emoclass.inference --model_dir outputs/goemotions_roberta --text "I love this!" "This is awful."

test:
	pytest -q

lint:
	ruff src tests
