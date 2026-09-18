DOCKER_IMAGE := m2cgen
DOCKER_RUN_ARGS := docker run --rm -it -v "$$PWD":"/m2cgen" $(DOCKER_IMAGE)
TARGETS := ./m2cgen ./tests

clean:
	rm -rf ./m2cgen.egg-info ./dist ./build

flake8:
	flake8 $(TARGETS)

isort:
	isort $(TARGETS)

isort-check:
	isort $(TARGETS) --check-only

test-api:
	pytest -v tests/ --cov=m2cgen/ --cov-report=xml:coverage.xml --ignore=tests/e2e/

package:
	python setup.py bdist_wheel --plat-name=any --python-tag=py3 && \
		python setup.py sdist

publish: clean package
	python -m twine upload dist/*

install-requirements:
	pip install -r requirements-test.txt

install-develop:
	pip install -e .

pre-pr: install-requirements flake8 isort test-api

generate-code-examples: install-develop
	# run as a module so that the repository root is on sys.path: the
	# wittgenstein wheel ships a top-level `tests` package which would
	# otherwise shadow the repository's own tests package
	python -m tools.generate_code_examples ./generated_code_examples

docker-build:
	docker build -t $(DOCKER_IMAGE) .

docker-generate-examples:
	$(DOCKER_RUN_ARGS) make generate-code-examples

docker-lint:
	$(DOCKER_RUN_ARGS) make flake8 isort-check

docker-test-api:
	$(DOCKER_RUN_ARGS) make test-api

docker-pre-pr:
	$(DOCKER_RUN_ARGS) make pre-pr

docker-shell:
	$(DOCKER_RUN_ARGS) bash
