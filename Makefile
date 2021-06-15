.PHONY: test

test:
	nosetests -v -s birdman_jr --with-coverage --cover-package=birdman_jr
