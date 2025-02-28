all: build

split:
	scripts/splitByEntry.sh README.md

build:
  scripts/buildDocument.sh README.md
