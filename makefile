override TARGET:=README

all: build

split:
	scripts/splitByEntry.sh $(TARGET).md

build:
	scripts/buildDocument.sh $(TARGET).md

buildPDF: build
	pandoc -V geometry:margin=2cm -s -o $(TARGET).pdf $(TARGET).md