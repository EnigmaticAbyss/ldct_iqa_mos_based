TEX = pdflatex
BIB = bibtex
TARGET = main

.PHONY: all clean

all: $(TARGET).pdf

$(TARGET).pdf: $(TARGET).tex semml_report.sty references.bib
	$(TEX) $(TARGET).tex
	$(BIB) $(TARGET)
	$(TEX) $(TARGET).tex
	$(TEX) $(TARGET).tex

clean:
	rm -f $(TARGET).aux $(TARGET).bbl $(TARGET).blg $(TARGET).log $(TARGET).out $(TARGET).toc $(TARGET).pdf
