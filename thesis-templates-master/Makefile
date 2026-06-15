MAIN=thesis_main

all:
	pdflatex $(MAIN).tex
	bibtex $(MAIN)
	pdflatex $(MAIN).tex
	pdflatex $(MAIN).tex

clean:
	rm -f *.aux $(MAIN).bbl $(MAIN).bcf $(MAIN).blg $(MAIN).log $(MAIN).out $(MAIN).run.xml $(MAIN).toc $(MAIN).lof $(MAIN).lot $(MAIN).glg $(MAIN).glo $(MAIN).gls $(MAIN).ist $(MAIN)-blx.bib
