TEX = pdflatex
BIB = bibtex
TARGET = main

.PHONY: all clean

all: $(TARGET).pdf

$(TARGET).pdf: $(TARGET).tex semml_report.sty references.bib figures/training_pipeline.tex figures/sft_prompt_flow.tex figures/grpo_prompt_flow.tex figures/ldct_example.png figures/sft_selected_scatter.png figures/grpo_selected_scatter.png
	$(TEX) $(TARGET).tex
	$(BIB) $(TARGET)
	$(TEX) $(TARGET).tex
	$(TEX) $(TARGET).tex

clean:
	rm -f $(TARGET).aux $(TARGET).bbl $(TARGET).blg $(TARGET).log $(TARGET).out $(TARGET).toc $(TARGET).pdf
