# Makefile for compiling the SENTRIX IEEE Research Paper

TARGET = main
PDF = $(TARGET).pdf

all: $(PDF)

$(PDF): $(TARGET).tex references.bib IEEEtran.cls
	pdflatex $(TARGET)
	bibtex $(TARGET)
	pdflatex $(TARGET)
	pdflatex $(TARGET)

clean:
	rm -f *.aux *.bbl *.blg *.log *.out *.toc *.synctex.gz

distclean: clean
	rm -f $(PDF)

.PHONY: all clean distclean
