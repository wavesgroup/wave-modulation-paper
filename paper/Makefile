.PHONY: all clean

all: paper

paper: paper.tex references.bib
	pdflatex -halt-on-error $@
	bibtex $@
	pdflatex -halt-on-error $@
	pdflatex -halt-on-error $@

clean:
	$(RM) *.aux *.bbl *.blg *.cut *fdb_latexmk *.fls *.log *.out *.pdf figures/*.png
