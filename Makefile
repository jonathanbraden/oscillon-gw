prefix=notes
suffixes=.toc .aux .out .bbl .blg

oscillons: $(prefix).tex #$(prefix).bib
	pdflatex $(prefix).tex
#	bibtex $(prefix)
	pdflatex $(prefix).tex
	pdflatex $(prefix).tex

submit: $(prefix).tex
	tar -cvf $(prefix).tar figures/*.pdf

clean:
	rm -f $(prefix).log $(prefix).toc $(prefix).aux $(prefix).out $(prefix).blg $(prefix).end \#*.*~ Makefile~ *~

purge:
	make clean
	rm -f $(prefix).bbl $(prefix).pdf

.PHONY: clean, purge
