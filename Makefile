.PHONY: all clean

all: $(DOCUMENTS)

clean:
	rm -f *.aux *.blg *.out *.bbl *.log *.snm *.toc *.nav *.pdf *.ps *.dvi
