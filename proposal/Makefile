build_pdf:
	# pandoc proposal.md -f markdown --template=../template.tex -o proposal.pdf
	pandoc proposal.md -f markdown --latex-engine=xelatex --filter pandoc-citeproc \
	 --bibliography=proposal.bib --template=../template.tex -o proposal.pdf \
	 --variable colorlinks --toc --variable toc-own-page
	# /usr/share/texlive/texmf-dist/tex/latex/
	# pandoc proposal.md -f markdown --pdf-engine=xelatex --template=../template.tex -o proposal.pdf
