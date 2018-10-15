build_html:
	# the html files
	# cd proposals
	pandoc -f markdown -t html5 --standalone -o ../docs/transfer.html -c ../style.css -H ../docs/mathjax-header.html --mathjax=../docs/mathjax.js transfer.md
	pandoc -f markdown -t html5 --standalone -o ../docs/model-based-rl.html -c ../style.css -H ../docs/mathjax-header.html --mathjax=../docs/mathjax.js model-based-rl.md

# the latex files
# pandoc -o transfer.pdf transfer.md
