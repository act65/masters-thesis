build_html:
	# the html files
	pandoc -f markdown -t html5 --standalone -o ../docs/transfer.html -c ../style.css -H ../docs/mathjax-header.html --mathjax=../docs/mathjax.js transfer.md
	pandoc -f markdown -t html5 --standalone -o ../docs/model-based.html -c ../style.css -H ../docs/mathjax-header.html --mathjax=../docs/mathjax.js model-based.md
	pandoc -f markdown -t html5 --standalone -o ../docs/actor-critic.html -c ../style.css -H ../docs/mathjax-header.html --mathjax=../docs/mathjax.js actor-critics.md
	pandoc -f markdown -t html5 --standalone -o ../docs/planning.html -c ../style.css -H ../docs/mathjax-header.html --mathjax=../docs/mathjax.js planning.md
	pandoc -f markdown -t html5 --standalone -o ../docs/search.html -c ../style.css -H ../docs/mathjax-header.html --mathjax=../docs/mathjax.js search.md
	pandoc -f markdown -t html5 --standalone -o ../docs/structured-models.html -c ../style.css -H ../docs/mathjax-header.html --mathjax=../docs/mathjax.js structured-models.md
	pandoc -f markdown -t html5 --standalone -o ../docs/exploitable-models.html -c ../style.css -H ../docs/mathjax-header.html --mathjax=../docs/mathjax.js exploitable-models.md


# the latex files
# pandoc -o transfer.pdf transfer.md
