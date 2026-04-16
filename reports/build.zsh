#!/opt/homebrew/bin/zsh

xelatex bare_jrnl.tex
bibtex bare_jrnl
xelatex bare_jrnl.tex
xelatex bare_jrnl.tex