#!/opt/homebrew/bin/zsh

xelatex slides.tex
biber slides
xelatex slides.tex
xelatex slides.tex