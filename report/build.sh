#!/usr/bin/env sh

set -xe

if ! podman image exists texlive; then
    echo "Image 'texlive' not found. Building from https://github.com/denisstrizhkin/texlive-container.git ..."
    podman build -t texlive https://github.com/denisstrizhkin/texlive-container.git
fi

podman run --rm -v "$(pwd):/work:Z" -w /work texlive pdflatex report.tex