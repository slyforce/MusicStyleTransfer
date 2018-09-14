#!/bin/sh

source venv/bin/activate;

for file in $(find music_style_transfer/); do  
	autopep8 --in-place --aggressive --aggressive $file;
done
