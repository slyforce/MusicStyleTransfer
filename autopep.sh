#!/bin/sh

source venv/bin/activate;

for file in $(find src); do  
	autopep8 --in-place --aggressive --aggressive $file;
done
