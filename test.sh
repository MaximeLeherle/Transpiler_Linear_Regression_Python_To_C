#!/bin/sh

python3 -m venv venv
venv/bin/activate

pip3 install -r requirement.txt

echo -e "Start the python file\n\n\n"
python3 src/transpiler.py

echo -e "\n\n\nCompile and execute the .c file"
gcc *.c
./a.out
