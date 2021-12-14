# Transpiler_Linear_Regression_Python_To_C

## Context

The goal is to have a python file that can take a linear regression model in a .joblib file and translate it into a C file.

## File

### Python

We have the file src/transpiler.py.
This is the main file it will load the model and create the .c file.

There are 3 possibilities :
    -    - Either a file is specified and so we will load this file and use it for the c code
    - Or no file is specified and in this case we use the file of california of skleran. We create it and save it then we use it for the c code
    - And if the file california.joblib is already existing and no file is specified then we load it and use it.

### Sh

The test.sh file is there to launch the python script without argument so on the california.joblib file and then it will launch the C file created.

## Use

Without using the test.sh file the usage is like this:

For the python script there are 2 options:

    --filename <str> : to use the str file, it must be a .joblib
    --out <str> name of the outfile, that is to be in .c, and prediction.c

For the c file just compile with `gcc *.c` and run the code with `/a.out`
