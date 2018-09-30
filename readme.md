# Teamlab repo

## Authors
  * Simon Tannert
  * Touhidul Alam
  
## Structure
The repository is structured as follows:
  * code/ contains our implementations for the different tasks in the class. The code is implemented in Python3 (tested on >=3.5)
  * results/ contains the output of our implementations
  * documentation/ will contain our presentation slides and report 
  
code/ is the root of a Python package. There are subdirectories/subpackages for the different implementation tasks.
  * code/lib/reader.py
  * code/lib/feature_extraction.py
  * code/lib/perceptron.py
  * code/lib/naive_bayes.py

To run our programs, you should use the shell scripts provided in the root of our git-repository.

## Dependencies
We use docopt as command line parser, you can install it with the following command

```
python3 -m pip install --user docopt
```
