# Advanced Computational Physics Lab course Hauke Uhde
This reposetory contais the code and results for the Advanced computational physics lab course SoSe 2025.
It is structured such that eah Project has its own directory in which one finds the python scripts for each Task and the results of the computations. If needed the Analytical work is also provided.
The scripts for the single Tasks and probably also for the different projets are not independant of each other as some functions are generally important. As an example: in Project1/Task1.py there is a generalized plotting function that is used in Task2.py as well. If needed it will also be used in other Projects though might it be through copy and paste.

## Actual scripts
In the scripts at the bottom there is the usefull if __name__=="__main__" part that allows for the scripts to contain functions used throughout other scripts. When running the script the whole Task is run by default though in the mentioned part that can be toggled through commenting or even through task "comprehension" as can be seen in Task2.py. This allows for independant checking and running of the Tasks as if it was a Notebook.
In the functions named after each Task all relevant parameters for that Task are present and may be changed at will.

## The reports
For each project a mendatory report is given in the directory of the project such that it all is at place.