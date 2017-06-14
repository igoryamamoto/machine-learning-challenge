# Machine Learning Challenge - Chaordic

Machine Learning [Challenge](https://chaordic.github.io/machinelearning-challenge/) from [Chaordic](https://chaordic.com.br/)

---
## Steps to solve the challenge

1. **Gathering information:** research similar problems and published articles

2. **Knowing the data:** play with the datasets and see how it is structured

3. **Cleaning the data:** split the dataset and remove irrelevant information

4. **Analysing the data:** find correlations between possible features and the target data

5. **Training the model:** split the data for train and test and try different models of classifiers

6. **Evaluating the model:** compare the results with an appropriate metric for a binary classification

7. **Submitting the answer:** generate csv file with the target data scored with the best trained model

8. **Iterating over the process:** go back to step 4, filter the relevant features, remove outliers, balance the trainig data, combine some variables, do some magic, ...

---
## Tools used during the challenge

1. **Gathering information:** google

2. **Knowing the data:** gedit and bash commands

3. **Cleaning the data:** bash scripts and OpenRefine

4. **Analysing the data:** jupyter notebooks and python libraries (pandas, numpy, matplotlib, ...)

5. **Training the model:** sklearn modules

6. **Evaluating the model:** sklearn functions and methods inside classes

7. **Submitting the answer:** python code

8. **Iterating over the process:** try different tools like azure machine learning studio

---
## Description of project files

* data/ : directory to store the datasets

    * data/split_data.sh : bash script to split dataset
        * e.g.:$./split_data.sh data

    * data/stats.sh : bash script to count masculine/feminine gender 

        * e.g.:$./stats.sh data

    * data/test* : file with samples of the datasets' content

    * data/users : all target users ids

* research/ : directory with articles related to the challenge

* playground.ipynb : notebook used to learn how to use python packages (pandas, sklearn, ...)

* Igor_final.ipynb : notebook with the workflow of the solution.

* final_answer.csv : file containing challenge's answer
