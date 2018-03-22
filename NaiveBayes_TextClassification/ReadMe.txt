0. Make sure python 3 along with numpy is installed

1. Running the code:

python NaiveBayes.py '20newsgroups/vocabulary.txt' '20newsgroups/map.csv' '20newsgroups/train_data.csv' '20newsgroups/train_label.csv' '20newsgroups/test_data.csv' '20newsgroups/test_label.csv'

2. There are a lot of flags which can be set as you wish to customize the output

3. Comments are included and the code is self explanatory.


4. In the predict function the numbers can be customized to predict on a part of the dataset.
   Use 'BE' for bayesian method
   Use 'MLE' for Maximum likelihood method
   Set the last flag to true to show the progress of the prediction on each document in the dataset.

5. In the performance function, the flag can be set to true to view individual prediction for each document