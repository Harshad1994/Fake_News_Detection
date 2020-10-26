# Fake_News_Detection
Here I have built a  ML model for detecting the fake news articles.

Approach.

1. The news articles are preprocessed to remove any unwanted punctuations and special characters.

2. The text is converted into a term-frequency inverse document frequency [TF-IDF] matrix using sklearn packages

3. A SVM classifier is trained on the data. THe SVM found to perform better.

4 THe model gives 98%  accuracy on validation data. 

           clas   precision    recall  f1-score

           0       0.97      0.98      0.97   
           1       0.97      0.96      0.97



4. A many to one seq2seq model can also be trained using RNN with LSTM blocks..





