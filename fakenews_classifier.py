# -*- coding: utf-8 -*-


#import requisite Libraries

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer


from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn import metrics

# import nltk
# import re
# from nltk.corpus import stopwords
# from nltk.stem.porter import PorterStemmer as ps
# nltk.download('stopwords')


def run():

  df = pd.read_csv('train.csv')
  df = df.dropna()
  df = df.drop(['id', 'author','title'], axis = 1)

  #Performing Data preprocessing
  corpus = []

  for text in df['text']:
    #text =  re.sub('[^a-zA-Z]',' ',text)
    text=text.lower()
    #tokens=text.split()
    #tokens=[ps().stem(word) for word in tokens if not word in stopwords.words('english')]
    #text=' '.join(tokens)
    corpus.append(text)


  # transforming text data into a tf-idf matrix
  tfTransformer = TfidfTransformer(smooth_idf=False)
  count_vectorizer = CountVectorizer(ngram_range=(1, 2))
  counts = count_vectorizer.fit_transform(corpus)
  tfidf = tfTransformer.fit_transform(counts)


  targets = df['label'].values

  #split samples
  X_train, X_test, y_train, y_test = train_test_split(tfidf, targets, random_state=0)


  #fitting a Logistic Regression Model
  classifier1 = LogisticRegression()

  classifier1.fit(X_train, y_train)


  #Fitting a SVM model
  classifier2 = SVC(C = 1.0, kernel='linear')
  classifier2.fit(X_train, y_train)



  y_pred = classifier2.predict(X_test)


  cm = metrics.confusion_matrix(y_test, y_pred)

  print(metrics.classification_report(y_test,y_pred))



  #inference
  test_df = pd.read_csv('test.csv')

  test_df = test_df.fillna(' ')

  test_counts = count_vectorizer.transform(test_df['text'].values)

  test_tfidf = tfTransformer.fit_transform(test_counts)


  out = classifier.predict(test_tfidf)

  test_df['label'] = out

  test_df = test_df.drop(['title', 'author','text'], axis = 1)

  test_df.to_csv('submit.csv',index=False)


if __name__ == "__main__":

  run()

