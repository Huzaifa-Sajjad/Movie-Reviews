import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn import metrics


#1 Read the dataaset into  a dataframe
df = pd.read_csv("./moviereviews2.tsv", sep="\t")
#removing empty values
df.dropna(inplace=True)
blanks = []
for i,lb,rv in df.itertuples():
    if rv.isspace():
        blanks.append(i)
df.drop(blanks, inplace=True)

#2 Split the data into test and train sets
X = df["review"]
y = df["label"]
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.33,random_state=42)

#3 Vectorize , Tfidf and train the model
#3.1 Create a pipeline
classifier = Pipeline([("tfidf",TfidfVectorizer()), ("svc",LinearSVC())])
#3.2 Fiting the classifier
classifier.fit(X_train,y_train)

#4 Evaluation
predictions = classifier.predict(X_test)
print(metrics.classification_report(y_test,predictions))
print(metrics.accuracy_score(y_test,predictions))


#Deployment
review = ["This move is fantastic, Worth watch.","This move is not watchable."]
result = classifier.predict(review)
print(result)