# Natural Processing Language

# Importing the Libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the Dataset
dataset = pd.read_csv("Restaurant_Reviews.tsv", delimiter = "\t", quoting = 3)

# Cleaning the texts
import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
corpus = []
for i in range(0, 1000):
    review = re.sub('[^A-Za-z]', ' ', dataset['Review'][i])
    review = review.lower()
    review = review.split()
    ps = PorterStemmer()
    review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
    review = ' '.join(review)
    corpus.append(review)

# Creating the Bag of Words
"""from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features=1500)
X = cv.fit_transform(corpus).toarray()
y = dataset.iloc[:,1].values"""

from sklearn.feature_extraction.text import  TfidfVectorizer
tfid = TfidfVectorizer()
X = tfid.fit_transform(corpus).toarray()
y = np.array(dataset.iloc[:,1])

# Splitting the Dataset into Training and Test Dataset
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2,  random_state = 0)

# Fitting the Model to the Training set
"""from sklearn.naive_bayes import MultinomialNB
classifier = MultinomialNB()
classifier.fit(X_train, y_train)"""

from sklearn.model_selection import  RandomizedSearchCV
parameters = {'n_estimators': [100,200,400,600,80,1000,1200,1400,1600], 
               'max_features' : ['auto', 'sqrt', 'log2', None],
               'min_samples_split' : [2, 5, 10],
               'min_samples_leaf' : [1,2,4],
               'bootstrap' : [True,False],
               'criterion' : ["gini", "entropy"]
               }


from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(random_state = 0)

Random_searchCV = RandomizedSearchCV(estimator = classifier, 
                           param_distributions = parameters, 
                           n_iter = 100,
                           cv = 10, 
                           n_jobs = -1)
Random_searchCV = Random_searchCV.fit(X_train, y_train)
best_accuracy = Random_searchCV.best_score_
best_parameters = Random_searchCV.best_params_



# Predicting the test set
y_pred = classifier.predict(X_test)

# Making the confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

from sklearn.metrics import accuracy_score
ac = accuracy_score(y_test, y_pred)

from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 100)
accuracies.mean()
accuracies.std()


             


