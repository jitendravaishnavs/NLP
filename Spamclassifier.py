import pandas as pd

messages = pd.read_csv('sms_spam_collection/SMSSpamCollection', sep='\t', names=["label", "message"])

#Data cleaning and preprocessing

import re
import nltk

from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer

ps = PorterStemmer()  # for stemming
lemmatizer = WordNetLemmatizer()  # for lemmatization

corpus = []  # list to hold the cleaned sentences
for i in range(0,len(messages)):
    review = re.sub('[^a-zA-Z]', ' ', messages['message'][i])  # remove non-alphabetic characters
    review = review.lower()  # convert to lowercase
    review = review.split()  # split into words
    review = [lemmatizer.lemmatize(word) for word in review if not word in stopwords.words('english')]  # lemmatization and removing stopwords
    review = ' '.join(review)  # join words back into a single string
    corpus.append(review)  # add cleaned sentence to corpus
    
    
#Creating the bag of words model (Document matrix all together)
from sklearn.feature_extraction.text import TfidfVectorizer  # TF-IDF Vectorization
cv = TfidfVectorizer(max_features=5000)  # Limit to 5000 features
X = cv.fit_transform(corpus).toarray() # Create the bag of words model
y = pd.get_dummies(messages['label'])  # Convert labels to dummy variables

y = y.iloc[:, 1].values  # Convert to numpy array for model training

#Train test split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0)

#Training the model using Naive Bayes classifier
from sklearn.naive_bayes import MultinomialNB # Naive Bayes Classifier
spam_detect_model = MultinomialNB().fit(X_train, y_train) # Train the model

y_pred = spam_detect_model.predict(X_test)
print(y_pred)


from sklearn.metrics import confusion_matrix # Confusion Matrix
confusion_m= confusion_matrix(y_test, y_pred)  # Create confusion matrix

# Evaluating the model
from sklearn.metrics import accuracy_score  # Accuracy and Classification Report
accuracy_score(y_test, y_pred) # Print accuracy