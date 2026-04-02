import joblib
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_20newsgroups

# Load dataset
news = fetch_20newsgroups(subset='all')
X = news.data
y = news.target

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a CountVectorizer to transform the text data into feature vectors
vectorizer = CountVectorizer()
X_train_vectorized = vectorizer.fit_transform(X_train)

# Train a Multinomial Naive Bayes model
model = MultinomialNB()
model.fit(X_train_vectorized, y_train)

# Save the trained model and vectorizer using joblib
joblib.dump(model, 'sentiment_model.joblib')
joblib.dump(vectorizer, 'vectorizer.joblib')
