import pandas as pd
import zipfile
import requests
import io
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/00228/smsspamcollection.zip'

response = requests.get(url)
zip_file = zipfile.ZipFile(io.BytesIO(response.content))


file_names = zip_file.namelist()
print("Files in the ZIP:", file_names)


with zip_file.open('SMSSpamCollection') as f:
    data = pd.read_csv(f, encoding='latin-1', sep='\t', header=None)

data.columns = ['Label', 'EmailText']
data['Label'] = data['Label'].map({'spam': 1, 'ham': 0})


X = data['EmailText']
y = data['Label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

vectorizer = CountVectorizer()
X_train_vectorized = vectorizer.fit_transform(X_train)
X_test_vectorized = vectorizer.transform(X_test)

model = MultinomialNB()
model.fit(X_train_vectorized, y_train)

y_pred = model.predict(X_test_vectorized)

accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")
print("Classification Report:")
print(classification_report(y_test, y_pred))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Function to predict if a new email is spam or not
def predict_spam(email):
    email_vectorized = vectorizer.transform([email])
    prediction = model.predict(email_vectorized)
    return "Spam" if prediction[0] == 1 else "Not Spam"

user_email = input("Please enter the email text you want to classify: ")
print(f"New Email Prediction: {predict_spam(user_email)}")
