import pickle
from imblearn.under_sampling import RandomUnderSampler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder

from app.flask_app import vectorizer


def train_model(x, y):
    #undersample majority class to balance dataset
    rus = RandomUnderSampler(random_state=42)
    x_resampled, y_resampled = rus.fit_resample(x, y)

    #split resampled data into train and test sets
    x_train, x_test, y_train, y_test = train_test_split(x_resampled, y_resampled, test_size=0.2, random_state=42)

    #encode labels as numbers
    label_encoder = LabelEncoder()
    y_train_encoded = label_encoder.fit_transform(y_train)
    y_test_encoded = label_encoder.transform(y_test)

    #train Logistic Regression model
    model = LogisticRegression(max_iter=1000, class_weight='balanced')
    model.fit(x_train, y_train_encoded)

    #evaluate
    y_pred = model.predict(x_test)
    print("Classification Report:")
    print(classification_report(y_test_encoded, y_pred))

    #save label encoder to transform predictions back to string labels
    with open('app/label_encoder.pkl', 'wb') as encoder_file:
        pickle.dump(label_encoder, encoder_file)

    #save vectorizer for later use
    with open('app/vectorizer.pkl', 'wb') as vectorizer_file:
        pickle.dump(vectorizer, vectorizer_file)

    print("Label Encoder and Vectorizer saved.")

    return model, label_encoder, vectorizer
