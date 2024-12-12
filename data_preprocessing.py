import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle

def load_data(file_path):
    data = pd.read_csv(file_path)
    print("Columns: in dataset: ", data.columns)
    return data

def clean_data(data):
    data = data.dropna(subset=['review_content'])
    data['cleaned_text'] = data['review_content'].str.lower().str.replace(r'[^\w\s]', '', regex=True)
    return data

def vectorize_data(data):
    vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 3), stop_words='english')
    x = vectorizer.fit_transform(data['cleaned_text']).toarray()

    with open('app/vectorizer.pkl', 'wb') as vectorizer_file:
        pickle.dump(vectorizer, vectorizer_file)

    print(f"TF-IDF feature matrix shape: {x.shape}")

    return x, vectorizer

def transform_with_vectorizer(vectorizer, reviews):
    return vectorizer.transform(reviews).toarray()
