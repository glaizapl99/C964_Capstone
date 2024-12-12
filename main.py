from data_preprocessing import load_data, clean_data, vectorize_data
from model_training import train_model
from visualizations import plot_sentiment_distribution
import pandas as pd

def predict_sentiment(sample_reviews, vectorizer, model, label_encoder):
    review_vectorized = vectorizer.transform(sample_reviews).toarray()
    predictions = model.predict(review_vectorized)
    return [
        (review, label_encoder.inverse_transform([sentiment])[0])
        for review, sentiment in zip(sample_reviews, predictions)
    ]

def main():
    #load data
    file_path = 'data/amazon.csv'
    data = load_data(file_path)

    #ensure rating is numeric
    data['rating'] = pd.to_numeric(data['rating'], errors='coerce')

    data['sentiment'] = data['rating'].apply(lambda rating: 'positive' if rating >=4 else 'negative')
    print("Sentiment Distribution:")
    print(data['sentiment'].value_counts())

    #clean data
    data = clean_data(data)

    # extract cleaned text for vectorization
    x = data['cleaned_text']
    y = data['sentiment']

    #vectorize cleaned data
    x_vectorized, vectorizer = vectorize_data(data)

    #train
    model, label_encoder, vectorizer= train_model(x_vectorized, y)


    #make predictions based on review text
    review_texts = data['cleaned_text']
    review_vectorized = vectorizer.transform(review_texts).toarray()
    predictions = model.predict(review_vectorized)
    #assign predicted sentiment labels to a new column
    data['predicted_sentiment'] = label_encoder.inverse_transform(predictions)

    #print sentiment distribution for verification
    print("Sentiment Distribution (Predicted):")
    print(data['predicted_sentiment'].value_counts())

    #visualize results
    plot_sentiment_distribution(data)

    # Test sample reviews
    sample_reviews = [
        "This is terrible!",
        "I absolutely hated this product.",
        "Not worth the money.",
        "This is amazing!",
        "I love it!"
    ]
    sample_predictions = predict_sentiment(sample_reviews, vectorizer, model, label_encoder)
    print("\nSample Predictions:")
    for review, sentiment in sample_predictions:
        print(f"Review: {review} | Predicted Sentiment: {sentiment}")


if __name__ == "__main__":
    main()

