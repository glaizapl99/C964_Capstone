import os
from flask import Flask, render_template, request
import pickle
import logging

from matplotlib import pyplot as plt
import io
import base64

#initialize flask
app = Flask(__name__)

#configure logging
logging.basicConfig(
    filename='app.log',
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

#load trained model and vectorizer
#define absolute paths
base_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(base_dir, 'sentiment_model.pkl')
vectorizer_path = os.path.join(base_dir, 'vectorizer.pkl')
label_encoder_path = os.path.join(base_dir, 'label_encoder.pkl')

# Load model and vectorizer
model = pickle.load(open(model_path, 'rb'))
vectorizer = pickle.load(open(vectorizer_path, 'rb'))
label_encoder = pickle.load(open(label_encoder_path, 'rb'))

#route for home page
@app.route('/')
def home():
    return render_template('index.html')

#route to handle review form and predict sentiment
@app.route('/predict', methods = ['POST'])
def predict():
    #get review text from form input
    review_text = request.form['review_text']

    #clean the review_text before passing to vectorizer
    review_text_cleaned = review_text.lower() #convert to lowercase
    review_text_cleaned = ''.join(e for e in review_text_cleaned if e.isalnum() or e.isspace())  #remove special characters


    #vectorize cleaned review text
    review_vectorized = vectorizer.transform([review_text_cleaned]).toarray()

    #predict sentiment
    prediction = model.predict(review_vectorized)

    if isinstance(prediction[0], (int, float)):
        sentiment = label_encoder.inverse_transform(prediction)[0]
    else:
        sentiment = prediction[0]

    #log user input and prediction
    app.logger.info(f"User Input: '{review_text_cleaned}' | Predicted Sentiment: '{sentiment}'")

    print(f"Raw Prediction: {prediction}")  # Outputs what the model predicts
    print(f"Decoded Sentiment: {sentiment}")  # Outputs the sentiment after decoding

    #return result to html page
    return render_template('index.html', prediction_text=f"Sentiment: {sentiment}")

# Function to generate a sentiment bar chart
def generate_bar_chart():
    sentiment_data = {'positive': 863, 'negative': 602}
    plt.bar(sentiment_data.keys(), sentiment_data.values(), color='steelblue')
    plt.title('Sentiment Distribution')
    plt.xlabel('Predicted Sentiment')
    plt.ylabel('Count')
    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    chart_url = base64.b64encode(img.getvalue()).decode()
    plt.close()
    return chart_url

# Route for sentiment chart
@app.route('/sentiment-chart')
def sentiment_chart():
    chart_url = generate_bar_chart()
    return render_template('sentiment_chart.html', chart_url=chart_url)

# Route for classification report
@app.route('/classification-report')
def classification_report():
    report = {
        'positive': {'precision': 0.80, 'recall': 0.83, 'f1-score': 0.81, 'support': 86},
        'negative': {'precision': 0.72, 'recall': 0.68, 'f1-score': 0.70, 'support': 56},
        'accuracy': 0.77
    }
    return render_template('classification_report.html', report=report)


if __name__ == '__main__':
    app.run(debug=True)
