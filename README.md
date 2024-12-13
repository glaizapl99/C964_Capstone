# C964_Capstone
Quick-Start Guide: Customer Sentiment Analyzer
System Requirements
Please ensure that your system meets the following requirements before continuing:
- Python 3.7+
- Flask: backend web application is built using Flask
- Scikit-learn
- Other Python libraries: numpy, pandas, nltk, pickle, etc.
Installation Instructions
1. Download the zip file
2. RECOMMENDED: Create a virtual environment
a. To avoid dependency conflicts, create a virtual environment:
i. python3 -m venv sentiment-env
b. Activate the virtual environment
i. Windows: sentiment-env\Scripts\activate
ii. macOS/Linux: source sentiment-env/bin/activate
c. Install Dependencies
i. Navigate to the folder where repository is located and install required libraries
using:
1. pip install -r requirements.txt
a. NOTE: requirements.txt includes all necessary Python libraries,
such as Flask, scikit-learn, pandas, etc.
3. Prepare Model
a. The model file (sentiment_model.pkl) is pre-trained and stored in the repository. If you
have a new dataset or want to retrain the model:
i. Ensure dataset is in CSV format
ii. Follow the model_training.py and data_preprocessing.py scripts to preprocess
dataset and retrain model
iii. Otherwise, skip this step and proceed with pre-trained model
4. Run the Application
a. Use the following command to start the Flask application:
i. python flask_app.py
ii. This will start the Flask web server on localhost (usually http://127.0.0.1:5000/)
b. Accessing the Web Interface:
i. Open your web browser to the localhost to access the Customer Sentiment
Analyzer web interface
ii. Input a product review in the text field
iii. Click on “Analyze Sentiment” to classify review as either positive or negative
Click on “Analyze Sentiment” to classify review as either positive or negative
1. Example Input:
a. Review: “This product is amazing, I loved it!”
b. Sentiment: Positive
2. Example Input:
a. Review: “This product is terrible, do not buy this.”
b. Sentiment: Negative
5. Stopping the Application
a. To stop, press Ctrl + C in the terminal/command prompt where the Flask app is running
