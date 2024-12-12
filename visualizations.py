from matplotlib import pyplot as plt
import seaborn as sns


def plot_sentiment_distribution(data):
    sns.countplot(x='predicted_sentiment', data=data)
    plt.title('Sentiment Distribution')
    plt.show()


