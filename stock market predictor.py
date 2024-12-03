import praw
import pandas as pd
import pytz
from datetime import datetime
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import yfinance as yf
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import numpy as np
import time

# Replace with your Reddit API credentials
reddit = praw.Reddit(client_id="jPFfMuLhQPh7JUYLRf8nTg",
            client_secret="4j5iUelahMidPOGRljxRMWt0hEWcug",
                            user_agent="sentiment_analysis")

# Define the subreddit and search query for stock-related posts
subreddit_name = 'stocks'  # You can change this to a specific subreddit like "StockMarket"
limit = 100  # Number of posts to scrape
n = int(input("enter the number of stocks whose prediction you want "))


# Function to fetch data from Reddit
def fetch_reddit_data(subreddit_name, search_query, limit):
    # Get the subreddit
    subreddit = reddit.subreddit(subreddit_name)

    posts_data = []

    for post in subreddit.search(search_query, limit=limit, sort='new'):
        posts_data.append({
            "title": post.title,
            "text": post.selftext,
            "upvotes": post.score,
            "comments": post.num_comments,
            "date": datetime.fromtimestamp(post.created_utc, pytz.UTC),
        })

    return posts_data


# Function to analyze sentiment using TextBlob
def analyze_sentiment_textblob(text):
    blob = TextBlob(text)
    return blob.sentiment.polarity  # returns sentiment polarity between -1 (negative) and 1 (positive)


# Function to analyze sentiment using VADER
def analyze_sentiment_vader(text):
    sentiment = analyzer.polarity_scores(text)
    return sentiment['compound']  # returns sentiment score


# Initialize VADER sentiment analyzer
analyzer = SentimentIntensityAnalyzer()


# Function to fetch  stock data
def fetch_stock_data():
    stock = yf.Ticker('TSLA')
    stock_data = stock.history(period="1d", interval="1m")  # Real-time data
    return stock_data

# Real-time Prediction: Fetch the latest sentiment data and make predictions
def real_time_prediction():
    # Get the latest posts and calculate the sentiment
    latest_posts = fetch_reddit_data(subreddit_name, search_query, limit=10)
    latest_sentiments = [analyze_sentiment_textblob(post['title'] + " " + post['text']) for post in latest_posts]

    # Average sentiment from the latest posts
    avg_sentiment = np.mean(latest_sentiments)

    # Predict stock movement (up or down) using the trained model
    input_features = pd.DataFrame([[avg_sentiment, avg_sentiment]], columns=['sentiment_textblob', 'sentiment_vader'])
    prediction = model.predict(input_features)  # Using both sentiment columns

    # Print the prediction result
    if prediction == 1:
        print("Predicted:",search_query," stock will go UP.")
    else:
        print("Predicted:",search_query," stock will go DOWN.")


# Process the data and analyze sentiment
posts_data = []
while n>0:
    search_query = input("Enter the stock name  ")# You can change this to any stock name
    # Fetch stock data
    stock_data = fetch_stock_data()

    # Feature Engineering: Create a binary target column for stock direction
    # 1: Stock price goes up, 0: Stock price goes down
    stock_data['target'] = np.where(stock_data['Close'].pct_change() > 0, 1,
                                    0)  # 1 means stock goes up, 0 means stock goes down

    # Start the process of extracting Reddit data
    print("Fetching Reddit data...")
    posts = fetch_reddit_data(subreddit_name, search_query, limit)
    for post in posts:
        try:
            sentiment_textblob = analyze_sentiment_textblob(post['title'] + " " + post['text'])
            sentiment_vader = analyze_sentiment_vader(post['title'] + " " + post['text'])
            posts_data.append({
                "sentiment_textblob": sentiment_textblob,
                "sentiment_vader": sentiment_vader
            })
        except Exception as e:
            print(f"Error analyzing post: {e}")

    # Store the data in a DataFrame and save it to a CSV
    if not posts_data:
        print("No data available for analysis.")
        exit()



    df = pd.DataFrame(posts_data)
    # print("DataFrame created successfully:")
    # print(df.head())

    # Use sentiment scores as features for model training
    X = df[['sentiment_textblob', 'sentiment_vader']]  # Sentiment scores
    y = stock_data['target'][:len(df)]  # Target: stock price up or down (adjust length)

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train the Logistic Regression model
    model = LogisticRegression()
    model.fit(X_train, y_train)

    # Make predictions on the test set
    y_pred = model.predict(X_test)

    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model Accuracy: {accuracy * 100:.2f}%")
    print("\nFetching real-time stock sentiment prediction...")
    real_time_prediction()# Run the real-time prediction function every minute
    time.sleep(1)  # Sleep for 10 seconds before checking again
    n-=1