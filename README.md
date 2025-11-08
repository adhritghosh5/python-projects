# 🧠 Stock Sentiment Analysis Using Reddit + Machine Learning  

### 📈 Predicting Stock Movements Using Social Media Sentiment  

This project leverages **Natural Language Processing (NLP)** and **Machine Learning (ML)** to analyze Reddit discussions related to various stocks and predict whether their prices are likely to go **UP** or **DOWN**.  
It combines **real-time Reddit data extraction**, **dual sentiment analysis (TextBlob + VADER)**, and **financial data correlation** using **Yahoo Finance** to model market sentiment and its potential effect on short-term stock trends.

---

## 🚀 Project Overview  

Financial markets are increasingly influenced by **social sentiment** — what investors say online often drives price movements.  
This project extracts real-time stock-related discussions from **Reddit** using the `PRAW` API, analyzes their sentiment using **TextBlob** and **VADER**, and correlates that with **real-time stock data** fetched from **Yahoo Finance** (`yfinance`).  

A **Logistic Regression** model is trained to predict stock direction (up/down) based on the analyzed sentiment data.  
It showcases how **machine learning + NLP** can provide actionable insights for **quantitative trading**, **financial forecasting**, and **AI-driven research**.

---

## 🧩 Tech Stack  

| Category | Tools / Libraries |
|-----------|------------------|
| **Language** | Python 3.8+ |
| **Data Source** | Reddit API via `PRAW`, Yahoo Finance via `yfinance` |
| **NLP Libraries** | TextBlob, VADER Sentiment |
| **Machine Learning** | scikit-learn (Logistic Regression) |
| **Data Handling** | pandas, numpy |
| **Utilities** | pytz, datetime, time |

---

## ⚙️ How the Code Works  

### **1. Reddit Data Extraction (via PRAW)**  

The code connects to Reddit through **PRAW (Python Reddit API Wrapper)** and extracts recent posts about a given stock from a specified subreddit (default: `r/stocks`).

Each Reddit post contains:
- **Title**
- **Body text**
- **Number of upvotes**
- **Number of comments**
- **Timestamp (UTC)**

The fetched posts are stored in a structured list for further sentiment processing.

```python
# Initialize Reddit API connection
reddit = praw.Reddit(
    client_id="YOUR_CLIENT_ID",
    client_secret="YOUR_CLIENT_SECRET",
    user_agent="YOUR_USER_AGENT"
)

# Function to fetch Reddit posts
def fetch_reddit_data(subreddit_name, search_query, limit=100):
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
```
2. Sentiment Analysis (TextBlob + VADER)

Each Reddit post undergoes dual sentiment analysis using two complementary NLP approaches:

TextBlob – A lexicon-based approach that computes polarity between -1 (negative) and +1 (positive).

VADER (Valence Aware Dictionary and sEntiment Reasoner) – A model fine-tuned for social media text, giving a compound score between -1 and +1.

Both sentiment scores are later used as features for the ML model.

```python
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# Initialize VADER analyzer
analyzer = SentimentIntensityAnalyzer()

# Function for TextBlob sentiment
def analyze_sentiment_textblob(text):
    blob = TextBlob(text)
    return blob.sentiment.polarity  # returns value between -1 to +1

# Function for VADER sentiment
def analyze_sentiment_vader(text):
    sentiment = analyzer.polarity_scores(text)
    return sentiment['compound']  # compound sentiment score

# Apply sentiment analysis to fetched posts
posts_data = []
for post in reddit_data:
    sentiment_textblob = analyze_sentiment_textblob(post['title'] + " " + post['text'])
    sentiment_vader = analyze_sentiment_vader(post['title'] + " " + post['text'])
    posts_data.append({
        "sentiment_textblob": sentiment_textblob,
        "sentiment_vader": sentiment_vader
    })
```

3. Stock Data Collection (via Yahoo Finance)

Real-time stock data is fetched from Yahoo Finance using yfinance.
It retrieves one-day price data at 1-minute intervals and labels each minute’s price movement as:

1 → Stock price went up

0 → Stock price went down

```python
import yfinance as yf
import numpy as np

def fetch_stock_data(ticker_symbol):
    stock = yf.Ticker(ticker_symbol)
    stock_data = stock.history(period="1d", interval="1m")
    stock_data['target'] = np.where(stock_data['Close'].pct_change() > 0, 1, 0)
    return stock_data
```
4. Model Training (Logistic Regression)

A Logistic Regression model is trained using sentiment scores as features (TextBlob and VADER)
and stock movement (target) as the label.

```python
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Prepare feature and target data
X = df[['sentiment_textblob', 'sentiment_vader']]
y = stock_data['target'][:len(df)]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = LogisticRegression()
model.fit(X_train, y_train)

# Evaluate accuracy
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy * 100:.2f}%")
```
5. Real-Time Prediction

Once trained, the model uses new Reddit posts to make real-time predictions.
It calculates the average sentiment of the latest posts and predicts the stock's next likely movement.

```python
def real_time_prediction(stock_name):
    latest_posts = fetch_reddit_data('stocks', stock_name, limit=10)
    latest_sentiments = [
        analyze_sentiment_textblob(post['title'] + " " + post['text'])
        for post in latest_posts
    ]
    avg_sentiment = np.mean(latest_sentiments)
    
    input_features = pd.DataFrame(
        [[avg_sentiment, avg_sentiment]],
        columns=['sentiment_textblob', 'sentiment_vader']
    )
    
    prediction = model.predict(input_features)
    if prediction == 1:
        print(f"Predicted: {stock_name} stock will go UP.")
    else:
        print(f"Predicted: {stock_name} stock will go DOWN.")
```

Example Output 

```python
enter the number of stocks whose prediction you want 3
Enter the stock name  Apple
Fetching Reddit data...
Model Accuracy: 35.00%

Fetching real-time stock sentiment prediction...
Predicted: Apple  stock will go DOWN.
Enter the stock name  Nvidia
Fetching Reddit data...
Model Accuracy: 40.00%

Fetching real-time stock sentiment prediction...
Predicted: Nvidia  stock will go UP.
Enter the stock name  Palantir
Fetching Reddit data...
Model Accuracy: 45.00%

Fetching real-time stock sentiment prediction...
Predicted: Palantir  stock will go UP.
```

🧠 Concepts Demonstrated

✅ API Integration – Fetching live Reddit and Yahoo Finance data.
✅ Natural Language Processing – Dual-model sentiment analysis (TextBlob + VADER).
✅ Machine Learning – Logistic Regression for stock movement prediction.
✅ Data Engineering – Feature extraction and labeling for supervised learning.
✅ Real-Time Inference – Automated sentiment-driven stock prediction.


