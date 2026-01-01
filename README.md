# twitter-sentiment-analysis
End-to-end Twitter Sentiment Analysis project covering data cleaning, visualization, supervised and unsupervised learning . Includes preprocessing, regression, classification, evaluation metrics, and clustering — all implemented in Python using Google Colab.

# 🧠 Twitter Sentiment Analysis (Week 1–8 Project)

This repository contains my complete **Data Science mid-term project** based on **Twitter Sentiment Analysis**, covering all weekly tasks from Week 1 to Week 8.

---

## 📅 Week-wise Overview

### Week 1: Orientation & Setup
- Installed required packages (Pandas, NumPy, Matplotlib, , Sklearn, NLTK)
- Loaded Twitter dataset and displayed first 10 rows  
📁 `Assignment 1`

### Week 2: Data Cleaning
- Removed duplicates and missing values  
- Cleaned tweets by removing stopwords, punctuation, and applied stemming & lemmatization  
📁 `Assignment 2`

### Week 3: Data Visualization
- Visualized **positive**, **negative**, and **neutral** sentiments using bar plots  
📁 `Assignment 3`

### Week 5: Supervised Learning – Regression
- Applied regression to predict sentiment score probabilities  
📁 `Assignment 5`

### Week 6: Supervised Learning – Classification
- Implemented Logistic Regression and Random Forest for sentiment classification  
📁 `Assignment 6`

### Week 7: Model Evaluation
- Evaluated models using **Precision**, **Recall**, and **F1-score**  
📁 `Assignment 7`

---

## 🧩 Project Description
This project demonstrates how NLP techniques can be integrated with core machine learning concepts — from data cleaning and visualization to regression, classification, and clustering — using a **Twitter Sentiment Dataset**.

---

## 🛠️ Technologies Used
- Python (Google Colab)
- Pandas, NumPy
- Matplotlib, Seaborn
- NLTK, Scikit-learn
- WordCloud

---

## 📈 Results
- Accuracy: 90%
- Best Model: Logistic Regression 
- Key Insight: Majority of tweets express **positive sentiment**.

---

Twitter Sentiment Analysis (Week 9–14 Project)

This repository extends the Twitter Sentiment Analysis project, covering Weeks 9 to 14. It incorporates deep learning techniques, including ANN, RNN (LSTM), NLP, and model deployment.

Week-wise Overview (Week 9-14)
**Week 9: Neural Networks Basics**

Objective: Implemented a basic Artificial Neural Network (ANN) as a baseline model for sentiment classification.

Key Tasks:

Built a feed-forward ANN with Keras.

Preprocessed the data, encoded labels, and split it into training and testing sets.

Trained and evaluated the model using accuracy as the evaluation metric.

Assignment 9

Implemented: Simple ANN model with Keras

Result: Evaluated the model performance using accuracy and loss functions.

Week 10: Advanced Deep Learning

Objective: Advanced the model by applying Recurrent Neural Networks (RNN), specifically Long Short-Term Memory (LSTM), to handle sequential text data.

Key Tasks:

Tokenized tweets and padded sequences.

Built an RNN (LSTM) model to improve sentiment classification.

Trained and evaluated the RNN model.

Assignment 10

Implemented: LSTM for sentiment classification

Result: Achieved better results than the ANN baseline model.

Week 11: Natural Language Processing (NLP)

Objective: Applied NLP techniques, including Word2Vec for word embeddings and stopword removal for text preprocessing.

Key Tasks:

Tokenized text, cleaned tweets, and removed stopwords.

Used Word2Vec to generate word embeddings for sentiment analysis.

Applied tokenization and vectorization techniques for preparing input features.

Assignment 11

Implemented: Word2Vec embeddings for better feature representation of tweets.

Result: Improved understanding of tweet sentiment through semantic word representations.

Week 12: AI in Data Science – Case Studies

Objective: This week involved writing a 2-page report explaining how the sentiment analysis project fits into real-world applications.

Key Tasks:

Discussed the industry applications of sentiment analysis, particularly in areas like marketing and customer service.

Assignment 12

Implemented: Report on the industry applications of sentiment analysis in business contexts.

Result: Gained insight into the practical use of sentiment analysis in industry.

Week 13: Model Deployment

Objective: Deployed the sentiment analysis model as a Streamlit web app, making it available for real-time predictions.

Key Tasks:

Created a simple Streamlit web app for users to input tweets and get sentiment predictions.

Deployed the model locally using Streamlit and ngrok.

Assignment 13

Implemented: Model deployment using Streamlit.

Result: Real-time sentiment analysis application deployed on a web interface.

Week 14: Ethics & Explainability

Objective: Focused on adding model explainability using SHAP to make predictions more transparent and understandable.

Key Tasks:

Used SHAP to visualize model predictions and understand feature importance.

Applied SHAP to explain the predictions made by the deployed model.

Assignment 14

Implemented: Model explainability using SHAP for transparency in predictions.

Result: Made the model more interpretable by showing which features contributed to the sentiment prediction.

Technologies Used (Week 9-14)

Deep Learning: Keras, TensorFlow

NLP: NLTK, Gensim (Word2Vec), Scikit-learn

Model Deployment: Streamlit, ngrok

Explainability: SHAP

Data Processing: Pandas, Numpy

Visualization: Matplotlib, Seaborn

Results (Week 9-14)

Best Model: LSTM (Week 10)

Final Accuracy: 87%

Key Insight: LSTM outperformed ANN by handling the sequential nature of tweets. Deploying the model allowed for real-time sentiment analysis, and SHAP helped us understand model decisions better.

How to Run the Project Locally (Week 9-14)

Clone the repository:

git clone https://github.com/yourusername/twitter-sentiment-analysis.git
cd twitter-sentiment-analysis


Install the required dependencies:

pip install -r requirements.txt


Run the Streamlit app:

streamlit run app.py




