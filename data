import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from collections import Counter
from wordcloud import WordCloud
import numpy as np

# Load the model
model = joblib.load('model.pkl')

# Load the vectorizer
vectorizer = joblib.load('vectorizer.pkl')

# Define function to preprocess text
def preprocess_text(text):
    # Implement your preprocessing logic here
    return text

# Define function to predict sentiment
def predict_sentiment(text):
    # Preprocess text
    preprocessed_text = preprocess_text(text)
    # Vectorize preprocessed text
    vectorized_text = vectorizer.transform([preprocessed_text])
    # Predict sentiment
    sentiment = model.predict(vectorized_text)[0]
    return sentiment

# Streamlit UI
def main():
    st.title('Sentiment Analysis App')

    # Input textarea for user to enter reviews
    reviews_input = st.text_area('Enter up to 10,000 reviews (one review per line):', height=200)

    # Button to analyze reviews
    if st.button('Analyze Reviews'):
        if reviews_input:
            # Split reviews into individual lines
            reviews_list = reviews_input.split('\n')
            # Predict sentiment for each review
            sentiments = [predict_sentiment(review) for review in reviews_list]
            # Count positive, negative, and neutral sentiments
            positive_count = sentiments.count('Positive')
            negative_count = sentiments.count('Negative')
            neutral_count = sentiments.count('Neutral')
            # Calculate percentage of each sentiment
            total_reviews = len(sentiments)
            positive_percentage = (positive_count / total_reviews) * 100
            negative_percentage = (negative_count / total_reviews) * 100
            neutral_percentage = (neutral_count / total_reviews) * 100

            # Overall sentiment analysis
            overall_sentiment = 'Positive' if positive_count > negative_count else 'Negative'

            # Display sentiment analysis results
            st.subheader('Sentiment Analysis Results:')
            st.write(f'Positive Sentiment: {positive_percentage:.2f}%')
            st.write(f'Negative Sentiment: {negative_percentage:.2f}%')
            st.write(f'Neutral Sentiment: {neutral_percentage:.2f}%')
            st.write(f'Overall Sentiment: {overall_sentiment}')

            # Plot sentiment distribution
            sentiment_data = pd.DataFrame({
                'Sentiment': ['Positive', 'Negative', 'Neutral'],
                'Percentage': [positive_percentage, negative_percentage, neutral_percentage]
            })
            fig, ax = plt.subplots()
            ax.bar(sentiment_data['Sentiment'], sentiment_data['Percentage'])
            ax.set_xlabel('Sentiment')
            ax.set_ylabel('Percentage')
            ax.set_title('Sentiment Distribution')
            st.pyplot(fig)

            # Find most common words for each sentiment
            positive_reviews = [review for review, sentiment in zip(reviews_list, sentiments) if sentiment == 'Positive']
            negative_reviews = [review for review, sentiment in zip(reviews_list, sentiments) if sentiment == 'Negative']
            neutral_reviews = [review for review, sentiment in zip(reviews_list, sentiments) if sentiment == 'Neutral']

            # Tokenize and preprocess reviews
            positive_words = ' '.join(positive_reviews).split()
            negative_words = ' '.join(negative_reviews).split()
            neutral_words = ' '.join(neutral_reviews).split()

            # Calculate most common words
            positive_word_counts = Counter(positive_words)
            negative_word_counts = Counter(negative_words)
            neutral_word_counts = Counter(neutral_words)

            # Plot most common words for each sentiment
            st.subheader('Most Common Words:')
            st.write('Most Positive Words:')
            st.bar_chart(pd.DataFrame(positive_word_counts.most_common(10), columns=['Word', 'Count']).set_index('Word'))
            st.write('Most Negative Words:')
            st.bar_chart(pd.DataFrame(negative_word_counts.most_common(10), columns=['Word', 'Count']).set_index('Word'))
            st.write('Most Neutral Words:')
            st.bar_chart(pd.DataFrame(neutral_word_counts.most_common(10), columns=['Word', 'Count']).set_index('Word'))

            # Tokenize and preprocess reviews
            positive_words = ' '.join(positive_reviews)
            negative_words = ' '.join(negative_reviews)
            neutral_words = ' '.join(neutral_reviews)

            # Create word cloud for each sentiment
            st.subheader('Word Clouds:')
            st.write('Positive Words:')
            positive_wordcloud = WordCloud(color_func=lambda *args, **kwargs: "green").generate(positive_words)
            st.image(positive_wordcloud.to_array(), caption='Positive Word Cloud', use_column_width=True)

            st.write('Negative Words:')
            negative_wordcloud = WordCloud(color_func=lambda *args, **kwargs: "red").generate(negative_words)
            st.image(negative_wordcloud.to_array(), caption='Negative Word Cloud', use_column_width=True)

            st.write('Neutral Words:')
            neutral_wordcloud = WordCloud(color_func=lambda *args, **kwargs: "gray").generate(neutral_words)
            st.image(neutral_wordcloud.to_array(), caption='Neutral Word Cloud', use_column_width=True)

        else:
            st.write('Please enter some reviews.')

if __name__ == '__main__':
    main()
