import streamlit as st
import pandas as pd

# Title of the app
st.title('IMDB Review Sentiment Analysis')

# Description
st.write('This app analyzes the sentiment of IMDB movie reviews using a pre-trained model.')

# Example reviews
example_reviews = [
    'The movie was fantastic! I loved it.',
    'It was a terrible waste of time.',
    'I found the film to be quite average.',
]

st.subheader('Example Reviews')
for review in example_reviews:
    st.write(f'- {review}')

# Input from user
st.subheader('Enter Your Review')
user_review = st.text_area('Review:', '')

# Analysis placeholder
if st.button('Analyze'):
    # Placeholder for sentiment analysis logic
    # Here you would call your model and perform prediction
    if user_review:
        result = 'Positive' if 'fantastic' in user_review else 'Negative'  # Dummy logic
        st.success(f'The sentiment of your review is: {result}')
    else:
        st.warning('Please enter a review to analyze.')

# Styling the app
st.markdown("""
<style>
body {
    font-family: Arial, sans-serif;
}
header {
    font-size: 24px;
    color: #4CAF50;
}
</style>
""")

st.sidebar.header('Additional Information')
st.sidebar.write('For more insights, feel free to check our other applications. Connect with us on our social media!')
