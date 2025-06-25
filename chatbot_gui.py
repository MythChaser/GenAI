import streamlit as st
from llama_cpp import Llama
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# Initialize the Llama model
llama = Llama(model_path="backend/model.gguf", n_ctx=512, n_batch=4, n_threads=8, n_predict=512, top_k=40, top_p=0.9, temperature=0.7, repeat_penalty=1.1)

# Function to analyze sentiment of user input
def get_sentiment(text):
    analyzer = SentimentIntensityAnalyzer()
    sentiment_score = analyzer.polarity_scores(text)['compound']

    if sentiment_score > 0.1:
        sentiment = "positive"
    elif sentiment_score < -0.1:
        sentiment = "negative"
    else:
        sentiment = "neutral"
    return sentiment

def preprocess(text):
    return f"<s>[INST] {text + '.'} [/INST]"

# Function to modify the prompt based on sentiment
def modify_prompt(text, sentiment):
    if sentiment == "positive":
        return f"You are a helpful, encouraging assistant. Answer this positively: {text}"
    elif sentiment == "negative":
        return f"You are a helpful, empathetic assistant. Answer this with empathy: {text}"
    else:
        return f"You are a neutral assistant. Answer this neutrally: {text}"

# Streamlit UI
st.title("Sentiment-Aware Chatbot")
st.write("Type your message below and the chatbot will respond with a sentiment-aware reply!")

# Text input from the user
user_input = preprocess(st.text_input("You:", ""))

if user_input:
    # Analyze the sentiment of the user input
    sentiment = get_sentiment(user_input)
    
    # Modify the prompt based on sentiment
    prompt = modify_prompt(user_input, sentiment)
    
    # Get response from Llama model
    try:
        result = llama(prompt, max_tokens=1000, stop=["User:", "You:"])
        bot_reply = result["choices"][0]["text"].strip()
        st.write(f"Bot ({sentiment}): {bot_reply}")
    except Exception as e:
        st.error(f"[Error]: {e}")

