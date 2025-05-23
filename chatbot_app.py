# chatbot_app.py
import random
import streamlit as st

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
except ModuleNotFoundError as e:
    st.error(f"Required module not found: {e.name}")
    st.markdown("Please add the following to your `requirements.txt` file:")
    st.code("streamlit\nscikit-learn\nnumpy", language="text")
    st.stop()

intents = {
    "intents": [
        {"tag": "greeting", "patterns": ["Hi", "Hello", "Hey", "Good morning"], "responses": ["Hello!", "Hi there!", "Greetings!"]},
        {"tag": "goodbye", "patterns": ["Bye", "See you", "Goodbye"], "responses": ["Goodbye!", "See you later!", "Take care!"]},
        {"tag": "thanks", "patterns": ["Thanks", "Thank you", "Much appreciated"], "responses": ["You're welcome!", "Glad to help!", "Anytime!"]},
        {"tag": "support", "patterns": ["I need help", "Can you help me?", "Support please"], "responses": ["How can I assist you today?", "I'm here to help!", "What can I do for you?"]}
    ]
}

X_train, y_train = [], []
for intent in intents["intents"]:
    for pattern in intent["patterns"]:
        X_train.append(pattern)
        y_train.append(intent["tag"])

model = make_pipeline(TfidfVectorizer(), LogisticRegression())
model.fit(X_train, y_train)

def get_response(user_input):
    intent = model.predict([user_input])[0]
    for i in intents["intents"]:
        if i["tag"] == intent:
            return random.choice(i["responses"])

st.set_page_config(page_title="Smart Chatbot", page_icon="🤖")
st.title("🤖 Smart Customer Support Chatbot")
st.write("Ask me anything!")

user_input = st.text_input("You:", placeholder="Type your message here...")

if user_input:
    response = get_response(user_input)
    st.text_area("Bot:", value=response, height=100, max_chars=None, key=None)

