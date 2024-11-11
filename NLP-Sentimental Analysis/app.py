import streamlit as st
import torch
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
import re
import torch.nn as nn
import torch.nn.functional as F
import time


# Define your model class
class NN(nn.Module):
    def __init__(self, input_size, num_classes, dropout_prob=0.5):
        super(NN, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.bn1 = nn.BatchNorm1d(128)
        self.fc2 = nn.Linear(128, 64)
        self.bn2 = nn.BatchNorm1d(64)
        self.fc3 = nn.Linear(64, 32)
        self.bn3 = nn.BatchNorm1d(32)
        self.fc4 = nn.Linear(32, num_classes)
        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, x):
        x = F.relu(self.bn1(self.fc1(x)))
        x = self.dropout(x)
        x = F.relu(self.bn2(self.fc2(x)))
        x = self.dropout(x)
        x = F.relu(self.bn3(self.fc3(x)))
        x = self.dropout(x)
        x = self.fc4(x)
        return x


# Initialize model with input size and number of classes
input_size = 4073  # Adjust based on your vectorizer
num_classes = 2
model = NN(input_size=input_size, num_classes=num_classes)

# Load the state dictionary
model_path = "model.pth"
model.load_state_dict(torch.load(model_path))
model.eval()

# Load the state dictionary into the model
model_path = "model.pth"  # Adjust this if needed
model.load_state_dict(torch.load(model_path))
model.eval()

# Load the fitted vectorizer
with open("vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)


# Preprocess the input text
def preprocess_text(text):
    text = text.lower()
    tokens = word_tokenize(text)
    tokens = [re.sub(r"[^a-zA-Z0-9]", "", token) for token in tokens if token]
    stop_words = set(stopwords.words("english"))
    tokens = [token for token in tokens if token not in stop_words]
    stemmer = PorterStemmer()
    tokens = [stemmer.stem(token) for token in tokens]
    return " ".join(tokens)


# Predict the sentiment of the input text
def predict_text_label(text):
    processed_text = preprocess_text(text)
    vectorized_text = vectorizer.transform([processed_text]).toarray()
    input_tensor = torch.tensor(vectorized_text, dtype=torch.float32)

    with torch.no_grad():
        scores = model(input_tensor)
        _, predicted_label = scores.max(1)

    label_map = {0: "Negative", 1: "Positive"}
    return label_map[predicted_label.item()]


# Apply CSS for a dark theme with white text and color key
st.markdown(
    """
    <style>
    body {
        background-color: #1E1E1E;
        color: #FFFFFF;
    }
    .stApp {
        background-color: #1E1E1E;
        color: #FFFFFF;
    }
    .title {
        color: #FFDD57;
        font-size: 40px;
        font-weight: bold;
        text-align: center;
        margin-bottom: 20px;
    }
    .subtitle {
        color: #9CDCFE;
        font-size: 20px;
        text-align: center;
    }
    .highlight-positive {
        background-color: #32CD32;
        color: #000000;
        font-weight: bold;
        padding: 4px;
    }
    .highlight-negative {
        background-color: #FF4500;
        color: #000000;
        font-weight: bold;
        padding: 4px;
    }
    .key {
        font-size: 18px;
        font-weight: bold;
        padding: 10px;
        color: #FFFFFF;
        margin-top: 20px;
        text-align: center;
    }
    .key-positive {
        color: #32CD32;
        font-weight: bold;
    }
    .key-negative {
        color: #FF4500;
        font-weight: bold;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Streamlit app layout
st.markdown(
    '<div class="title">Real-Time Sentiment Analysis</div>', unsafe_allow_html=True
)
st.markdown(
    '<div class="subtitle">Type text, and predictions will appear after each full stop.</div>',
    unsafe_allow_html=True,
)

# Display color key
st.markdown(
    """
<div class="key">
    <span class="key-positive">Green: Positive Sentiment</span> | 
    <span class="key-negative">Red: Negative Sentiment</span>
</div>
""",
    unsafe_allow_html=True,
)

# Text input for user with automatic rerun upon changes
if "user_text" not in st.session_state:
    st.session_state["user_text"] = ""

# Update user input state
user_input = st.text_area("Enter your text here:", value=st.session_state["user_text"])
if user_input != st.session_state["user_text"]:
    st.session_state["user_text"] = user_input
    st.experimental_rerun()

# Process input dynamically
if user_input:
    # Split text by full stops and process each part
    sentences = user_input.split(".")
    highlighted_text = ""
    for sentence in sentences:
        if sentence.strip():  # Only process non-empty sentences
            prediction = predict_text_label(sentence)
            # Apply different styles based on sentiment
            if prediction == "Positive":
                highlighted_text += (
                    f'<span class="highlight-positive">{sentence}.</span> '
                )
            else:
                highlighted_text += (
                    f'<span class="highlight-negative">{sentence}.</span> '
                )
    st.markdown(highlighted_text, unsafe_allow_html=True)
