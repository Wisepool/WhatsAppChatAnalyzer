import os
import streamlit as st


# Stopwords directory path
STOPWORDS_PATH = 'assets/stopwords'

# Path to your Bengali/Banglish stopwords list (one word per line)
STOPWORDS_FILE_PATHS = [
    os.path.join(STOPWORDS_PATH, "stopwords_bengali.txt"),
    os.path.join(STOPWORDS_PATH, "stopwords_banglish.txt"),
    os.path.join(STOPWORDS_PATH, "stopwords_english.txt"),
]

# !wget -q https://raw.githubusercontent.com/hmoazzem/bangla-fonts/refs/heads/master/Siyamrupali.ttf -O Siyamrupali.ttf
BENGALI_FONT_PATH = 'assets/font/Siyamrupali.ttf'

# Replace with your actual API key from https://aistudio.google.com/apikey
# GEMINI_API_KEYS = [
#     "gemini_api_key1",
#     "gemini_api_key2",
# ]

try:
    GEMINI_API_KEYS = [
        st.secrets["gemini_api_key1"],
        st.secrets["gemini_api_key2"],
    ]
    NGROK_AUTH_TOKEN = st.secrets["ngrok_auth_token"]
except (FileNotFoundError, KeyError):
    GEMINI_API_KEYS = [
        key for key in (
            os.getenv("GEMINI_API_KEY1"),
            os.getenv("GEMINI_API_KEY2"),
        )
        if key
    ]
    NGROK_AUTH_TOKEN = os.getenv("NGROK_AUTH_TOKEN", "")

# Replace with your actual API key from https://dashboard.ngrok.com/get-started/your-authtoken
# NGROK_AUTH_TOKEN = "ngrok_auth_token"
PORT = 8501