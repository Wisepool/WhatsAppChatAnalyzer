import os
import streamlit as st
from streamlit.errors import StreamlitSecretNotFoundError


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
    secret_values = st.secrets
except StreamlitSecretNotFoundError:
    secret_values = {}

GEMINI_API_KEYS = [
    key for key in (
        secret_values.get("gemini_api_key1") or os.getenv("GEMINI_API_KEY1"),
        secret_values.get("gemini_api_key2") or os.getenv("GEMINI_API_KEY2"),
    )
    if key
]
NGROK_AUTH_TOKEN = secret_values.get("ngrok_auth_token") or os.getenv("NGROK_AUTH_TOKEN", "")

if not GEMINI_API_KEYS:
    GEMINI_API_KEYS = [
        key for key in (
            os.getenv("GEMINI_API_KEY1"),
            os.getenv("GEMINI_API_KEY2"),
        )
        if key
    ]

# Replace with your actual API key from https://dashboard.ngrok.com/get-started/your-authtoken
# NGROK_AUTH_TOKEN = "ngrok_auth_token"
PORT = 8501