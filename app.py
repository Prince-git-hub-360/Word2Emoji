import streamlit as st
from textblob import TextBlob
from transformers import pipeline
import nltk
import re

# Download NLTK data safely
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)

# -----------------------------
# Config / Model names
# -----------------------------
EMOTION_MODEL = "j-hartmann/emotion-english-distilroberta-base"
SENTIMENT_MODEL = "distilbert-base-uncased-finetuned-sst-2-english"

# -----------------------------
# Load Hugging Face Pipelines
# -----------------------------
@st.cache_resource(show_spinner=False)
def load_pipelines():
    emotion_pipe, sentiment_pipe = None, None
    try:
        emotion_pipe = pipeline("text-classification", model=EMOTION_MODEL, return_all_scores=True, device=-1)
    except Exception as e:
        logging.exception("Emotion model failed to load")

    try:
        sentiment_pipe = pipeline("sentiment-analysis", model=SENTIMENT_MODEL, device=-1)
    except Exception as e:
        logging.exception("Sentiment model failed to load")

    return emotion_pipe, sentiment_pipe

emotion_pipe, sentiment_pipe = load_pipelines()

# -----------------------------
# Helper Functions
# -----------------------------
def clean_text(text):
    return re.sub(r"[^a-zA-Z\s]", " ", text).strip().lower()

def contains_bad_words(text):
    bad_words = {"kill", "die", "suicide", "sex", "stupid", "idiot", "dumb", "hate", "abuse"}
    toks = set(clean_text(text).split())
    return len(bad_words.intersection(toks)) > 0

def detect_self_harm(text):
    t = text.lower()
    patterns = [
        r"\bkill myself\b", r"\bkill me\b", r"\bi want to die\b",
        r"\bi want to end my life\b", r"\bsuicid(e|al)\b", r"\bi can.?t go on\b",
        r"\bi.?m done\b", r"\bi am done\b"
    ]
    return any(re.search(p, t) for p in patterns)

def is_gibberish(text):
    """Detects nonsense/gibberish text."""
    if not text.strip():
        return True

    tokens = text.split()
    if len(tokens) < 2:
        return True

    if re.search(r"(.)\1{3,}", text):  # repeated characters
        return True

    vowels = sum(1 for w in tokens if re.search(r"[aeiou]", w))
    if vowels / max(1, len(tokens)) < 0.4:
        return True

    letters = sum(1 for ch in text if ch.isalpha())
    if letters / max(1, len(text)) < 0.6:
        return True

    try:
        blob = TextBlob(text)
        if abs(blob.sentiment.polarity) < 0.05 and vowels / len(tokens) < 0.5:
            return True
    except Exception:
        pass

    return False

# -----------------------------
# Emoji map
# -----------------------------
EMOJI_MAP = {
    "joy": ("😀", "Sounds happy!"),
    "happy": ("😀", "Sounds happy!"),
    "love": ("😀", "That sounds lovely!"),
    "surprise": ("😲", "That sounds surprising!"),
    "anger": ("😠", "You sound upset."),
    "sadness": ("😞", "Seems sad."),
    "fear": ("😟", "You seem worried or scared."),
    "neutral": ("😐", "Feels neutral.")
}

# -----------------------------
# Main logic
# -----------------------------
def predict_mood(text):
    text = text.strip()

    if detect_self_harm(text):
        return ("⚠️", "If you’re feeling this way, please talk to a trusted adult or call your local helpline immediately. You are not alone.")

    if is_gibberish(text):
        return ("🤔", "I’m not sure what you mean. Could you rephrase that?")

    if contains_bad_words(text):
        return ("🚫", "Please use kind and respectful words.")

    # Emotion model
    if emotion_pipe:
        try:
            results = emotion_pipe(text)
            if isinstance(results, list) and len(results) > 0:
                scores = results[0]
                best = max(scores, key=lambda x: x["score"])
                label, score = best["label"].lower(), best["score"]
                for key, (emoji, msg) in EMOJI_MAP.items():
                    if key in label and score > 0.4:
                        return emoji, msg
        except Exception:
            pass

    # Sentiment model
    if sentiment_pipe:
        try:
            res = sentiment_pipe(text)
            if res:
                label = res[0]["label"].lower()
                if label == "positive":
                    return ("😀", "Sounds happy!")
                elif label == "negative":
                    return ("😞", "Seems sad.")
                else:
                    return ("😐", "Feels neutral.")
        except Exception:
            pass

    # TextBlob fallback
    blob = TextBlob(text)
    polarity = blob.sentiment.polarity
    if polarity > 0.25:
        return ("😀", "Sounds happy!")
    elif polarity < -0.25:
        return ("😞", "Seems sad.")
    else:
        return ("😐", "Feels neutral.")

# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(page_title="Mood2Emoji", page_icon="😀")
st.title(" Mood2Emoji — Kid-Safe Text Mood Detector")

st.write("This AI understands tone, context, and emotion — not just words. ")
user_input = st.text_input("Type or paste a short sentence here:", placeholder="e.g., I won the match today!")

if st.button("Check Mood"):
    if not user_input.strip():
        st.warning("Please type something first.")
    else:
        emoji, message = predict_mood(user_input)
        if emoji == "⚠️":
            st.error(f"{emoji} {message}")
        elif emoji == "🚫":
            st.warning(f"{emoji} {message}")
        elif emoji == "🤔":
            st.info(f"{emoji} {message}")
        else:
            st.markdown(f"## {emoji} {message}")

with st.expander("Teacher Mode — How it works"):
    st.markdown("""
    **How this AI works:**
    1. Cleans and checks for safety (bad or harmful words).  
    2. Detects gibberish and asks to rephrase if unclear.  
    3. Uses a **Transformer model** for emotion (joy, sadness, anger, fear, surprise).  
    4. Uses **TextBlob** as backup for tone understanding.  
    5. Returns a kid-safe emoji and friendly explanation.  
    """)

st.caption("""Made by Prince Kumar 
           with using Hugging Face + Streamlit + TextBlob • \n For ages 12–16""")


