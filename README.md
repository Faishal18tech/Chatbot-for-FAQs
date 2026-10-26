# Chatbot-for-FAQs

# app.py
import re
import streamlit as st
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Download NLTK stopwords
nltk.download("stopwords", quiet=True)
STOP_WORDS = set(stopwords.words("english"))

# 1. FAQ dataset: questions and answers
faqs = [
    {
        "question": "What is a computer?",
        "answer": "A computer is an electronic device that processes data and performs tasks."
    },
    {
        "question": "What is RAM?",
        "answer": "RAM is temporary memory used by a computer to store data for active tasks."
    },
    {
        "question": "What is a CPU?",
        "answer": "The CPU is the central processing unit that executes instructions."
    },
    {
        "question": "How do I connect to Wi-Fi?",
        "answer": "Open your device's Wi-Fi settings, choose a network, and enter its password."
    },
    {
        "question": "What is an operating system?",
        "answer": "An operating system manages computer hardware and software. Examples include Windows, Linux, and macOS."
    },
    {
        "question": "How can I protect my computer?",
        "answer": "Keep software updated, use strong passwords, enable security protection, and avoid suspicious downloads."
    },
    {
        "question": "What is a virus?",
        "answer": "A computer virus is malicious software that can infect files or disrupt a computer."
    },
    {
        "question": "How do I make my computer faster?",
        "answer": "Close unused apps, free up storage, install updates, and check which programs run at startup."
    },
    {
        "question": "What is cloud storage?",
        "answer": "Cloud storage saves files on remote servers that you can access over the internet."
    },
    {
        "question": "How do I reset my password?",
        "answer": "Use the official password-reset option on the relevant service and follow its verification steps."
    }
]

# 2. NLP preprocessing
def preprocess(text):
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    words = text.split()
    words = [word for word in words if word not in STOP_WORDS]
    return " ".join(words)

# 3. Prepare the FAQ search index
faq_questions = [faq["question"] for faq in faqs]
processed_questions = [preprocess(q) for q in faq_questions]

vectorizer = TfidfVectorizer()
faq_vectors = vectorizer.fit_transform(processed_questions)

def get_answer(user_question):
    processed = preprocess(user_question)

    if not processed.strip():
        return "Please enter a question.", 0.0

    query_vector = vectorizer.transform([processed])
    scores = cosine_similarity(query_vector, faq_vectors)[0]
    best_index = scores.argmax()
    best_score = float(scores[best_index])

    # Reject weak matches instead of guessing
    if best_score < 0.15:
        return (
            "Sorry, I couldn't find a suitable answer. "
            "Please try asking in a different way.",
            best_score
        )

    return faqs[best_index]["answer"], best_score

# 4. Streamlit chat interface
st.set_page_config(
    page_title="FAQ Chatbot",
    page_icon="🤖",
    layout="centered"
)

st.title("🤖 FAQ Chatbot")
st.caption("Ask me a question about computers and technology.")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

prompt = st.chat_input("Type your question here...")

if prompt:
    st.session_state.messages.append({
        "role": "user",
        "content": prompt
    })

    answer, score = get_answer(prompt)

    st.session_state.messages.append({
        "role": "assistant",
        "content": answer
    })

    st.rerun()

with st.sidebar:
    st.header("Example questions")
    st.write("• What is a computer?")
    st.write("• Explain RAM")
    st.write("• How do I connect to Wi-Fi?")
    st.write("• How can I protect my computer?")
    if st.button("Clear conversation"):
        st.session_state.messages = []
        st.rerun()
