import streamlit as st
import joblib
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import PyPDF2

# Ensure NLTK stopwords are downloaded
nltk.download('stopwords')
nltk.download('punkt')

# Load pre-trained model and vectorizer
model = joblib.load("resume_classifier.pkl")
vectorizer = joblib.load("vectorizer.pkl")

# Function to preprocess text
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)  # Remove special characters
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words('english')) if stopwords.words('english') else set()
    tokens = [word for word in tokens if word not in stop_words]
    return ' '.join(tokens)

# Function to extract text from PDF
def extract_text_from_pdf(pdf_file):
    pdf_reader = PyPDF2.PdfReader(pdf_file)
    text = ""
    for page in pdf_reader.pages:
        page_text = page.extract_text()
        if page_text:  # Check if text is extracted
            text += page_text + "\n"
    return text if text else "No text found in PDF"

# Streamlit UI
st.set_page_config(page_title="Resume Job Role Predictor", layout="wide")

# Custom CSS styling with multiple colors
st.markdown("""
    <style>
    .main {
        background-color: #e3f2fd;
    }
    .stSidebar {
        background-color: #1e3a56;
        color: white;
    }
    .stButton > button {
        background-color: linear-gradient(45deg, #ff8a65, #ff4081);
        color: white;
        font-size: 16px;
        border-radius: 10px;
        border: none;
        padding: 10px;
    }
    .stFileUploader {
        border: 2px dashed #ff7043;
        padding: 10px;
    }
    .stSuccess {
        color: #388e3c;
        font-weight: bold;
    }
    .stError {
        color: #d32f2f;
        font-weight: bold;
    }
    .stTitle {
        color: #1976d2;
        font-size: 28px;
        font-weight: bold;
    }
    </style>
""", unsafe_allow_html=True)

st.sidebar.title("📄 About This App")
st.sidebar.write("This app uses Machine Learning to predict the most suitable job role based on a resume.")
st.sidebar.write("Simply upload a PDF file, and the model will analyze its content to classify the job role.")
st.sidebar.write("🔧 Built with **Python, Streamlit, and Machine Learning**")

st.markdown("<h1 class='stTitle'>🚀 Resume Job Role Predictor</h1>", unsafe_allow_html=True)
st.write("### Upload a resume (PDF) to predict the job role.")

uploaded_file = st.file_uploader("Upload Resume (PDF file)", type=["pdf"], help="Accepted format: PDF")

if uploaded_file is not None:
    st.write(f"✅ Uploaded file: {uploaded_file.name}")
    resume_text = extract_text_from_pdf(uploaded_file)
    
    if resume_text.strip():
        cleaned_text = clean_text(resume_text)
        vectorized_text = vectorizer.transform([cleaned_text]).toarray()
        predicted_role = model.predict(vectorized_text)[0]
        
        st.markdown("<p class='stSuccess'>🎯 Job Role Prediction Completed!</p>", unsafe_allow_html=True)
        st.subheader("Predicted Job Role:")
        st.write(f"🛠 **{predicted_role}**")
    else:
        st.markdown("<p class='stError'>⚠ No readable text found in the uploaded PDF. Please try another file.</p>", unsafe_allow_html=True)