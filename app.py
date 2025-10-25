import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import random

st.set_page_config(
    page_title="Fauna -Network- Classifier",    
    page_icon="🐾", 
    layout="centered",                         
)
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Bitcount+Single+Ink:wght@100..900&display=swap');
    html, body, [class*="css"]  {
        font-family: 'Bitcount Single Ink', monospace !important;
        background: #16171d;
        color: #00ffea;
        letter-spacing: 2px;
    }
    h1, h2, h3, h4, h5, h6,
    .stMarkdown, .stTextInput, .stTextArea, .stSelectbox,
    .stFileUploader, .stButton>button, .stAlert, .stInfo, .stError, .stWarning {
        font-family: 'Bitcount Single Ink', monospace !important;
        letter-spacing: 2px;
    }
    
    /* Super curvy, glassy file uploader */
    .stFileUploader {
        background: rgba(44,46,78,0.95) !important;
        border-radius: 38px !important;
        border: 2.5px solid #33ffe6 !important;
        box-shadow: 0 4px 32px 0 #33ffe688, 0 2px 10px 0 #22223b;
        padding: 30px !important;
        margin-top: 16px;
        margin-bottom: 28px;
        transition: box-shadow 0.3s, border-color 0.3s;
    }
    .stFileUploader:hover {
        box-shadow: 0 10px 48px 0 #33ffe6cc, 0 2px 10px 0 #22223b;
        border-color: #a8002c !important;
    }
    /* Browse files button 3D glass effect - blackish red */
    .stFileUploader button, .stFileUploader input[type="file"]::file-selector-button {
        font-family: 'Bitcount Single Ink', monospace !important;
        background: linear-gradient(90deg, #240404 60%, #a8002c 100%) !important;
        border-radius: 18px !important;
        border: none !important;
        color: #ffb3c6 !important;
        font-size: 1.22em;
        font-weight: 700;
        box-shadow: 0 2px 12px #a8002c44;
        padding: 8px 28px;
        margin-right: 10px;
        transition: background 0.4s, box-shadow 0.3s;
        text-shadow: 0 0 2px #a8002c88, 0 2px 8px #22223b;
    }
    .stFileUploader button:hover, .stFileUploader input[type="file"]::file-selector-button:hover {
        background: linear-gradient(90deg, #a8002c 45%, #240404 100%) !important;
        color: #fff0f5 !important;
        box-shadow: 0 4px 24px #a8002ccc;
    }
    /* Info/collapsible sections */
    .stAlert, .stInfo, .stError, .stWarning {
        font-family: 'Bitcount Single Ink', monospace !important;
        border-radius: 18px !important;
        font-size: 1.12em;
        background: rgba(44,46,78,0.94) !important;
        color: #bb86fc !important;
        box-shadow: 0 2px 8px #33ffe655;
        border: 1px solid #33ffe6 !important;
        letter-spacing: 1.5px;
        margin-bottom: 18px;
    }
    /* Centered, glassy header */
    .bitcount-header {
        font-family: 'Bitcount Single Ink', monospace !important;
        color: #00fff7;
        font-size: 2.2em;
        text-align: center;
        margin-top: 28px;
        margin-bottom: 16px;
        letter-spacing: 2.3px;
        text-shadow: 0 2px 24px #33ffe6cc, 0 1px 2px #23253a;
    }
    /* Centered, glassy subheader */
    .bitcount-subheader {
        font-family: 'Bitcount Single Ink', monospace !important;
        color: #00fff7;
        font-size: 1.3em;
        text-align: center;
        margin-bottom: 36px;
        letter-spacing: 2px;
    }
    /* Add space between info and update text */
    .bitcount-info {
        margin-bottom: 26px;
        margin-top: 6px;
    }
    .bitcount-update {
        margin-top: 22px;
        margin-bottom: 12px;
    }
    </style>
""", unsafe_allow_html=True)

# Fun facts
cat_facts = [
    "Cats have five toes on their front paws, but only four on the back!",
    "A group of cats is called a clowder.",
    "Cats sleep for 70% of their lives.",
    "The oldest cat lived to 38 years old!",
]
dog_facts = [
    "Dogs have a sense of smell that's 40x better than humans.",
    "A dog’s nose print is as unique as a human’s fingerprint.",
    "Dogs can understand up to 250 words and gestures.",
    "The Basenji is the only barkless dog.",
]

# Track counts
if "cat_count" not in st.session_state:
    st.session_state.cat_count = 0
if "dog_count" not in st.session_state:
    st.session_state.dog_count = 0

@st.cache_resource
def load_model():
    model = tf.keras.models.load_model('cat_vs_dog_model.keras')
    return model

model = load_model()

#  header with Bitcount font
st.markdown(
    """
    <div style='text-align:center; margin-bottom: 20px;'>
        <h1 style='font-size:2.6em;'>🐾 Cat vs Dog Classifier</h1>
        <h3 style='font-weight:400; color:#00ffea;'>Upload a photo, get a prediction, and learn something cool!</h3>
    </div>
    """, unsafe_allow_html=True
)

uploaded_files = st.file_uploader(
    "Drag and drop or select image files (jpg/png) for prediction",
    type=["jpg", "jpeg", "png"],
    accept_multiple_files=True,
    key="main_file_uploader"
)

if uploaded_files:
    st.markdown("<h2 style='margin-top:16px;'>Results</h2>", unsafe_allow_html=True)
    for uploaded_file in uploaded_files:
        st.markdown(
            f"<div class='uploadedFile'><b>Image:</b> {uploaded_file.name}</div>",
            unsafe_allow_html=True
        )
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, width=256)

        # Preprocess
        img = image.resize((128, 128))
        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        # Predict
        prediction = model.predict(img_array)
        if prediction[0][0] > 0.5:
            label = "DOG 🐶"
            st.session_state.dog_count += 1
            st.markdown(
                f"<div style='color:#08f7fe; font-size:1.3em; font-family:Bitcount Single Ink, monospace;'>"
                f"Woof! That's a pawsome dog!<br>"
                f"<span style='font-size:1.1em;'>Dog Fact:</span> {random.choice(dog_facts)}"
                f"</div>", unsafe_allow_html=True
            )
        else:
            label = "CAT 🐱"
            st.session_state.cat_count += 1
            st.markdown(
                f"<div style='color:#f72585; font-size:1.3em; font-family:Bitcount Single Ink, monospace;'>"
                f"Meow! That’s a purrfect cat!<br>"
                f"<span style='font-size:1.1em;'>Cat Fact:</span> {random.choice(cat_facts)}"
                f"</div>", unsafe_allow_html=True
            )
        st.markdown(f"<h3 style='text-align:center; margin:8px 0; color:#00ffea; font-family:Bitcount Single Ink, monospace;'>Prediction: {label}</h3>", unsafe_allow_html=True)
        st.divider()

    st.markdown(
        "<div class='bitcount-info' style='text-align:center; font-family:Bitcount Single Ink, monospace; color:#bb86fc; font-size:1.18em;'>"
        "Upload jpg/png images of cats or dogs to get playful predictions!"
        "</div>",
        unsafe_allow_html=True
    )
else:
    st.markdown(
        "<center><p class='bitcount-info' style='color:#bb86fc; font-size:1.18em; font-family:Bitcount Single Ink, monospace;'>Upload jpg/png images of cats or dogs to get playful predictions!</p></center>",
        unsafe_allow_html=True
    )
st.markdown(
    "<div class='bitcount-update' style='text-align:center; font-family:Bitcount Single Ink, monospace; color:#bb86fc; font-size:1.13em;'>"
    "Stay tuned for more such updates!! 🐾"
    "</div>",
    unsafe_allow_html=True
)
