import streamlit as st
import numpy as np
import cv2
from PIL import Image
import tensorflow as tf

# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="Fashion AI Classifier",
    page_icon="👕",
    layout="wide",
    initial_sidebar_state="collapsed"
)


# =========================================================
# CUSTOM CSS
# =========================================================

st.markdown("""
<style>

    /* Main background */
    .stApp {
        background: linear-gradient(135deg, #0f172a, #111827, #1e293b);
        color: white;
    }

    /* Remove top padding */
    .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
        max-width: 1200px;
    }

    /* Main title */
    .main-title {
        text-align: center;
        font-size: 3rem;
        font-weight: 800;
        margin-bottom: 0.3rem;

        background: linear-gradient(
            90deg,
            #38bdf8,
            #818cf8,
            #c084fc
        );

        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }

    /* Subtitle */
    .subtitle {
        text-align: center;
        color: #cbd5e1;
        font-size: 1.1rem;
        margin-bottom: 2.5rem;
    }

    /* Cards */
    .card {
        background: rgba(30, 41, 59, 0.75);
        border: 1px solid rgba(148, 163, 184, 0.2);
        border-radius: 20px;
        padding: 25px;
        box-shadow: 0 10px 30px rgba(0,0,0,0.25);
        backdrop-filter: blur(10px);
    }

    /* Upload card */
    .upload-card {
        background: rgba(15, 23, 42, 0.8);
        border: 2px dashed #475569;
        border-radius: 20px;
        padding: 25px;
        text-align: center;
        margin-bottom: 25px;
    }

    /* Prediction card */
    .prediction-card {
        background: linear-gradient(
            135deg,
            rgba(14, 165, 233, 0.15),
            rgba(139, 92, 246, 0.15)
        );

        border: 1px solid rgba(129, 140, 248, 0.4);
        border-radius: 20px;
        padding: 30px;
        text-align: center;

        box-shadow:
            0 10px 35px rgba(0,0,0,0.3);
    }

    /* Prediction label */
    .prediction-label {
        color: #94a3b8;
        font-size: 0.95rem;
        margin-bottom: 5px;
    }

    /* Prediction class */
    .prediction-class {
        font-size: 2.2rem;
        font-weight: 800;
        color: #f8fafc;
        margin-bottom: 10px;
    }

    /* Confidence */
    .confidence {
        font-size: 1.2rem;
        color: #38bdf8;
        font-weight: 700;
    }

    /* Section headings */
    .section-title {
        font-size: 1.3rem;
        font-weight: 700;
        color: #f8fafc;
        margin-bottom: 15px;
    }

    /* Footer */
    .footer {
        text-align: center;
        color: #64748b;
        margin-top: 50px;
        font-size: 0.9rem;
    }

    /* Streamlit file uploader */
    [data-testid="stFileUploader"] {
        background: transparent;
    }

    [data-testid="stFileUploaderDropzone"] {
        background: rgba(30, 41, 59, 0.5);
        border: 1px dashed #64748b;
        border-radius: 15px;
    }

    /* Buttons */
    .stButton > button {
        width: 100%;
        border-radius: 12px;
        border: none;
        padding: 0.7rem;
        font-weight: 700;
    }

    /* Hide Streamlit menu */
    #MainMenu {
        visibility: hidden;
    }

    footer {
        visibility: hidden;
    }

</style>
""", unsafe_allow_html=True)


# =========================================================
# LOAD MODEL
# =========================================================

@st.cache_resource
def load_model():
    return tf.keras.models.load_model("ANN.h5")


loaded_model = load_model()


# =========================================================
# CLASS NAMES
# =========================================================

class_names = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot"
]


# =========================================================
# HEADER
# =========================================================

st.markdown(
    '<div class="main-title">👕 Fashion AI Classifier</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Upload a fashion image and let the neural network identify it'
    '</div>',
    unsafe_allow_html=True
)


# =========================================================
# UPLOAD SECTION
# =========================================================

st.markdown(
    '<div class="upload-card">',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="section-title">📤 Upload Your Fashion Image</div>',
    unsafe_allow_html=True
)

uploaded_file = st.file_uploader(
    "Choose an image",
    type=["jpg", "jpeg", "png"],
    label_visibility="collapsed"
)

st.markdown(
    '</div>',
    unsafe_allow_html=True
)


# =========================================================
# IMAGE + PREDICTION
# =========================================================

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    # -----------------------------------------------------
    # IMAGE PROCESSING
    # -----------------------------------------------------

    image_array = np.array(image)

    gray = cv2.cvtColor(
        image_array,
        cv2.COLOR_RGB2GRAY
    )

    resized = cv2.resize(
        gray,
        (28, 28)
    )

    resized = resized.astype("float32") / 255.0

    sample = np.expand_dims(
        resized,
        axis=0
    )

    # -----------------------------------------------------
    # PREDICTION
    # -----------------------------------------------------

    prediction = loaded_model.predict(
        sample,
        verbose=0
    )

    predicted_index = np.argmax(
        prediction[0]
    )

    predicted_class = class_names[
        predicted_index
    ]

    confidence = (
        prediction[0][predicted_index] * 100
    )


    # =====================================================
    # TWO COLUMN LAYOUT
    # =====================================================

    col1, col2 = st.columns(
        [1, 1],
        gap="large"
    )


    # =====================================================
    # IMAGE PREVIEW
    # =====================================================

    with col1:

        st.markdown(
            '<div class="card">',
            unsafe_allow_html=True
        )

        st.markdown(
            '<div class="section-title">🖼️ Uploaded Image</div>',
            unsafe_allow_html=True
        )

        st.image(
            image,
            use_container_width=True
        )

        st.markdown(
            '</div>',
            unsafe_allow_html=True
        )


    # =====================================================
    # PREDICTION RESULT
    # =====================================================

    with col2:

        st.markdown(
            '<div class="prediction-card">',
            unsafe_allow_html=True
        )

        st.markdown(
            '<div class="prediction-label">'
            '🤖 AI Prediction'
            '</div>',
            unsafe_allow_html=True
        )

        st.markdown(
            f'<div class="prediction-class">'
            f'{predicted_class}'
            f'</div>',
            unsafe_allow_html=True
        )

        st.markdown(
            f'<div class="confidence">'
            f'Confidence: {confidence:.2f}%'
            f'</div>',
            unsafe_allow_html=True
        )

        st.progress(
            float(confidence / 100)
        )

        st.markdown(
            '</div>',
            unsafe_allow_html=True
        )


    # =====================================================
    # ALL CLASS PROBABILITIES
    # =====================================================

    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown(
        '<div class="section-title">'
        '📊 Prediction Probabilities'
        '</div>',
        unsafe_allow_html=True
    )

    # Sort predictions from highest to lowest
    sorted_indices = np.argsort(
        prediction[0]
    )[::-1]

    for index in sorted_indices:

        probability = (
            prediction[0][index] * 100
        )

        st.write(
            f"**{class_names[index]}** — "
            f"{probability:.2f}%"
        )

        st.progress(
            float(prediction[0][index])
        )


# =========================================================
# FOOTER
# =========================================================

st.markdown(
    '<div class="footer">'
    'Built with ❤️ using Streamlit + TensorFlow + OpenCV'
    '</div>',
    unsafe_allow_html=True
)