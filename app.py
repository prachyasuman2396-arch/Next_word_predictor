import streamlit as st
import numpy as np
import pickle
from tensorflow.keras.preprocessing.sequence import pad_sequences

# --------------------------------------------------
# Page configuration
# --------------------------------------------------
st.set_page_config(
    page_title="Next Word Predictor",
    page_icon="🔍"
)

st.title("🔍 Next Word Predictor")
st.write("Start typing and let the model complete the text for you.")

# --------------------------------------------------
# Load model, tokenizer and max_len
# --------------------------------------------------
@st.cache_resource
def load_artifacts():
    with open("model.pkl", "rb") as f:
        data = pickle.load(f)

    return data["model"], data["tokenizer"], data["max_len"]


model, tokenizer, max_len = load_artifacts()

# --------------------------------------------------
# User input
# --------------------------------------------------
input_text = st.text_input(
    "Type something:",
    placeholder="example: i am"
)

num_words = st.slider(
    "Number of words to predict",
    min_value=1,
    max_value=20,
    value=10
)

# --------------------------------------------------
# Prediction logic
# --------------------------------------------------
if st.button("Predict"):
    if input_text.strip() == "":
        st.warning("Please enter some text.")
    else:
        generated_text = input_text.strip()

        for _ in range(num_words):
            # Convert text to sequence
            sequence = tokenizer.texts_to_sequences([generated_text])[0]

            if len(sequence) == 0:
                st.error("Input contains words not present in the vocabulary.")
                break

            # Keep only last (max_len - 1) tokens (important)
            sequence = sequence[-(max_len - 1):]

            # Pad sequence
            padded_sequence = pad_sequences(
                [sequence],
                maxlen=max_len - 1,
                padding="pre"
            )

            # Predict next word
            prediction = model.predict(padded_sequence, verbose=0)[0]
            next_index = np.argmax(prediction)

            # Map index to word
            next_word = None
            for word, index in tokenizer.word_index.items():
                if index == next_index:
                    next_word = word
                    break

            if next_word is None:
                break

            generated_text += " " + next_word

        st.markdown("### ✨ Prediction")
        st.write(generated_text)
