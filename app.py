from flask import Flask, request, render_template_string
import numpy as np
import pickle
from tensorflow.keras.preprocessing.sequence import pad_sequences

# --------------------------------------------------
# App setup
# --------------------------------------------------
app = Flask(__name__)

# --------------------------------------------------
# Load model, tokenizer, max_len
# --------------------------------------------------
with open("model.pkl", "rb") as f:
    data = pickle.load(f)

model = data["model"]
tokenizer = data["tokenizer"]
max_len = data["max_len"]

# --------------------------------------------------
# HTML template (kept simple & human)
# --------------------------------------------------
HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>Next Word Predictor</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            background: #f5f5f5;
            padding: 40px;
        }
        .box {
            background: white;
            padding: 30px;
            max-width: 700px;
            margin: auto;
            border-radius: 8px;
        }
        textarea, input {
            width: 100%;
            padding: 10px;
            margin-top: 10px;
        }
        button {
            margin-top: 15px;
            padding: 10px 20px;
            cursor: pointer;
        }
        .result {
            margin-top: 20px;
            background: #eef;
            padding: 15px;
            border-radius: 6px;
        }
    </style>
</head>
<body>
    <div class="box">
        <h2>Next Word Prediction</h2>
        <form method="POST">
            <label>Enter text:</label>
            <textarea name="text" rows="4" required>{{ text }}</textarea>

            <label>Number of words to predict:</label>
            <input type="number" name="num_words" value="10" min="1" max="50">

            <button type="submit">Predict</button>
        </form>

        {% if result %}
        <div class="result">
            <strong>Generated Text:</strong>
            <p>{{ result }}</p>
        </div>
        {% endif %}
    </div>
</body>
</html>
"""

# --------------------------------------------------
# Routes
# --------------------------------------------------
@app.route("/", methods=["GET", "POST"])
def index():
    result = ""
    text = ""

    if request.method == "POST":
        text = request.form["text"].strip()
        num_words = int(request.form["num_words"])

        generated_text = text

        for _ in range(num_words):
            sequence = tokenizer.texts_to_sequences([generated_text])[0]

            if len(sequence) == 0:
                break

            # Keep last max_len-1 tokens
            sequence = sequence[-(max_len - 1):]

            padded = pad_sequences(
                [sequence],
                maxlen=max_len - 1,
                padding="pre"
            )

            prediction = model.predict(padded, verbose=0)[0]
            next_index = np.argmax(prediction)

            next_word = None
            for word, idx in tokenizer.word_index.items():
                if idx == next_index:
                    next_word = word
                    break

            if next_word is None:
                break

            generated_text += " " + next_word

        result = generated_text

    return render_template_string(
        HTML_TEMPLATE,
        result=result,
        text=text
    )

# --------------------------------------------------
# Entry point for gunicorn
# --------------------------------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)
