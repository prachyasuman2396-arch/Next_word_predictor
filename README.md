# Next Word Prediction System

This project is a **Next Word Prediction web application** built using a deep learning language model.  
Given an input text, the model predicts the next few words **iteratively**, similar to text autocomplete or search suggestion systems.

The application is developed using **Flask** for the backend and a trained **LSTM-based language model** for prediction. It is deployed using **Gunicorn** on Render.

---

## Live Demo

The application is deployed and accessible at:

https://next-word-predictor-zdw8.onrender.com

---

## Features

- Predicts the next *N* words based on user input  
- Uses autoregressive text generation (one word at a time)  
- Handles long input safely using a sliding context window  
- Simple and clean web interface  
- Optimized to run on limited resources  
- Deployed as a production-ready Flask application  

---

## How It Works

1. The user enters an input text.
2. The text is tokenized using the same tokenizer used during training.
3. Only the most recent `max_len - 1` tokens are considered (sliding window approach).
4. The model predicts the next word using a softmax probability distribution.
5. The predicted word is appended back to the input.
6. Steps 3–5 are repeated to generate multiple words.

This approach is known as **autoregressive next-word generation**.

---

## Model Architecture

- Embedding Layer
- Stacked LSTM layers
- Dense output layer with softmax activation
- Loss function: Sparse Categorical Crossentropy
- Optimizer: Adam

---

## Tech Stack

- Python  
- TensorFlow / Keras  
- Flask  
- Gunicorn  
- NumPy  

---

## Project Structure

├── app.py
├── model.pkl
├── requirements.txt
├── README.md


---

## Running Locally

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd <project-folder>
   
2. create and activate a virtual environment
python -m venv myenv
source myenv/bin/activate

3. install dependancies
 pip install -r requirements.txt

4. run the application
  python app.py

5. Open your browser and go to:
  http://127.0.0.1:8000

Deployment
The application is deployed on Render using Gunicorn with a single worker to ensure stable inference under limited memory constraints.

Limitations and Future Improvements
The model uses a limited context window and may lose long-range coherence.
Text generation quality can be improved using sampling techniques such as top-k or temperature sampling.
A Transformer-based model could further improve performance for longer contexts.

