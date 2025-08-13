# LSTM Quote Generator

A web application that generates inspirational quotes using an LSTM neural network trained on travel, life, and wisdom quotes.

<img width="2846" height="1525" alt="image" src="https://github.com/user-attachments/assets/936b67b7-3bc1-4ace-802d-ea6afcd7ff45" />

## Features

- LSTM text generation with TensorFlow/Keras
- Web interface with Flask backend
- Customizable seed text and output length
- Trained on 200+ inspirational quotes
- Responsive design with real-time generation

## Technology Stack

- **Backend**: Flask, TensorFlow, NumPy
- **Frontend**: HTML, CSS, JavaScript
- **Model**: LSTM with embedding layer (64 units each)

## Installation

1. Clone the repository
2. Install dependencies:
   ```bash
   pip install flask tensorflow numpy
   ```
3. Run the application:
   ```bash
   python app.py
   ```
4. Open http://127.0.0.1:5000

## How It Works

The LSTM model is trained on n-gram sequences from inspirational quotes. During generation, it predicts the next word based on the input seed text and context, iteratively building longer quotes.

**Model Architecture:**
- Embedding layer (64 dimensions)
- LSTM layer (64 units)  
- Dense output with softmax activation

## Usage

1. Enter a seed word or phrase
2. Set number of words to generate (1-50)
3. Click "Generate Quote"

**Example:**
- Input: "Travel" 
- Output: "Travel far enough to meet yourself and discover new adventures"

## Dataset

Training data includes quotes about:
- Travel & Adventure
- Life Philosophy  
- Nature & Environment
- Personal Growth
- Success & Achievement
- Relationships
- Learning & Wisdom
