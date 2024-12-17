# Keyword Intent Similarity Analyzer

This Streamlit application analyzes the semantic similarity between keywords and topics based on their underlying intent. It uses the sentence-transformers library to calculate similarity scores between pairs of keywords/topics.

## Features

- Input multiple keywords and topics
- Calculate semantic similarity scores between all pairs
- Visual representation of similarity scores
- Highlights the most similar keyword/topic pairs
- User-friendly interface built with Streamlit

## Installation

1. Clone this repository
2. Install the required packages:
```bash
pip install -r requirements.txt
```

## Usage

1. Run the Streamlit app:
```bash
streamlit run app.py
```

2. Enter your keywords/topics (one per line) in the text area
3. Click "Analyze Similarities" to see the results

## How it works

The application uses the `all-MiniLM-L6-v2` model from sentence-transformers to generate embeddings for each keyword/topic. It then calculates the cosine similarity between these embeddings to determine how semantically similar the keywords/topics are. Higher similarity scores (closer to 1.0) indicate that the keywords/topics are more closely related in terms of their underlying intent.
