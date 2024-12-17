import streamlit as st
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from typing import List
import itertools

class KeywordIntentAnalyzer:
    def __init__(self):
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
    
    def calculate_similarities(self, items: List[str]) -> pd.DataFrame:
        # Generate embeddings for all items
        embeddings = self.model.encode(items)
        
        # Calculate cosine similarity between all pairs
        similarities = []
        for i, j in itertools.combinations(range(len(items)), 2):
            similarity = self._cosine_similarity(embeddings[i], embeddings[j])
            similarities.append({
                'Item 1': items[i],
                'Item 2': items[j],
                'Similarity Score': round(float(similarity), 3)
            })
        
        return pd.DataFrame(similarities)
    
    def _cosine_similarity(self, a, b):
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def main():
    st.set_page_config(
        page_title="Keyword Intent Similarity Analyzer",
        page_icon="🔍",
        layout="wide"
    )
    
    st.title("🔍 Keyword Intent Similarity Analyzer")
    st.write("""
    This tool analyzes the semantic similarity between keywords and topics based on their underlying intent.
    Enter your keywords/topics (one per line) in the text area below.
    """)
    
    # Initialize the analyzer
    analyzer = KeywordIntentAnalyzer()
    
    # Text input for keywords/topics
    text_input = st.text_area(
        "Enter your keywords/topics (one per line):",
        height=200,
        placeholder="Enter keywords or topics here...\nExample:\nseo tools\nkeyword research\ncontent optimization"
    )
    
    if st.button("Analyze Similarities"):
        if text_input.strip():
            # Process the input
            items = [item.strip() for item in text_input.split('\n') if item.strip()]
            
            if len(items) < 2:
                st.error("Please enter at least 2 keywords/topics to compare.")
            else:
                with st.spinner("Calculating similarities..."):
                    # Calculate similarities
                    df = analyzer.calculate_similarities(items)
                    
                    # Display results
                    st.subheader("Similarity Results")
                    
                    # Style the dataframe
                    def color_similarity(val):
                        color = f'background-color: rgba(76, 175, 80, {val})'
                        return color
                    
                    styled_df = df.style.applymap(
                        color_similarity,
                        subset=['Similarity Score']
                    )
                    
                    st.dataframe(
                        styled_df,
                        use_container_width=True
                    )
                    
                    # Find highest similarity pair
                    highest_sim = df.loc[df['Similarity Score'].idxmax()]
                    st.success(f"Highest similarity found between '{highest_sim['Item 1']}' and '{highest_sim['Item 2']}' with a score of {highest_sim['Similarity Score']}")
        else:
            st.warning("Please enter some keywords/topics to analyze.")

if __name__ == "__main__":
    main()
