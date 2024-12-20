import streamlit as st
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from typing import List
import itertools
import io
import re

def clean_text(text: str) -> str:
    """Clean text by removing special characters and extra whitespace."""
    if not isinstance(text, str):
        return ""
    # Remove the 'Â' character specifically
    text = text.replace('Â', '')
    # Remove other special characters but keep basic punctuation
    text = re.sub(r'[^\w\s\-.,]', '', text)
    # Remove extra whitespace
    text = ' '.join(text.split())
    return text.strip()

class KeywordIntentAnalyzer:
    def __init__(self):
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
    
    def calculate_similarities(self, items: List[str]) -> pd.DataFrame:
        # Clean and remove any empty strings and duplicates while preserving order
        items = [clean_text(item) for item in items]
        items = list(dict.fromkeys([item for item in items if item]))
        
        # Generate embeddings for all items
        embeddings = self.model.encode(items)
        
        # Calculate cosine similarity between all pairs
        similarities = []
        for i, j in itertools.combinations(range(len(items)), 2):
            similarity = self._cosine_similarity(embeddings[i], embeddings[j])
            similarities.append({
                'Item 1': items[i],
                'Item 2': items[j],
                'Similarity Score': round(float(similarity), 3)  # Keep as number for filtering
            })
        
        return pd.DataFrame(similarities)
    
    def _cosine_similarity(self, a, b):
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def process_uploaded_file(uploaded_file):
    try:
        # Try different encodings if utf-8 fails
        encodings = ['utf-8', 'latin1', 'iso-8859-1', 'cp1252']
        df = None
        
        for encoding in encodings:
            try:
                df = pd.read_csv(uploaded_file, header=None, encoding=encoding)
                break
            except UnicodeDecodeError:
                continue
        
        if df is None:
            st.error("Could not read the file with any supported encoding.")
            return []
        
        # Clean the text data
        return [clean_text(str(item)) for item in df[0].dropna().tolist()]
    except Exception as e:
        st.error(f"Error processing file: {str(e)}")
        return []

def main():
    st.set_page_config(
        page_title="Keyword Intent Similarity Analyzer",
        page_icon="🔍",
        layout="wide"
    )
    
    # Initialize session state
    if 'results_df' not in st.session_state:
        st.session_state.results_df = None
    
    st.title("🔍 Keyword Intent Similarity Analyzer")
    st.write("""
    This tool analyzes the semantic similarity between keywords and topics based on their underlying intent.
    You can either enter keywords manually or upload a CSV file.
    """)
    
    # Initialize the analyzer
    analyzer = KeywordIntentAnalyzer()
    
    # File upload option
    st.subheader("Option 1: Upload CSV File")
    uploaded_file = st.file_uploader("Upload a CSV file with keywords/topics in the first column (no header needed)", 
                                   type=['csv'])
    
    # Text input option
    st.subheader("Option 2: Manual Entry")
    text_input = st.text_area(
        "Or enter your keywords/topics (one per line):",
        height=200,
        placeholder="Enter keywords or topics here...\nExample:\nseo tools\nkeyword research\ncontent optimization"
    )
    
    # Process input and analyze
    if st.button("Analyze Similarities"):
        items = []
        
        # Get items from CSV if uploaded
        if uploaded_file is not None:
            items = process_uploaded_file(uploaded_file)
        
        # Add items from text area if any
        if text_input.strip():
            items.extend([clean_text(item) for item in text_input.split('\n') if item.strip()])
        
        # Remove duplicates while preserving order
        items = list(dict.fromkeys([item for item in items if item]))
        
        if len(items) < 2:
            st.error("Please enter at least 2 keywords/topics to compare.")
        else:
            with st.spinner("Calculating similarities..."):
                # Calculate similarities and store in session state
                df = analyzer.calculate_similarities(items)
                st.session_state.results_df = df
    
    # Only show filter and results if we have data
    if st.session_state.results_df is not None:
        st.subheader("Filter Results (Optional)")
        col1, col2 = st.columns(2)
        with col1:
            min_score = st.number_input("Minimum Similarity Score", 
                                      min_value=0.0, 
                                      max_value=1.0, 
                                      value=0.0, 
                                      step=0.01,
                                      format="%.3f")
        with col2:
            max_score = st.number_input("Maximum Similarity Score", 
                                      min_value=0.0, 
                                      max_value=1.0, 
                                      value=1.0, 
                                      step=0.01,
                                      format="%.3f")

        # Filter the dataframe
        filtered_df = st.session_state.results_df[
            (st.session_state.results_df['Similarity Score'] >= min_score) & 
            (st.session_state.results_df['Similarity Score'] <= max_score)
        ].copy()

        st.subheader("Similarity Results")
        
        if filtered_df.empty:
            st.warning("No results match the selected filter criteria.")
        else:
            # Format similarity scores for display
            filtered_df['Similarity Score'] = filtered_df['Similarity Score'].apply(lambda x: f"{x:.3f}")
            
            # Style the filtered dataframe
            def color_similarity(val):
                # Convert back to float for coloring
                color = f'background-color: rgba(76, 175, 80, {float(val)})'
                return color
            
            styled_df = filtered_df.style.applymap(
                color_similarity,
                subset=['Similarity Score']
            )
            
            st.dataframe(
                styled_df,
                use_container_width=True
            )
            
            # Find highest similarity pair from filtered results
            highest_sim = filtered_df.loc[filtered_df.index[0]]  # Use first row after sorting
            st.success(f"Highest similarity in filtered results found between '{highest_sim['Item 1']}' and '{highest_sim['Item 2']}' with a score of {highest_sim['Similarity Score']}")
            
            # Show number of results after filtering
            st.info(f"Showing {len(filtered_df)} results after filtering")
            
            # Download button for filtered results
            csv = filtered_df.to_csv(index=False)
            st.download_button(
                label="Download Filtered Results as CSV",
                data=csv,
                file_name="filtered_similarity_results.csv",
                mime="text/csv"
            )

if __name__ == "__main__":
    main()
