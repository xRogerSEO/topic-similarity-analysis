import streamlit as st
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from typing import List
import itertools
import io

class KeywordIntentAnalyzer:
    def __init__(self):
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
    
    def calculate_similarities(self, items: List[str]) -> pd.DataFrame:
        # Remove any empty strings and duplicates while preserving order
        items = list(dict.fromkeys([item.strip() for item in items if item.strip()]))
        
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

def process_uploaded_file(uploaded_file):
    try:
        df = pd.read_csv(uploaded_file, header=None)
        return df[0].dropna().tolist()  # Get first column values, drop any NaN values
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
            items.extend([item.strip() for item in text_input.split('\n') if item.strip()])
        
        # Remove duplicates while preserving order
        items = list(dict.fromkeys(items))
        
        if len(items) < 2:
            st.error("Please enter at least 2 keywords/topics to compare.")
        else:
            with st.spinner("Calculating similarities..."):
                # Calculate similarities and store in session state
                st.session_state.results_df = analyzer.calculate_similarities(items)
    
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
            # Style the filtered dataframe
            def color_similarity(val):
                color = f'background-color: rgba(76, 175, 80, {val})'
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
            highest_sim = filtered_df.loc[filtered_df['Similarity Score'].idxmax()]
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
