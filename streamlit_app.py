import streamlit as st
import requests
import json
import os
import base64

# Set the FastAPI endpoint
FASTAPI_URL = "http://localhost:8000"

st.title("Research Agent UI")

# Input form
with st.form("research_form"):
    st.subheader("Research Question")
    research_question = st.text_area("Enter your research question:", height=100)
    
    st.subheader("Options")
    
    # Create tabs for different configuration sections
    tab1, tab2, tab3 = st.tabs(["General Settings", "Zotero Settings", "ArXiv Settings"])
    
    with tab1:
        st.write("### General Settings")
        use_full_text = st.checkbox("Use Full Text", value=False)
        max_papers_to_download = st.number_input("Max Papers to Download", min_value=1, max_value=100, value=10)
        llm_model = st.selectbox(
            "LLM Model",
            ["o4-mini", "o3-mini", "gpt-4", "gpt-3.5-turbo"],
            index=0
        )
    
    with tab2:
        st.write("### Zotero Settings")
        use_zotero = st.checkbox("Use Zotero", value=True)
        if use_zotero:
            zotero_library_id = st.text_input("Zotero Library ID", value="5310176")
            zotero_library_type = st.selectbox(
                "Zotero Library Type",
                ["group", "user"],
                index=0
            )
            zotero_api_key = st.text_input("Zotero API Key", value="91Z1BHYMHJonusqkbBP6hE60", type="password")
            zotero_max_papers = st.number_input("Max Zotero Papers to Use", min_value=1, max_value=100, value=10)
            zotero_collection_keys = st.text_input("Zotero Collection Keys (comma-separated)", value="")
    
    with tab3:
        st.write("### ArXiv Settings")
        use_arxiv = st.checkbox("Use ArXiv", value=True)
        if use_arxiv:
            arxiv_max_results = st.number_input("Max ArXiv Results", min_value=1, max_value=1000, value=300)
            arxiv_max_papers = st.number_input("Max ArXiv Papers to Use", min_value=1, max_value=100, value=100)
    
    submitted = st.form_submit_button("Run Research")

# Display results
if submitted and research_question:
    st.write("Running research...")
    
    # Prepare the request
    request_data = {
        "research_question": research_question,
        "use_zotero": use_zotero,
        "use_arxiv": use_arxiv,
        "use_full_text": use_full_text,
        "max_papers_to_download": max_papers_to_download,
        "llm_model": llm_model,
        "zotero_config": {
            "library_id": zotero_library_id,
            "library_type": zotero_library_type,
            "api_key": zotero_api_key,
            "llm_model": llm_model,
            "max_papers_used": zotero_max_papers,
            "download_dir": "./zotero_downloads",
            "cache_dir": "./paper_cache",
            "local_storage_path": None
        } if use_zotero else None,
        "arxiv_config": {
            "llm_model": llm_model,
            "max_results": arxiv_max_results,
            "max_papers_used": arxiv_max_papers,
            "download_dir": "./arxiv_downloads",
            "cache_dir": "./arxiv_cache"
        } if use_arxiv else None,
        "zotero_collection_keys": [key.strip() for key in zotero_collection_keys.split(",") if key.strip()] if zotero_collection_keys else None
    }
    
    try:
        # Make the API request
        response = requests.post(f"{FASTAPI_URL}/run_research", json=request_data)
        
        if response.status_code == 200:
            results = response.json()
            
            # Display the literature review
            st.subheader("Literature Review")
            st.write(results["literature_review"])
            
            # Display file paths
            st.subheader("Output Files")
            st.write(f"Literature review saved to: {results['file_path']}")
            st.write(f"Results saved to JSON: {results['json_path']}")
            
            # Add download button for the report
            if os.path.exists(results['file_path']):
                with open(results['file_path'], 'rb') as f:
                    report_content = f.read()
                    b64 = base64.b64encode(report_content).decode()
                    href = f'<a href="data:application/octet-stream;base64,{b64}" download="literature_review.txt">Download Literature Review</a>'
                    st.markdown(href, unsafe_allow_html=True)
            
            # Display papers used from each source
            st.subheader("Papers Used")
            if os.path.exists(results['json_path']):
                with open(results['json_path'], 'r') as f:
                    json_results = json.load(f)
                    
                    # Show Zotero papers if used
                    if use_zotero and 'zotero_papers' in json_results:
                        st.write("### Zotero Papers")
                        for paper in json_results['zotero_papers']:
                            with st.expander(f"{paper.get('title', 'Untitled')}"):
                                st.write(f"**Authors:** {paper.get('authors', 'Unknown')}")
                                st.write(f"**Year:** {paper.get('year', 'Unknown')}")
                                st.write(f"**Abstract:** {paper.get('abstract', 'No abstract available')}")
                    
                    # Show ArXiv papers if used
                    if use_arxiv and 'arxiv_papers' in json_results:
                        st.write("### ArXiv Papers")
                        for paper in json_results['arxiv_papers']:
                            with st.expander(f"{paper.get('title', 'Untitled')}"):
                                st.write(f"**Authors:** {paper.get('authors', 'Unknown')}")
                                st.write(f"**Year:** {paper.get('year', 'Unknown')}")
                                st.write(f"**Abstract:** {paper.get('abstract', 'No abstract available')}")
                                st.write(f"**Link:** {paper.get('link', 'No link available')}")
            
        else:
            # Show detailed error message
            error_detail = response.json().get('detail', 'Unknown error occurred')
            st.error(f"Error: {error_detail}")
            st.error(f"Status code: {response.status_code}")
            
    except requests.exceptions.RequestException as e:
        st.error(f"Error connecting to the server: {str(e)}")
        st.error("Make sure the FastAPI server is running at http://localhost:8000")

# Add some styling
st.markdown("""
<style>
    .stTextArea textarea {
        height: 200px;
    }
    .stButton button {
        width: 100%;
    }
    .download-button {
        display: inline-block;
        padding: 0.5rem 1rem;
        background-color: #4CAF50;
        color: white;
        text-decoration: none;
        border-radius: 4px;
        margin-top: 1rem;
    }
    .download-button:hover {
        background-color: #45a049;
    }
</style>
""", unsafe_allow_html=True) 