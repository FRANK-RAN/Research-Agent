import streamlit as st
import requests
import json
import os
import base64
import pandas as pd

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
        max_papers_to_download = st.number_input("Max Papers to Download per Source", min_value=1, max_value=100, value=10)
        llm_model = st.selectbox(
            "LLM Model",
            ["o4-mini", "o3-mini", "gpt-4o", "o3"],
            index=0
        )
    
    with tab2:
        st.write("### Zotero Settings")
        use_zotero = st.checkbox("Use Zotero", value=True)
        zotero_library_id = st.text_input("Zotero Library ID", value="000", disabled=not use_zotero)
        zotero_library_type = st.selectbox(
            "Zotero Library Type",
            ["group", "user"],
            index=0,
            disabled=not use_zotero
        )
        zotero_api_key = st.text_input("Zotero API Key", value="1234567890", type="password", disabled=not use_zotero)
        zotero_max_papers = st.number_input("Max Zotero Papers to Use", min_value=1, max_value=100, value=10, disabled=not use_zotero)
    
    with tab3:
        st.write("### ArXiv Settings")
        use_arxiv = st.checkbox("Use ArXiv", value=True)
        arxiv_max_results = st.number_input("Max ArXiv Results Retrieved", min_value=1, max_value=1000, value=300, disabled=not use_arxiv)
        arxiv_max_papers = st.number_input("Max ArXiv Papers to Use", min_value=1, max_value=100, value=100, disabled=not use_arxiv)
    
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
        } if use_arxiv else None
    }
    
    try:
        # Make the API request
        response = requests.post(f"{FASTAPI_URL}/run_research", json=request_data)
        
        if response.status_code == 200:
            results = response.json()
            
            # Display the literature review
            st.subheader("Literature Review")
            st.write(results["literature_review"])
            
            # Add download button for the markdown file
            if os.path.exists(results['file_path']):
                with open(results['file_path'], 'rb') as f:
                    md_content = f.read()
                    b64 = base64.b64encode(md_content).decode()
                    href = f'<a href="data:text/markdown;base64,{b64}" download="literature_review.md">Download Literature Review</a>'
                    st.markdown(href, unsafe_allow_html=True)
            
            # Display papers used from each source
            st.subheader("Papers Used")
            
            # Show Zotero papers if used
            if use_zotero and results.get('zotero_papers'):
                st.write("### Zotero Papers")
                zotero_data = []
                for i, paper in enumerate(results['zotero_papers'], 1):
                    has_full_text = paper.get('has_full_text', False)
                    zotero_data.append({
                        'Paper #': i,
                        'Title': paper.get('title', 'Untitled'),
                        'Authors': paper.get('authors', 'Unknown'),
                        'Year': paper.get('year', 'Unknown'),
                        'Full Text': '✓' if has_full_text else ''
                    })
                if zotero_data:
                    df = pd.DataFrame(zotero_data)
                    st.dataframe(
                        df,
                        hide_index=True,
                        column_config={
                            "Paper #": st.column_config.NumberColumn("Paper #", width="small"),
                            "Title": st.column_config.TextColumn("Title", width="large"),
                            "Authors": st.column_config.TextColumn("Authors", width="medium"),
                            "Year": st.column_config.TextColumn("Year", width="small"),
                            "Full Text": st.column_config.TextColumn("Full Text", width="small")
                        },
                        use_container_width=True
                    )
            
            # Show ArXiv papers if used
            if use_arxiv and results.get('arxiv_papers'):
                st.write("### ArXiv Papers")
                arxiv_data = []
                # Start numbering from the last Zotero paper number + 1, or 1 if no Zotero papers
                start_number = len(results.get('zotero_papers', [])) + 1 if use_zotero and results.get('zotero_papers') else 1
                for i, paper in enumerate(results['arxiv_papers'], start_number):
                    link = paper.get('link', '#')
                    has_full_text = paper.get('has_full_text', False)
                    arxiv_data.append({
                        'Paper #': i,
                        'Title': paper.get('title', 'Untitled'),
                        'Authors': paper.get('authors', 'Unknown'),
                        'Year': paper.get('year', 'Unknown'),
                        'Full Text': '✓' if has_full_text else '',
                        'Link': link if link != '#' else '-'
                    })
                if arxiv_data:
                    df = pd.DataFrame(arxiv_data)
                    st.dataframe(
                        df,
                        hide_index=True,
                        column_config={
                            "Paper #": st.column_config.NumberColumn("Paper #", width="small"),
                            "Title": st.column_config.TextColumn("Title", width="large"),
                            "Authors": st.column_config.TextColumn("Authors", width="medium"),
                            "Year": st.column_config.TextColumn("Year", width="small"),
                            "Full Text": st.column_config.TextColumn("Full Text", width="small"),
                            "Link": st.column_config.LinkColumn(
                                "Link",
                                width="small",
                                display_text="View Paper"
                            )
                        },
                        use_container_width=True
                    )
            
            # Add note about full text papers
            st.markdown("*Note: Papers marked with ✓ in the 'Full Text' column were processed with full text analysis.*")
            
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