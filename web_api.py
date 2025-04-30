from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
import os
import sys
import logging
from dotenv import load_dotenv
from src.research_core import ResearchCore, save_results_to_json
from src.models import CustomOpenAI
import json
import markdown
import pdfkit

# Load environment variables from .env file
load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add the src directory to the Python path so we can import research_core
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Get OpenAI API key from environment variable
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY environment variable is not set")

# Initialize the OpenAI client with the API key
# openai_client = CustomOpenAI(api_key=OPENAI_API_KEY)

app = FastAPI()

# Pydantic model for the request body
class ResearchRequest(BaseModel):
    research_question: str
    use_zotero: bool = True
    use_arxiv: bool = True
    use_full_text: bool = False
    max_papers_to_download: int = 10
    llm_model: str = "o4-mini"
    zotero_config: Optional[Dict[str, Any]] = None
    arxiv_config: Optional[Dict[str, Any]] = None
    zotero_collection_keys: Optional[List[str]] = None

# Pydantic model for the response
class ResearchResponse(BaseModel):
    literature_review: str
    file_path: str
    json_path: str
    pdf_path: Optional[str] = None
    zotero_papers: Optional[List[Dict[str, Any]]] = None
    arxiv_papers: Optional[List[Dict[str, Any]]] = None

def convert_md_to_pdf(md_file_path: str) -> Optional[str]:
    """Convert markdown file to PDF if pdfkit is available."""
    try:
        import pdfkit
        pdf_path = md_file_path.replace('.md', '.pdf')
        
        # Convert markdown to HTML
        with open(md_file_path, 'r') as f:
            md_content = f.read()
        html_content = markdown.markdown(md_content)
        
        # Convert HTML to PDF
        pdfkit.from_string(html_content, pdf_path)
        return pdf_path
    except ImportError:
        logger.warning("pdfkit not installed - skipping PDF conversion")
        return None
    except Exception as e:
        logger.error(f"Error converting to PDF: {str(e)}")
        return None
    

@app.post("/run_research", response_model=ResearchResponse)
async def run_research(request: ResearchRequest):
    try:
        logger.info(f"Received research request: {request.research_question}")
        openai_client = CustomOpenAI(api_key=OPENAI_API_KEY, model=request.llm_model)
        # Use provided configs or default values
        zotero_config = request.zotero_config 
        
        arxiv_config = request.arxiv_config 
        
        logger.info("Initializing ResearchCore...")
        # Initialize the research core with both engines
        research_core = ResearchCore(
            llm_model=request.llm_model,
            output_dir='./research_output',
            zotero_config=zotero_config,
            arxiv_config=arxiv_config
        )
        
        # Set the OpenAI client for the research core
        research_core.llm = openai_client
        
        logger.info("Running literature review...")
        # Run the literature review
        results = research_core.run_literature_review(
            research_question=request.research_question,
            use_zotero=request.use_zotero,
            use_arxiv=request.use_arxiv,
            zotero_collection_keys=request.zotero_collection_keys,
            use_full_text=request.use_full_text,
            max_papers_to_download=request.max_papers_to_download
        )
        
        logger.info("Saving results to JSON...")
        # Save results to JSON
        json_path = save_results_to_json(results)
        
        # Load the JSON file to get the papers used
        with open(json_path, 'r') as f:
            json_results = json.load(f)
        
        # Convert markdown to PDF
        pdf_path = convert_md_to_pdf(results['file_path'])
        
        # Extract paper details from the documents
        zotero_papers = []
        arxiv_papers =  []
        
        if 'documents' in results:
            # Create a mapping of paper titles to their order in the literature review
            lit_review = results.get('literature_review', '')
            paper_order = {}
            for i, line in enumerate(lit_review.split('\n')):
                if line.strip().startswith('[') and ']' in line:
                    title = line.split(']', 1)[1].strip()
                    paper_order[title.lower()] = i
            
            # Process documents and maintain order
            for doc in results['documents']:
                # Handle both dictionary and object metadata
                if isinstance(doc, dict):
                    metadata = doc.get('metadata', {})
                else:
                    metadata = getattr(doc, 'metadata', {})
                
                # Extract title and clean it
                title = metadata.get('title', 'Untitled')
                if title.endswith('.'):
                    title = title[:-1]
                
                # Extract year based on source
                year = 'Unknown'
                source = metadata.get('source', '').lower()
                
                if source == 'arxiv':
                    # For ArXiv papers, extract year from ID or published date
                    if 'arxiv_id' in metadata:
                        year = '20' + metadata['arxiv_id'][:2]
                    elif 'published' in metadata:
                        try:
                            year = metadata['published'][:4]
                        except:
                            pass
                elif source == 'zotero':
                    # For Zotero papers, try to get year from various fields
                    if 'date' in metadata:
                        try:
                            year = metadata['date'][:4]
                        except:
                            pass
                    elif 'year' in metadata:
                        year = str(metadata['year'])
                
                # Check if paper has full text
                has_full_text = metadata.get('has_full_text', False)
                
                paper_info = {
                    'title': title,
                    'authors': metadata.get('authors', 'Unknown'),
                    'year': year,
                    'order': paper_order.get(title.lower(), 9999),  # Default to end if not found
                    'has_full_text': has_full_text
                }
                
                # Add to appropriate list based on source
                if source == 'zotero':
                    zotero_papers.append(paper_info)
                elif source == 'arxiv':
                    # Format ArXiv link properly
                    arxiv_id = None
                    if 'arxiv_id' in metadata:
                        arxiv_id = metadata['arxiv_id']
                    elif 'id' in metadata:
                        # Extract ID from the full URL if present
                        id_match = metadata['id'].split('/')[-1]
                        if id_match:
                            arxiv_id = id_match
                    
                    if arxiv_id:
                        # Clean up the ID (remove version number if present)
                        arxiv_id = arxiv_id.split('v')[0]
                        paper_info['link'] = f"https://arxiv.org/abs/{arxiv_id}"
                    else:
                        paper_info['link'] = '#'
                    arxiv_papers.append(paper_info)
                else:
                    # If source is not specified, try to determine from metadata
                    if 'arxiv_id' in metadata or 'id' in metadata:
                        arxiv_id = metadata.get('arxiv_id') or metadata.get('id', '').split('/')[-1].split('v')[0]
                        paper_info['link'] = f"https://arxiv.org/abs/{arxiv_id}" if arxiv_id else '#'
                        arxiv_papers.append(paper_info)
                    else:
                        zotero_papers.append(paper_info)
            
            # Sort papers by their order in the literature review
            arxiv_papers.sort(key=lambda x: x['order'])
            zotero_papers.sort(key=lambda x: x['order'])
            
            # Remove the order field before sending to frontend
            for paper in arxiv_papers + zotero_papers:
                paper.pop('order', None)
        
        logger.info(f"Extracted {len(zotero_papers)} Zotero papers and {len(arxiv_papers)} ArXiv papers")
        
        logger.info("Research completed successfully")
        return ResearchResponse(
            literature_review=results['literature_review'],
            file_path=results['file_path'],
            json_path=json_path,
            pdf_path=pdf_path,
            zotero_papers=zotero_papers if zotero_papers else None,
            arxiv_papers=arxiv_papers if arxiv_papers else None
        )
    except Exception as e:
        logger.error(f"Error processing research request: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 