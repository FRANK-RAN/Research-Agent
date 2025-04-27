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
openai_client = CustomOpenAI(api_key=OPENAI_API_KEY)

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
    zotero_papers: Optional[List[Dict[str, Any]]] = None
    arxiv_papers: Optional[List[Dict[str, Any]]] = None

@app.post("/run_research", response_model=ResearchResponse)
async def run_research(request: ResearchRequest):
    try:
        logger.info(f"Received research request: {request.research_question}")
        
        # Use provided configs or default values
        zotero_config = request.zotero_config or {
            'library_id': '5310176',
            'library_type': 'group',
            'api_key': '91Z1BHYMHJonusqkbBP6hE60',
            'llm_model': request.llm_model,
            'max_papers_used': 10,
            'download_dir': './zotero_downloads',
            'cache_dir': './paper_cache',
            'local_storage_path': None
        }
        
        arxiv_config = request.arxiv_config or {
            'llm_model': request.llm_model,
            'max_results': 300,
            'max_papers_used': 100,
            'download_dir': './arxiv_downloads',
            'cache_dir': './arxiv_cache'
        }
        
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
        
        logger.info("Research completed successfully")
        return ResearchResponse(
            literature_review=results['literature_review'],
            file_path=results['file_path'],
            json_path=json_path,
            zotero_papers=json_results.get('zotero_papers'),
            arxiv_papers=json_results.get('arxiv_papers')
        )
    except Exception as e:
        logger.error(f"Error processing research request: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 