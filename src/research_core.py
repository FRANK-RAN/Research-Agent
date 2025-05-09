import os
import json
import time
import concurrent.futures
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
from llama_index.core.schema import Document
# from llama_index.llms.openai import OpenAI
from src.models import CustomOpenAI as OpenAI
import concurrent.futures

# Import the Zotero and ArXiv engines
from src.zotero_engine import ZoteroRAG
from src.arxiv_engine import ArxivRAG


class ResearchCore:
    """
    Core research agent for literature review generation.
    
    This system:
    1. Retrieves papers from Zotero and/or ArXiv
    2. Extracts relevant information from full-text papers in parallel
    3. Prepares a prompt using papers and extractions
    4. Generates a comprehensive literature review
    """
    
    def __init__(
        self, 
        llm_model: str = "o4-mini",
        output_dir: str = "./research_output",
        zotero_config: Optional[Dict[str, Any]] = None,
        arxiv_config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize the Research Core system.
        
        Args:
            llm_model: The OpenAI model to use
            output_dir: Directory to save output reports
            zotero_config: Configuration for Zotero engine (None to disable)
            arxiv_config: Configuration for ArXiv engine (None to disable)
        """
        self.llm = OpenAI(model=llm_model)
        self.output_dir = output_dir
        
        # Create output directory if it doesn't exist
        if not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
        
        # Initialize engines if configured
        self.zotero_engine = None
        self.arxiv_engine = None
        
        if zotero_config:
            self.zotero_engine = ZoteroRAG(**zotero_config)
        
        if arxiv_config:
            self.arxiv_engine = ArxivRAG(**arxiv_config)
            
        if not self.zotero_engine and not self.arxiv_engine:
            raise ValueError("At least one of Zotero or ArXiv engines must be configured")
        
        # Keep track of research history
        self.research_history = []



    def retrieve_papers(
            self, 
            research_question: str, 
            use_zotero: bool = True,
            use_arxiv: bool = True,
            zotero_collection_names: List[str] = None,
            use_full_text: bool = True,
            max_papers_to_download: int = 3
        ) -> Tuple[List[Document], Dict[str, Any]]:
            """
            Retrieve papers from Zotero and/or ArXiv based on the research question.
            Uses parallel processing to fetch from both sources simultaneously.
            
            Args:
                research_question: The research question to answer
                use_zotero: Whether to retrieve papers from Zotero
                use_arxiv: Whether to retrieve papers from ArXiv
                zotero_collection_names: Specific Zotero collection names to search (if None, searches entire library)
                use_full_text: Whether to download and process full-text papers
                max_papers_to_download: Maximum number of papers to download for full-text analysis
                
            Returns:
                Tuple of (combined list of documents, retrieval stats)
            """
            print(f"\n[RESEARCH CORE] Retrieving papers for research question: '{research_question}'")
            start_time = time.time()
            
            all_documents = []
            retrieval_stats = {
                "research_question": research_question,
                "zotero_used": use_zotero,
                "arxiv_used": use_arxiv,
                "full_text_used": use_full_text,
                "total_documents": 0,
                "documents_with_full_text": 0,
                "zotero_documents": 0,
                "arxiv_documents": 0,
                "retrieval_time": 0
            }
            
            # Define retrieval functions for each source
            def retrieve_from_zotero():
                if not use_zotero or not self.zotero_engine:
                    return []
                    
                print(f"[RESEARCH CORE] Retrieving papers from Zotero")
                zotero_result = self.zotero_engine.get_papers_for_research(
                    research_question=research_question,
                    collection_names=zotero_collection_names,
                    use_full_text=use_full_text,
                    max_papers_to_download=max_papers_to_download
                )
                
                zotero_documents = zotero_result["documents"]
                
                # Add source metadata to documents
                for doc in zotero_documents:
                    doc.metadata["source"] = "zotero"
                
                print(f"[RESEARCH CORE] Retrieved {len(zotero_documents)} documents from Zotero")
                return zotero_documents
            
            def retrieve_from_arxiv():
                if not use_arxiv or not self.arxiv_engine:
                    return []
                    
                print(f"[RESEARCH CORE] Retrieving papers from ArXiv")
                arxiv_result = self.arxiv_engine.get_papers_for_research(
                    research_question=research_question,
                    use_llm_query=True,
                    use_full_text=use_full_text,
                    max_papers_to_download=max_papers_to_download
                )
                
                if "documents" in arxiv_result:
                    arxiv_documents = arxiv_result["documents"]
                    
                    # Add source metadata to documents
                    for doc in arxiv_documents:
                        doc.metadata["source"] = "arxiv"
                    
                    print(f"[RESEARCH CORE] Retrieved {len(arxiv_documents)} documents from ArXiv")
                    return arxiv_documents
                else:
                    print(f"[RESEARCH CORE] No documents retrieved from ArXiv")
                    return []
            
            # Execute both retrieval operations in parallel
            with concurrent.futures.ThreadPoolExecutor() as executor:
                # Submit both retrieval tasks
                zotero_future = executor.submit(retrieve_from_zotero)
                arxiv_future = executor.submit(retrieve_from_arxiv)
                
                # Collect results as they complete
                zotero_documents = zotero_future.result()
                arxiv_documents = arxiv_future.result()
                
                # Update documents and stats
                all_documents.extend(zotero_documents)
                all_documents.extend(arxiv_documents)
                retrieval_stats["zotero_documents"] = len(zotero_documents)
                retrieval_stats["arxiv_documents"] = len(arxiv_documents)
            
            # Update statistics
            retrieval_stats["total_documents"] = len(all_documents)
            retrieval_stats["documents_with_full_text"] = sum(1 for doc in all_documents if doc.metadata.get("has_full_text", False))
            retrieval_stats["retrieval_time"] = time.time() - start_time
            
            print(f"[RESEARCH CORE] Retrieved {retrieval_stats['total_documents']} total documents ({retrieval_stats['documents_with_full_text']} with full text)")
            print(f"[RESEARCH CORE] Retrieval completed in {retrieval_stats['retrieval_time']:.2f} seconds")
            
            return all_documents, retrieval_stats
    
    def get_full_text_papers(self, documents: List[Document]) -> List[int]:
        """
        Get all papers that have full text available.
        
        Args:
            documents: List of Document objects
            
        Returns:
            List of indices for documents that have full text
        """
        print(f"\n[EXTRACTION] Identifying papers with full text")
        
        # Filter papers with full text
        full_text_indices = [i for i, doc in enumerate(documents) if doc.metadata.get("has_full_text", False)]
        
        if not full_text_indices:
            print(f"[EXTRACTION] No documents with full text available for extraction")
        else:
            print(f"[EXTRACTION] Found {len(full_text_indices)} documents with full text available for extraction")
        
        return full_text_indices
    
    def extract_from_paper(self, research_question: str, doc: Document) -> Document:
        """
        Extract detailed information from a full-text paper relevant to the research question.
        
        Args:
            research_question: The research question
            doc: Document object with full text
            
        Returns:
            Updated Document object with extraction field
        """
        title = doc.metadata.get("title", "Untitled")
        print(f"[EXTRACTION] Extracting information from paper: {title}")
        
        # Create prompt for extraction
        prompt = f"""
        You are a research assistant extracting key information from a scientific paper to help answer a research question.
        
        RESEARCH QUESTION: {research_question}
        
        PAPER:
        {doc.text}
        
        INSTRUCTIONS:
        Create a comprehensive summary of this paper's content that is relevant to the research question.
        Focus on:
        - The main findings, results, or measurements related to the research question
        - Key methodologies or techniques used
        - Any specific numerical data, values, or parameters that address the question
        - Any limitations or challenges mentioned
        
        Provide a detailed and thorough extraction of all relevant information that will be most useful 
        for answering the research question. Include specific numbers, values, and details whenever available.
        
        Only return the summary extraction text with no additional formatting or structure.
        """
        
        start_time = time.time()
        extraction = self.llm.complete(prompt).text.strip()
        
        # Create a new document that includes the extracted information
        # Keep the original text and metadata, but add the extraction
        # for use in the literature review
        updated_doc = Document(text=doc.text)
        updated_doc.metadata = doc.metadata.copy()
        updated_doc.metadata["extraction"] = extraction
        
        extraction_time = time.time() - start_time
        print(f"[EXTRACTION] Extracted information from paper in {extraction_time:.2f} seconds")
        
        return updated_doc
    
    def extract_from_papers_parallel(self, research_question: str, documents: List[Document], full_text_indices: List[int]) -> List[Document]:
        """
        Extract information from multiple papers in parallel and update the documents with extractions.
        
        Args:
            research_question: The research question
            documents: List of Document objects
            full_text_indices: Indices of documents with full text to extract from
            
        Returns:
            Updated list of Document objects with extractions added
        """
        print(f"\n[PARALLEL EXTRACTION] Extracting information from {len(full_text_indices)} papers with full text in parallel")
        start_time = time.time()
        
        updated_documents = documents.copy()  # Create a copy to avoid modifying original list
        
        # Use ThreadPoolExecutor for parallel processing
        with concurrent.futures.ThreadPoolExecutor() as executor:
            # Submit extraction tasks
            future_to_index = {
                executor.submit(self.extract_from_paper, research_question, documents[idx]): idx
                for idx in full_text_indices
            }
            
            # Process results as they complete
            for future in concurrent.futures.as_completed(future_to_index):
                idx = future_to_index[future]
                try:
                    # Get the updated document with extraction added
                    updated_doc = future.result()
                    updated_documents[idx] = updated_doc
                    print(f"[PARALLEL EXTRACTION] Completed extraction for document {idx+1}/{len(full_text_indices)}")
                except Exception as e:
                    print(f"[PARALLEL EXTRACTION] Error extracting from document {idx}: {str(e)}")
        
        total_time = time.time() - start_time
        print(f"[PARALLEL EXTRACTION] Completed extraction for {len(full_text_indices)} papers in {total_time:.2f} seconds")
        
        return updated_documents
    
    def generate_literature_review(
        self, 
        research_question: str, 
        documents: List[Document]
    ) -> str:
        """
        Generate a comprehensive literature review based on the papers and their extractions.
        
        Args:
            research_question: The research question
            documents: List of Document objects (some with extraction field in metadata)
            
        Returns:
            The generated literature review text
        """
        print(f"\n[LITERATURE REVIEW] Generating literature review for: '{research_question}'")
        start_time = time.time()
        
        # Format papers for the prompt
        papers_context = ""
        
        # Process each paper
        for i, doc in enumerate(documents):
            has_full_text = doc.metadata.get("has_full_text", False)
            has_extraction = "extraction" in doc.metadata
            
            papers_context += f"\nPaper {i+1}"
            
            # For papers with extraction, use the extraction instead of full text
            if has_extraction:
                papers_context += f" (with full text extraction):\n"
                papers_context += f"Title: {doc.metadata.get('title', 'Untitled')}\n"
                papers_context += f"Authors: {doc.metadata.get('authors', 'Unknown')}\n"
                papers_context += f"Source: {doc.metadata.get('source', 'Unknown')}\n"
                papers_context += f"ID: {doc.metadata.get('id', doc.metadata.get('zotero_key', 'Unknown'))}\n"
                papers_context += f"EXTRACTION:\n{doc.metadata['extraction']}\n\n"
            else:
                # For papers without extraction, use the original text
                papers_context += f" ({('with full text' if has_full_text else 'abstract only')}):\n"
                papers_context += f"{doc.text}\n\n"
        
        # Create prompt for literature review
        prompt = f"""
        You are a research assistant tasked with creating a comprehensive literature review based on the following research question and papers.
        
        RESEARCH QUESTION: {research_question}
        
        PAPERS:
        {papers_context}
        
        INSTRUCTIONS:
        Create a detailed literature review that synthesizes information across these papers to address the research question.
        
        Your literature review should:
        1. Provide a comprehensive answer to the research question with proper in-text citations 
        2. Include specific data, measurements, and values when available
        3. Compare different methodologies and their effectiveness
        4. Analyze similarities, differences, and contradictions in the literature
        5. Identify knowledge gaps and suggest future research directions
        6. Conclude with direct answers to the research question based on the available evidence
        
        When citing papers, use the format [Paper X] where X is the paper number.
        
        Include a "REFERENCES" section at the end that lists all papers cited, including titles and authors.
        Format each reference as:
        [X] Authors. "Title"
        """
        
        # Generate the literature review
        literature_review = self.llm.complete(prompt).text
        
        total_time = time.time() - start_time
        print(f"[LITERATURE REVIEW] Generated literature review in {total_time:.2f} seconds")
        
        return literature_review
    
    def save_literature_review(self, research_question: str, literature_review: str, stats: Dict[str, Any]) -> str:
        """
        Save the literature review to a markdown file.
        
        Args:
            research_question: The research question
            literature_review: The generated literature review text
            stats: Statistics about the papers retrieved
            
        Returns:
            Path to the saved file
        """
        print(f"\n[SAVE] Saving literature review")
        
        # Create a markdown file with the research question as title
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"literature_review_{timestamp}.md"
        filepath = os.path.join(self.output_dir, filename)
        
        # Add some metadata at the top
        content = f"# Literature Review: {research_question}\n\n"
        content += f"**Generated on:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
        content += f"**Papers analyzed:** {stats['total_documents']} "
        content += f"({stats['documents_with_full_text']} with full text)\n\n"
        
        if stats['zotero_used'] and stats['zotero_documents'] > 0:
            content += f"**Zotero papers:** {stats['zotero_documents']}\n\n"
        
        if stats['arxiv_used'] and stats['arxiv_documents'] > 0:
            content += f"**ArXiv papers:** {stats['arxiv_documents']}\n\n"
        
        content += "---\n\n"
        
        # Add the literature review content
        content += literature_review
        
        # Save to file
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)
        
        print(f"[SAVE] Literature review saved to {filepath}")
        return filepath
    
    def run_literature_review(
        self,
        research_question: str,
        use_zotero: bool = True,
        use_arxiv: bool = True,
        zotero_collection_names: List[str] = None,
        use_full_text: bool = True,
        max_papers_to_download: int = 3
    ) -> Dict[str, Any]:
        """
        Run a complete literature review process from start to finish.
        
        Args:
            research_question: The research question to answer
            use_zotero: Whether to retrieve papers from Zotero
            use_arxiv: Whether to retrieve papers from ArXiv
            zotero_collection_names: Specific Zotero collection names to search (if None, searches entire library)
            use_full_text: Whether to download and process full-text papers
            max_papers_to_download: Maximum number of papers to download for full-text analysis
            
        Returns:
            Dictionary with literature review results
        """
        print(f"\n[RESEARCH CORE] Starting literature review process for: '{research_question}'")
        start_time = time.time()
        
        # Step 1: Retrieve papers from sources
        documents, retrieval_stats = self.retrieve_papers(
            research_question=research_question,
            use_zotero=use_zotero,
            use_arxiv=use_arxiv,
            zotero_collection_names=zotero_collection_names,
            use_full_text=use_full_text,
            max_papers_to_download=max_papers_to_download
        )
        
        if not documents:
            print(f"[RESEARCH CORE] No documents retrieved. Cannot proceed with literature review.")
            return {
                "error": "No documents retrieved",
                "research_question": research_question,
                "retrieval_stats": retrieval_stats
            }
        
        # Step 2: Identify full-text papers for extraction
        full_text_indices = self.get_full_text_papers(documents)
        
        # Step 3: Extract information from full-text papers in parallel
        if full_text_indices:
            documents = self.extract_from_papers_parallel(
                research_question=research_question,
                documents=documents,
                full_text_indices=full_text_indices
            )
        
        # Step 4: Generate the literature review using the papers with extractions
        literature_review = self.generate_literature_review(
            research_question=research_question,
            documents=documents
        )
        
        # Step 5: Save the literature review to a file
        filepath = self.save_literature_review(
            research_question=research_question,
            literature_review=literature_review,
            stats=retrieval_stats
        )
        
        # Calculate total time
        total_time = time.time() - start_time
        print(f"[RESEARCH CORE] Completed literature review in {total_time:.2f} seconds")
        
        # Return results
        return {
            "research_question": research_question,
            "literature_review": literature_review,
            "file_path": filepath,
            "documents": documents,
            "retrieval_stats": retrieval_stats,
            "total_time": total_time
        }


def save_results_to_json(results, output_dir="./json_output"):
    """
    Save research results to a JSON file
    
    Args:
        results: Dictionary containing the research results
        output_dir: Directory to save the JSON file
    
    Returns:
        Path to the saved JSON file
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    question_snippet = results["research_question"][:30].replace(" ", "_")
    filename = f"{timestamp}_{question_snippet}.json"
    filepath = os.path.join(output_dir, filename)
    
    # Extract relevant data for saving
    data_to_save = {
        "research_question": results["research_question"],
        "literature_review": results["literature_review"],
        "retrieval_stats": results["retrieval_stats"],
        "total_time": results["total_time"]
    }
    
    # Save the data to JSON file
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data_to_save, f, indent=2, ensure_ascii=False)
    
    print(f"[RESEARCH CORE] Saved results to {filepath}")
    return filepath

