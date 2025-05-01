import time
import os
import json
import pickle
import hashlib
from typing import List, Dict, Any, Optional
import arxiv
from datetime import datetime
# from llama_index.llms.openai import OpenAI
from src.models import CustomOpenAI as OpenAI
from llama_index.core.schema import Document
from llama_parse import LlamaParse
from llama_index.core import SimpleDirectoryReader
import concurrent.futures

class ArxivRAG:
    """
    A RAG system for arXiv papers.
    
    This system:
    1. Uses an LLM to generate optimized ArXiv search queries
    2. Retrieves relevant papers from ArXiv
    3. Uses an LLM to screen papers for relevance
    4. Downloads and processes PDFs for deeper analysis
    5. Prepares prompts with the papers for research question answering
    """
    
    def __init__(
        self, 
        llm_model: str = "o4-mini", 
        max_results: int = 100,
        max_papers_used: int = 30,
        download_dir: str = "./arxiv_downloads",
        cache_dir: str = "./arxiv_cache"
    ):
        """
        Initialize the ArXiv RAG system.
        
        Args:
            llm_model: The OpenAI model to use
            max_results: Maximum number of ArXiv results to retrieve
            max_papers_used: Maximum number of papers to include in the context
            download_dir: Directory to save downloaded PDFs
            cache_dir: Directory to cache parsed papers
        """
        self.llm = OpenAI(model=llm_model)
        self.max_results = max_results
        self.max_papers_used = max_papers_used
        self.download_dir = download_dir
        self.cache_dir = cache_dir
        self.search_history = []
        
        # Create required directories if they don't exist
        for directory in [download_dir, cache_dir]:
            if not os.path.exists(directory):
                os.makedirs(directory, exist_ok=True)
    
    def generate_search_query(self, research_question: str) -> str:
        """
        Generate an optimized ArXiv search query from a research question.
        
        Args:
            research_question: The research question to transform
            
        Returns:
            A formatted ArXiv search query
        """
        print(f"\n[QUERY GENERATION] Analyzing research question: '{research_question}'")
        
        prompt = f"""
        Generate an optimized ArXiv search query for the following research question. 
        The query should follow ArXiv's search syntax with proper operators and field specifiers.
        
        Guidelines for ArXiv search query formatting:
        - Use field specifiers like 'ti:' for title, 'au:' for author, 'abs:' for abstract, 'all' for all fields''
        - Use Boolean operators: AND, OR, ANDNOT (all caps)
        - Group terms with parentheses for complex queries
        - Use quotes for exact phrases: "quantum computing"
        - For author searches, format as 'au:lastname_f' or 'au:"lastname, firstname"'
        - Include category specifiers if appropriate: cat:cs.AI
        
        Research question: "{research_question}"
        
        Return ONLY the formatted ArXiv query string without any additional text.
        """
        
        search_query = self.llm.complete(prompt).text.strip()
        
        # Clean up the query - remove any markdown formatting if present
        search_query = search_query.replace('`', '').strip()
        
        print(f"[QUERY GENERATION] Generated ArXiv query: '{search_query}'")
        return search_query
    
    def retrieve_papers(
        self, 
        query: str, 
        research_question: str,
        sort_by: arxiv.SortCriterion = arxiv.SortCriterion.Relevance,
        max_retry: int = 2
    ) -> List[arxiv.Result]:
        """
        Retrieve papers from ArXiv using the provided query.
        
        Args:
            query: The ArXiv search query
            research_question: The original research question (for query refinement)
            sort_by: How to sort the results
            max_retry: Maximum number of retry attempts with refined queries
            
        Returns:
            A list of arxiv.Result objects
        """
        print(f"\n[ARXIV SEARCH] Executing query: '{query}'")
        start_time = time.time()
        
        search = arxiv.Search(
            query=query,
            max_results=self.max_results,
            sort_by=sort_by,
            sort_order=arxiv.SortOrder.Descending
        )
        
        client = arxiv.Client()
        results = list(client.results(search))
        
        # If no results found, try using LLM to generate a better query
        if not results and max_retry > 0:
            print(f"[ARXIV SEARCH] No results found for query: '{query}'. Attempting to generate a better query...")
            
            new_query = self._generate_new_query(query, research_question)
            if new_query != query:
                print(f"[ARXIV SEARCH] Generated new query: '{new_query}'")
                return self.retrieve_papers(new_query, research_question, sort_by, max_retry - 1)
            else:
                print(f"[ARXIV SEARCH] Could not generate a different query. Using very basic fallback...")
                # Last resort - use a few keywords from the research question
                fallback_query = ' '.join(research_question.split()[:3])
                print(f"[ARXIV SEARCH] Using fallback query: '{fallback_query}'")
                return self.retrieve_papers(fallback_query, research_question, sort_by, max_retry - 1)
        
        execution_time = time.time() - start_time
        print(f"[ARXIV SEARCH] Found {len(results)} papers in {execution_time:.2f} seconds")
        
        # Display results
        papers_info = []
        for i, paper in enumerate(results):
            paper_info = {
                "title": paper.title,
                "authors": ", ".join(author.name for author in paper.authors),
                "id": paper.get_short_id(),
                "url": paper.entry_id,
                "published": paper.published.strftime('%Y-%m-%d') if paper.published else "Unknown",
                "categories": ", ".join(paper.categories),
                "summary_snippet": paper.summary
            }
            papers_info.append(paper_info)
            
            if i < 5:  # Show first 5
                print(f"[ARXIV PAPER {i+1}] {paper.title} (ID: {paper.get_short_id()})")
        
        if len(results) > 5:
            print(f"[ARXIV SEARCH] ...and {len(results)-5} more papers")
        
        # Log the search
        self.search_history.append({
            "query": query,
            "timestamp": time.ctime(),
            "execution_time": execution_time,
            "num_results": len(results),
            "papers": papers_info
        })
        
        return results
    
    def _generate_new_query(self, failed_query: str, research_question: str) -> str:
        """
        Generate a new ArXiv query when the original one failed to return results.
        
        Args:
            failed_query: The original query that returned no results
            research_question: The original research question
            
        Returns:
            A new query with a different approach
        """
        print(f"[QUERY REFINEMENT] Generating new query for research question: '{research_question}'")
        
        prompt = f"""
        The following ArXiv search query returned NO RESULTS:
        
        Query: "{failed_query}"
        
        For the research question: "{research_question}"
        
        Please generate a completely new ArXiv search query that:
        1. Uses a different approach to search for relevant papers
        2. Uses more general or standard terminology
        3. Avoids specialized operators if they were used previously
        4. Focuses on broader concepts that are likely to have papers in the ArXiv database
        
        IMPORTANT: The new query should be significantly different from the original failed query.
        
        Return ONLY the new query string without any explanation.
        """
        
        new_query = self.llm.complete(prompt).text.strip()
        # Clean up the query - remove any markdown formatting if present
        new_query = new_query.replace('`', '').strip()
        
        print(f"[QUERY REFINEMENT] Generated new query: '{new_query}'")
        return new_query
    
    def process_papers(self, papers: List[arxiv.Result]) -> List[Document]:
        """
        Process ArXiv papers into Documents for the RAG system.
        
        Args:
            papers: List of arxiv.Result objects
            
        Returns:
            List of Document objects
        """
        print(f"\n[PAPER PROCESSING] Processing {len(papers)} papers for RAG")
        
        documents = []
        for paper in papers:
            # Extract metadata
            authors_str = ", ".join(author.name for author in paper.authors)
            paper_id = paper.get_short_id()
            published = paper.published.strftime('%Y-%m-%d') if paper.published else "Unknown"
            categories = ", ".join(paper.categories)
            
            # Create document text with structured information
            doc_text = f"Title: {paper.title}\n"
            doc_text += f"Authors: {authors_str}\n"
            doc_text += f"ID: {paper_id}\n"
            doc_text += f"Published: {published}\n"
            doc_text += f"Categories: {categories}\n"
            doc_text += f"Summary: {paper.summary}\n"
            
            # Create document
            doc = Document(text=doc_text)
            
            # Add metadata
            doc.metadata = {
                "title": paper.title,
                "authors": authors_str,
                "id": paper_id,
                "url": paper.entry_id,
                "published": published,
                "categories": categories,
                "has_full_text": False
            }
            
            documents.append(doc)
        
        print(f"[PAPER PROCESSING] Processed {len(documents)} documents")
        return documents
    
    def screen_items_with_llm(self, research_question: str, documents: List[Document], max_items: int = None) -> tuple[List[Document], List[int]]:
        """
        Use LLM to screen items for relevance to the research question.
        
        Args:
            research_question: The research question
            documents: List of Document objects
            max_items: Maximum number of items after screening to return (default: self.max_papers_used)
            
        Returns:
             Tuple containing (selected_documents, selected_indices)
        """
        if max_items is None:
            max_items = self.max_papers_used
            
        print(f"\n[SCREENING] Screening {len(documents)} documents for relevance to: '{research_question}'")
        
            
        # Format documents as context
        docs_context = ""
        for i, doc in enumerate(documents):
            docs_context += f"Document {i+1}:\n{doc.text}\n\n"
            
        # Create prompt for screening
        prompt = f"""
        You are a research assistant helping to identify relevant papers for a literature review.
        
        RESEARCH QUESTION: {research_question}
        
        Below are summaries of academic papers. Review each paper and determine its relevance to the research question.
        Select ONLY the relevant papers for this research question, up to a maximum of {max_items} papers.
        
        DOCUMENT SUMMARIES:
        {docs_context}
        
        INSTRUCTIONS:
        Analyze each document's relevance to the research question.
        Consider: topic match, recency, methodological relevance, theoretical alignment, and citation impact.
        
        IMPORTANT: Only include papers that are truly relevant. You do not need to select {max_items} papers if fewer are relevant.
        
        RESPOND WITH:
        1. The document numbers of the relevant papers (up to {max_items}) in a JSON array format, ordered by relevance.
        2. ONLY return a JSON object with this format: {{"relevant_docs": [1, 5, 10, ...]}}
        """
            
        print(f"[SCREENING] Running LLM screening...")
        start_time = time.time()
            
        response = self.llm.complete(prompt).text
            
        # Parse the response
        relevant_docs = []
        try:
            # Try to parse as JSON
            import json
            response_json = json.loads(response)
            if 'relevant_docs' in response_json and isinstance(response_json['relevant_docs'], list):
                relevant_docs = response_json['relevant_docs']
        except:
            # Fallback: try to extract numbers using regex
            import re
            numbers = re.findall(r'\d+', response)
            relevant_docs = [int(num) for num in numbers if 1 <= int(num) <= len(documents)]
            
        # Ensure we have valid document indices
        valid_indices = [i-1 for i in relevant_docs if 1 <= i <= len(documents)]
            
        # If we couldn't extract valid indices, take the first max_items
        if not valid_indices:
            print(f"[SCREENING] Could not extract valid document indices from LLM response, using first {max_items} documents")
            valid_indices = list(range(min(max_items, len(documents))))
            
        # Get the relevant documents
        screened_documents = [documents[i] for i in valid_indices]
            
        screening_time = time.time() - start_time
        print(f"[SCREENING] Selected {len(screened_documents)} relevant documents in {screening_time:.2f} seconds")
            
        return screened_documents, valid_indices
    
    def _get_cache_path(self, paper_id: str) -> str:
        """Get the path for cached parsed paper using the same naming convention as PDF downloads"""
        # Replace dots with underscores, just like in the download_pdf method
        filename = f"{paper_id.replace('.', '_')}.pkl"
        return os.path.join(self.cache_dir, filename)

    def _is_paper_cached(self, paper_id: str) -> bool:
        """Check if paper is already parsed and cached"""
        cache_path = self._get_cache_path(paper_id)
        return os.path.exists(cache_path)

    def _save_parsed_paper(self, paper_id: str, full_text: str) -> None:
        """Save parsed paper to cache"""
        cache_path = self._get_cache_path(paper_id)
        try:
            with open(cache_path, 'wb') as f:
                # Save both the paper ID and the full text to verify correctness
                data_to_save = {
                    'paper_id': paper_id,
                    'full_text': full_text,
                    'timestamp': datetime.now().isoformat()
                }
                pickle.dump(data_to_save, f)
            print(f"[CACHE] Saved parsed paper to cache: {cache_path}")
        except Exception as e:
            print(f"[CACHE] Error saving to cache: {str(e)}")

    def _load_parsed_paper(self, paper_id: str) -> Optional[str]:
        """Load parsed paper from cache"""
        cache_path = self._get_cache_path(paper_id)
        try:
            with open(cache_path, 'rb') as f:
                data = pickle.load(f)
                # New format with metadata
                full_text = data['full_text']
                loaded_id = data['paper_id']
                
                if loaded_id != paper_id:
                    print(f"[CACHE] WARNING: Loaded paper ID ({loaded_id}) doesn't match requested ID ({paper_id})")
                
                print(f"[CACHE] Loaded parsed paper from cache: {cache_path}")
                    
                return full_text
        except Exception as e:
            print(f"[CACHE] Error loading from cache: {str(e)}")
            return None
    
    def download_pdf(self, paper: arxiv.Result) -> str:
        """
        Download a PDF of a paper from ArXiv.
        
        Args:
            paper: arxiv.Result object
            
        Returns:
            Path to the downloaded PDF
        """
        paper_id = paper.get_short_id()
        filename = f"{paper_id.replace('.', '_')}.pdf"
        pdf_path = os.path.join(self.download_dir, filename)
        
        # Skip if already downloaded
        if os.path.exists(pdf_path):
            print(f"[PDF DOWNLOAD] Paper {paper_id} already downloaded")
            return pdf_path
        
        # Download the PDF
        print(f"[PDF DOWNLOAD] Downloading paper {paper_id}...")
        try:
            paper.download_pdf(dirpath=self.download_dir, filename=filename)
            print(f"[PDF DOWNLOAD] Successfully downloaded paper {paper_id} to {pdf_path}")
            return pdf_path
        except Exception as e:
            print(f"[PDF DOWNLOAD] Error downloading paper {paper_id}: {str(e)}")
            return None
    
    def process_pdf_with_llamaparse(self, pdf_path: str) -> str:
        """
        Process a PDF with LlamaParse.
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            Parsed text from the PDF
        """
        print(f"[PDF PARSING] Parsing PDF {pdf_path}...")
        
        try:
            # Set up the PDF parser
            parser = LlamaParse(
                result_type="text",
                num_workers=8,
                page_prefix="START OF PAGE: {pageNumber}\n",
                page_suffix="\nEND OF PAGE: {pageNumber}",
            )
            
            # Use SimpleDirectoryReader to read and parse the file
            file_extractor = {".pdf": parser}
            documents = SimpleDirectoryReader(
                input_files=[pdf_path],
                file_extractor=file_extractor
            ).load_data()
            
            # Combine the documents into a single text
            full_text = "\n\n".join([doc.text for doc in documents])
            print(f"[PDF PARSING] Successfully parsed PDF with {len(documents)} sections")
            
            return full_text
        except Exception as e:
            print(f"[PDF PARSING] Error parsing PDF: {str(e)}")
            return f"Error parsing PDF: {str(e)}"
    
    def select_documents_for_full_text(self, research_question: str, documents: List[Document], max_papers_to_download: int = 3) -> List[int]:
        """
        Use LLM to select which documents should have their full text retrieved.
        
        Args:
            research_question: The research question
            documents: List of Document objects
            max_papers_to_download: Maximum number of papers to download full text for
            
        Returns:
            List of indices for documents that should have full text retrieved
        """
        print(f"\n[FULL TEXT SELECTION] Selecting documents for full text retrieval (up to {max_papers_to_download})")
        
        if len(documents) <= max_papers_to_download:
            print(f"[FULL TEXT SELECTION] Number of documents ({len(documents)}) is less than or equal to max papers to download ({max_papers_to_download}), selecting all")
            return list(range(len(documents)))
        
        # Format documents as context
        docs_context = ""
        for i, doc in enumerate(documents):
            docs_context += f"Document {i+1}:\n{doc.text}\n\n"
        
        # Create prompt for selection
        prompt = f"""
        You are a research assistant helping to identify which papers need full text analysis to best answer a research question.
        
        RESEARCH QUESTION: {research_question}
        
        Below are summaries of academic papers with titles, authors, and abstracts. Your task is to determine which papers would 
        most benefit from full text analysis (downloading and processing the entire PDF) to better answer the research question.
        
        DOCUMENT SUMMARIES:
        {docs_context}
        
        INSTRUCTIONS:
        Analyze each document's potential value for answering the research question if we had its full text.
        Consider:
        1. Relevance to the specific research question
        2. Papers that likely contain detailed methodologies, specific measurements, or experimental setups related to the question
        3. Papers where the abstract suggests valuable content but lacks sufficient detail
        4. Papers that represent diverse perspectives or approaches to the question
        
        SELECT UP TO {max_papers_to_download} PAPERS that would most benefit from full text analysis.
        
        RESPOND WITH:
        A JSON object with this format: {{"selected_docs": [1, 5, 10, ...]}} containing the document numbers (starting from 1) that should have full text retrieved.
        """
        
        print(f"[FULL TEXT SELECTION] Running LLM selection...")
        start_time = time.time()
        
        response = self.llm.complete(prompt).text
        
        # Parse the response
        selected_docs = []
        try:
            # Try to parse as JSON
            import json
            response_json = json.loads(response)
            if 'selected_docs' in response_json and isinstance(response_json['selected_docs'], list):
                selected_docs = response_json['selected_docs']
        except:
            # Fallback: try to extract numbers using regex
            import re
            numbers = re.findall(r'\d+', response)
            selected_docs = [int(num) for num in numbers if 1 <= int(num) <= len(documents)]
        
        # Ensure we have valid document indices (adjusting to 0-based)
        valid_indices = [i-1 for i in selected_docs if 1 <= i <= len(documents)]
        
        # If we couldn't extract valid indices, take the first max_papers_to_download
        if not valid_indices:
            print(f"[FULL TEXT SELECTION] Could not extract valid document indices from LLM response, using first {max_papers_to_download} documents")
            valid_indices = list(range(min(max_papers_to_download, len(documents))))
        
        # Limit to max_papers_to_download
        valid_indices = valid_indices[:max_papers_to_download]
        
        selection_time = time.time() - start_time
        print(f"[FULL TEXT SELECTION] Selected {len(valid_indices)} documents for full text retrieval in {selection_time:.2f} seconds")
        
        return valid_indices
    
    def get_document_full_text(self, doc: Document, paper: arxiv.Result) -> Document:
        """
        Get full text for a document, using cache if available.
        
        Args:
            doc: Document object
            paper: Corresponding arXiv.Result object
            
        Returns:
            Updated Document object with full text (if available)
        """
        paper_id = doc.metadata.get('id', '')
        
        if not paper_id:
            print(f"[FULL TEXT] Document has no arXiv ID, skipping full text processing")
            return doc
        
        # Check if already processed and cached
        if self._is_paper_cached(paper_id):
            print(f"[FULL TEXT] Using cached parsed paper for {paper_id}")
            full_text = self._load_parsed_paper(paper_id)
            
            if full_text:
                # Create updated document with full text
                updated_text = doc.text + f"\nFULL TEXT:\n{full_text}\n"
                updated_doc = Document(text=updated_text)
                
                # Copy metadata and update has_full_text flag
                updated_metadata = doc.metadata.copy()
                updated_metadata['has_full_text'] = True
                updated_doc.metadata = updated_metadata
                
                return updated_doc
        
        print(f"[FULL TEXT] Processing document: {doc.metadata.get('title', 'Untitled')}")
        
        # Download PDF
        pdf_path = self.download_pdf(paper)
        
        if pdf_path and os.path.exists(pdf_path):
            # Parse PDF with LlamaParse
            full_text = self.process_pdf_with_llamaparse(pdf_path)
            
            # Cache the parsed result
            self._save_parsed_paper(paper_id, full_text)
            
            # Create updated document with full text
            updated_text = doc.text + f"\nFULL TEXT:\n{full_text}\n"
            updated_doc = Document(text=updated_text)
            
            # Copy metadata and update has_full_text flag
            updated_metadata = doc.metadata.copy()
            updated_metadata['has_full_text'] = True
            updated_doc.metadata = updated_metadata
            
            return updated_doc
        else:
            print(f"[FULL TEXT] No PDF could be downloaded, using abstract only")
            return doc
                     
    def process_documents_with_full_text(self, research_question: str, documents: List[Document], papers: List[arxiv.Result], max_papers_to_download: int = 3) -> List[Document]:
        """
        Process documents including downloading and parsing PDFs for full text analysis in parallel.
        
        Args:
            research_question: The research question
            documents: List of Document objects
            papers: List of corresponding arxiv.Result objects
            max_papers_to_download: Maximum number of papers to download
            
        Returns:
            List of Document objects with full text when available
        """
        print(f"\n[FULL TEXT PROCESSING] Processing documents with full text extraction")
        
        # Use LLM to select which documents should have full text retrieved
        selected_indices = self.select_documents_for_full_text(research_question, documents, max_papers_to_download)
        
        updated_documents = documents.copy()  # Create a copy to avoid modifying the original list
        
        if not selected_indices:
            print(f"[FULL TEXT PROCESSING] No documents selected for full text processing")
            return updated_documents
        
        
        print(f"[FULL TEXT PROCESSING] Processing {len(selected_indices)} documents in parallel")
        
        # Create a list of (document, paper) tuples for selected indices
        docs_to_process = [(documents[i], papers[i]) for i in selected_indices if i < len(papers)]
        
        # Define a worker function for parallel processing
        def process_doc_worker(doc_paper_tuple):
            doc, paper = doc_paper_tuple
            return self.get_document_full_text(doc, paper)
        
        # Use ThreadPoolExecutor for parallel processing
        with concurrent.futures.ThreadPoolExecutor(max_workers=min(len(docs_to_process), 5)) as executor:
            # Submit all tasks and collect futures
            future_to_index = {
                executor.submit(process_doc_worker, (documents[i], papers[i])): i 
                for i in selected_indices if i < len(papers)
            }
            
            # Process results as they complete
            for future in concurrent.futures.as_completed(future_to_index):
                i = future_to_index[future]
                try:
                    # Get the result and update the document
                    updated_documents[i] = future.result()
                    print(f"[FULL TEXT PROCESSING] Completed document {i+1}: {documents[i].metadata.get('title', 'Untitled')}")
                except Exception as e:
                    print(f"[FULL TEXT PROCESSING] Error processing document {i+1}: {str(e)}")
        
        print(f"[FULL TEXT PROCESSING] Processed {len(updated_documents)} documents ({sum(1 for doc in updated_documents if doc.metadata.get('has_full_text', False))} with full text)")
        return updated_documents
    
    def prepare_prompt(self, research_question: str, documents: List[Document]) -> str:
        """
        Prepare a prompt for answering the research question using the retrieved papers.
        
        Args:
            research_question: The original research question
            documents: List of Document objects containing paper information
            
        Returns:
            A prompt that can be used with an LLM to generate a response
        """
        print(f"\n[PROMPT PREPARATION] Preparing prompt for: '{research_question}'")
        
        # Limit the number of papers to include in the context
        used_documents = documents[:self.max_papers_used]
        print(f"[PROMPT PREPARATION] Using {len(used_documents)} documents in prompt")
        
        # Format papers as context
        papers_context = ""
        for i, doc in enumerate(used_documents):
            has_full_text = doc.metadata.get('has_full_text', False)
            papers_context += f"\nPaper {i+1} "
            papers_context += f"(with full text)" if has_full_text else "(abstract only)"
            papers_context += f":\n{doc.text}\n"
        
        # Create prompt
        prompt = f"""
        You are a research assistant that answers questions based on scientific papers from arXiv.
        Use the following papers to answer the research question. If the papers don't contain relevant information, say so.
        Always cite papers using their arXiv ID (e.g., [2107.05580]) when you use information from them.
        Some papers include full text while others only have abstracts - prioritize information from full-text papers.
        
        PAPERS:
        {papers_context}
        
        RESEARCH QUESTION: {research_question}
        
        Provide a comprehensive answer with proper citations to arXiv papers.
        Include a "REFERENCES" section at the end that lists all papers you cited, including titles and authors.
        Format the references as:
        [ID] Author1, Author2, et al. "Title"
        """
        
        print(f"[PROMPT PREPARATION] Prompt prepared successfully")
        return prompt
    
    def get_papers_for_research(
        self,
        research_question: str,
        use_llm_query: bool = True,
        use_full_text: bool = False,
        max_papers_to_download: int = 3,
        sort_by: arxiv.SortCriterion = arxiv.SortCriterion.Relevance,
        papers: List[arxiv.Result] = None
    ) -> Dict[str, Any]:
        """
        Get papers from ArXiv to answer a research question and prepare a prompt.
        
        Args:
            research_question: The research question to answer
            use_llm_query: Whether to use LLM to generate the query or use the question directly
            use_full_text: Whether to download and parse PDFs or just use abstracts
            max_papers_to_download: Maximum number of papers to download (for full text analysis)
            sort_by: How to sort ArXiv results
            papers: Optional pre-retrieved papers to use (to avoid duplicate searches)
            
        Returns:
            A dictionary containing the prompt and process details
        """
        print(f"\n[ARXIV RAG] Processing research question: '{research_question}'")
        start_time = time.time()
        
        # Store the research question in search history for potential query refinement
        self.search_history.append({
            "research_question": research_question,
            "timestamp": time.ctime()
        })
        
        # Step 1 & 2: Generate query and retrieve papers (if not already provided)
        if papers is None:
            # Step 1: Generate optimized search query if requested
            if use_llm_query:
                arxiv_query = self.generate_search_query(research_question)
            else:
                arxiv_query = research_question
                print(f"[ARXIV RAG] Using research question directly as query: '{arxiv_query}'")
            
            # Step 2: Retrieve papers from ArXiv
            papers = self.retrieve_papers(arxiv_query, research_question, sort_by=sort_by)
        else:
            # Use the provided papers and get the query from the latest search history
            arxiv_query = self.search_history[-1]['query'] if len(self.search_history) > 1 else research_question
            print(f"[ARXIV RAG] Using {len(papers)} previously retrieved papers")
        
        # If still no papers found after all retries, provide a useful error message
        if not papers:
            print(f"[ARXIV RAG] WARNING: No papers found after multiple query attempts")
            result = {
                "research_question": research_question,
                "arxiv_query": arxiv_query,
                "papers_retrieved": 0,
                "documents_processed": 0,
                "full_text_used": use_full_text,
                "total_time": time.time() - start_time,
                "error": "No papers found after multiple query attempts",
                "prompt": f"No papers were found on ArXiv for the research question: '{research_question}'. Please try a different query or research question."
            }
            return result
            
        # Step 3: Process papers into documents
        documents = self.process_papers(papers)
        
        # Step 4: Screen documents for relevance
        documents, slected_indices = self.screen_items_with_llm(research_question, documents)

        # update papers list to maintaining the same order as documents
        papers = [papers[i] for i in slected_indices]
        
        # Step 5: Process documents with full text if needed
        if use_full_text:
            print(f"[ARXIV RAG] Using full text analysis for up to {max_papers_to_download} papers")
            documents = self.process_documents_with_full_text(research_question, documents, papers, max_papers_to_download)
        else:
            print(f"[ARXIV RAG] Using abstracts only")
        
        # Step 6: Prepare prompt
        prompt = self.prepare_prompt(research_question, documents)
        
        # Calculate total time
        total_time = time.time() - start_time
        print(f"[ARXIV RAG] Completed in {total_time:.2f} seconds")
        
        # Return all the details for transparency
        result = {
            "research_question": research_question,
            "arxiv_query": arxiv_query,
            "papers_retrieved": len(papers),
            "documents_processed": len(documents),
            "full_text_used": use_full_text,
            "documents": documents,  # Include the documents for reference
            "total_time": total_time,
            "prompt": prompt,
        }
        
        return result

def save_papers_to_json(retrieved_papers, output_dir="./json_output"):
    """
    Save retrieved papers to a JSON file
    
    Args:
        retrieved_papers: Dictionary containing the research results
        output_dir: Directory to save the JSON file
    
    Returns:
        Path to the saved JSON file
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    question_snippet = retrieved_papers["research_question"][:30].replace(" ", "_")
    filename = f"{timestamp}_{question_snippet}.json"
    filepath = os.path.join(output_dir, filename)
    
    # Save data to JSON file - note we're just saving the prompt
    # You could modify this to save more details if needed
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(retrieved_papers['prompt'], f, indent=2, ensure_ascii=False)
    
    print(f"[ARXIV RAG] Saved retrieved papers to {filepath}")
    return filepath

