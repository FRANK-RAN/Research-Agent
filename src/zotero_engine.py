import time
import os
import json
import pickle
import hashlib
from typing import List, Dict, Any, Optional
from pyzotero import zotero
# from llama_index.llms.openai import OpenAI
from llama_index.core.schema import Document
from llama_parse import LlamaParse
from llama_index.core import SimpleDirectoryReader
from datetime import datetime

from src.models import CustomOpenAI as OpenAI



class ZoteroRAG:
    """
    A RAG system for Zotero libraries.
    
    This system:
    1. Connects to a user's Zotero library
    2. Retrieves items from specified collections
    3. Uses an LLM to screen papers for relevance
    4. Accesses attachments (local or downloads them if needed)
    5. Prepares prompts with the papers for research question answering
    """
    
    def __init__(
        self, 
        library_id: str,
        library_type: str = 'user',  # 'user' or 'group'
        api_key: str = None,
        llm_model: str = "o3-mini",
        max_papers_used: int = 30,
        download_dir: str = "./zotero_downloads",
        cache_dir: str = "./paper_cache",
        local_storage_path: str = None
    ):
        """
        Initialize the Zotero RAG system.
        
        Args:
            library_id: Zotero library ID (user or group)
            library_type: Type of library ('user' or 'group')
            api_key: Zotero API key
            llm_model: The OpenAI model to use
            max_papers_used: Maximum number of papers to include in the context
            download_dir: Directory to save downloaded attachments
            cache_dir: Directory to cache parsed papers
            local_storage_path: Path to local Zotero storage (if available)
        """
        self.library_id = library_id
        self.library_type = library_type
        self.api_key = api_key
        self.llm = OpenAI(model=llm_model)
        self.max_papers_used = max_papers_used
        self.download_dir = download_dir
        self.cache_dir = cache_dir
        self.local_storage_path = local_storage_path
        
        # Create required directories if they don't exist
        for directory in [download_dir, cache_dir]:
            if not os.path.exists(directory):
                os.makedirs(directory, exist_ok=True)
        
        # Initialize Zotero client
        self.zot = zotero.Zotero(library_id, library_type, api_key)
        
        # Keep track of search history
        self.search_history = []
    
    def get_collections(self) -> List[Dict[str, Any]]:
        """
        Get all collections in the Zotero library.
        
        Returns:
            List of collection objects
        """
        print(f"\n[ZOTERO] Retrieving collections from library {self.library_id}")
        
        try:
            collections = self.zot.collections()
            print(f"[ZOTERO] Found {len(collections)} collections")
            
            # Display collections
            for i, collection in enumerate(collections):
                name = collection['data']['name']
                key = collection['key']
                print(f"[ZOTERO COLLECTION {i+1}] {name} (Key: {key})")
            
            return collections
        except Exception as e:
            print(f"[ZOTERO] Error retrieving collections: {str(e)}")
            return []
    
    def get_collection_items(self, collection_keys: List[str]) -> List[Dict[str, Any]]:
        """
        Get all items from specified collections.
        
        Args:
            collection_keys: List of collection keys to retrieve items from
            
        Returns:
            List of Zotero item objects
        """
        print(f"\n[ZOTERO] Retrieving items from {len(collection_keys)} collections")
        
        all_items = []
        
        for collection_key in collection_keys:
            start = 0
            limit = 100  # Maximum items per request
            collection_items = []
            
            while True:
                try:
                    items = self.zot.collection_items(collection_key, start=start, limit=limit)
                    if not items:
                        break
                    
                    collection_items.extend(items)
                    start += limit
                    
                    # Check if we've retrieved all items
                    if len(items) < limit:
                        break
                except Exception as e:
                    print(f"[ZOTERO] Error retrieving items from collection {collection_key}: {str(e)}")
                    break
            
            # Get collection name for logging
            try:
                collection = self.zot.collection(collection_key)
                collection_name = collection['data']['name']
            except:
                collection_name = collection_key
            
            print(f"[ZOTERO] Retrieved {len(collection_items)} items from collection: {collection_name}")
            all_items.extend(collection_items)
        
        # Remove duplicates (items might exist in multiple collections)
        unique_items = {}
        for item in all_items:
            if 'key' in item:
                unique_items[item['key']] = item
        
        deduplicated_items = list(unique_items.values())
        print(f"[ZOTERO] Retrieved {len(deduplicated_items)} unique items from all collections")
        
        return deduplicated_items
    
    def get_all_library_items(self) -> List[Dict[str, Any]]:
        """
        Get all items from the Zotero library.
        
        Returns:
            List of Zotero item objects
        """
        print(f"\n[ZOTERO] Retrieving all items from library")
        
        start = 0
        limit = 100  # Maximum items per request
        all_items = []
        
        while True:
            try:
                items = self.zot.items(start=start, limit=limit)
                if not items:
                    break
                
                all_items.extend(items)
                start += limit
                print(f"[ZOTERO] Retrieved {len(all_items)} items so far...")
                
                # Check if we've retrieved all items
                if len(items) < limit:
                    break
            except Exception as e:
                print(f"[ZOTERO] Error retrieving items: {str(e)}")
                break
        
        print(f"[ZOTERO] Retrieved {len(all_items)} total items from library")
        return all_items
    
    def filter_items_by_type(self, items: List[Dict[str, Any]], included_types: List[str] = None) -> List[Dict[str, Any]]:
        """
        Filter items by their type.
        
        Args:
            items: List of Zotero items
            included_types: List of item types to include (e.g., ['journalArticle', 'book'])
                            If None, keep all types except attachments and notes
        
        Returns:
            Filtered list of items
        """
        if included_types is None:
            # Default: exclude attachments and notes, keep everything else
            excluded_types = ['attachment', 'note']
            filtered_items = [item for item in items if item.get('data', {}).get('itemType') not in excluded_types]
        else:
            # Include only specified types
            filtered_items = [item for item in items if item.get('data', {}).get('itemType') in included_types]
        
        print(f"[ZOTERO] Filtered from {len(items)} to {len(filtered_items)} items based on type")
        return filtered_items



    def process_items(self, items: List[Dict[str, Any]]) -> List[Document]:
        """
        Process Zotero items into Documents for the RAG system.
        
        Args:
            items: List of Zotero item objects
            
        Returns:
            List of Document objects
        """
        print(f"\n[ITEM PROCESSING] Processing {len(items)} items for RAG")
        
        documents = []
        for item in items:
            item_data = item.get('data', {})
            
            # Extract metadata
            title = item_data.get('title', 'Untitled')
            
            # Process creators
            creators = item_data.get('creators', [])
            authors = []
            for creator in creators:
                if 'lastName' in creator and 'firstName' in creator:
                    authors.append(f"{creator['lastName']}, {creator['firstName']}")
                elif 'name' in creator:
                    authors.append(creator['name'])
            
            authors_str = ", ".join(authors) if authors else "Unknown"
            
            # Other metadata
            date = item_data.get('date', 'Unknown date')
            item_type = item_data.get('itemType', 'Unknown type')
            abstract = item_data.get('abstractNote', '')
            tags = [tag.get('tag', '') for tag in item_data.get('tags', [])]
            tags_str = ", ".join(tags) if tags else ""
            
            # Create document text with structured information
            doc_text = f"Title: {title}\n"
            doc_text += f"Authors: {authors_str}\n"
            doc_text += f"Date: {date}\n"
            doc_text += f"Type: {item_type}\n"
            
            if tags_str:
                doc_text += f"Tags: {tags_str}\n"
            
            if abstract:
                doc_text += f"Abstract: {abstract}\n"
            else:
                doc_text += "Abstract: Not available\n"
            
            # Create document
            doc = Document(text=doc_text)
            
            # Add metadata
            doc.metadata = {
                "title": title,
                "authors": authors_str,
                "date": date,
                "type": item_type,
                "tags": tags_str,
                "has_abstract": bool(abstract),
                "zotero_key": item.get('key', ''),
                "has_full_text": False  # Will be updated if full text is added
            }
            
            documents.append(doc)
        
        print(f"[ITEM PROCESSING] Processed {len(documents)} documents")
        return documents
    
    def screen_items_with_llm(self, research_question: str, documents: List[Document], max_items: int = None) -> List[Document]:
        """
        Use LLM to screen items for relevance to the research question.
        
        Args:
            research_question: The research question
            documents: List of Document objects
            max_items: Maximum number of items after screening to return (default: self.max_papers_used)
            
        Returns:
            List of relevant Document objects
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
            
        return screened_documents
    
    def analyze_full_text_need(self, research_question: str, documents: List[Document] = None) -> Dict[str, Any]:
        """
        Analyze if the research question requires full text analysis.
        
        Args:
            research_question: The research question
            documents: Optional list of Document objects to consider
            
        Returns:
            Dictionary with analysis results and recommendation
        """
        print(f"\n[ANALYSIS] Determining if full text analysis is needed for: '{research_question}'")
        
        # Base prompt
        prompt = f"""
        Analyze the following research question and determine if full text analysis of scientific papers 
        would be beneficial or if paper abstracts would likely contain sufficient information.
        
        Research Question: "{research_question}"
        
        When making your determination, consider these factors:
        1. If the question asks for detailed methodologies or specific experimental results, full text is likely needed
        2. If the question is about comparing/contrasting multiple approaches in detail, full text is likely needed
        3. If the question only seeks high-level concepts, trends, or basic facts, abstracts may be sufficient
        4. If the question involves specific measurements, datasets, or implementation details, full text is likely needed
        """
        
        # Add document context if provided
        if documents:
            docs_context = ""
            for i, doc in enumerate(documents[:5]):  # Only use first 5 documents to avoid overloading context
                docs_context += f"Document {i+1}:\n{doc.text}\n\n"
            
            prompt += f"""
            Here are some of the paper abstracts that might be used to answer this question:
            
            {docs_context}
            
            Based on both the research question and these available abstracts, determine if full text analysis is needed.
            """
        
        prompt += """
        Please analyze and provide:
        1. If full paper text analysis is likely needed (yes/no)
        2. Confidence level (low/medium/high)
        3. A brief explanation why
        
        Format your response as a JSON object with keys: "full_text_needed" (boolean), "confidence" (string), "explanation" (string)
        """
        
        response = self.llm.complete(prompt).text.strip()
        
        # Default values in case parsing fails
        default_analysis = {
            "full_text_needed": True,  # Default to True for safety
            "confidence": "medium",
            "explanation": "Default explanation: Parser couldn't extract structured data from LLM response."
        }
        
        # Try to parse as JSON first
        try:
            import json
            parsed_json = json.loads(response)
            
            # Ensure the required keys exist and are of the right type
            if "full_text_needed" in parsed_json:
                # Convert string "yes"/"no" to boolean if needed
                if isinstance(parsed_json["full_text_needed"], str):
                    parsed_json["full_text_needed"] = parsed_json["full_text_needed"].lower() in ["yes", "true", "1"]
            else:
                parsed_json["full_text_needed"] = default_analysis["full_text_needed"]
                
            if "confidence" not in parsed_json:
                parsed_json["confidence"] = default_analysis["confidence"]
                
            if "explanation" not in parsed_json:
                parsed_json["explanation"] = default_analysis["explanation"]
                
            analysis = parsed_json
            
        except json.JSONDecodeError:
            # If not valid JSON, extract information with simpler parsing
            analysis = default_analysis.copy()
            
            # Try to extract full_text_needed
            if "yes" in response.lower() or "full text is needed" in response.lower():
                analysis["full_text_needed"] = True
            elif "no" in response.lower() or "abstracts are sufficient" in response.lower():
                analysis["full_text_needed"] = False
            
            # Try to extract confidence
            if "high confidence" in response.lower():
                analysis["confidence"] = "high"
            elif "medium confidence" in response.lower() or "moderate confidence" in response.lower():
                analysis["confidence"] = "medium"
            elif "low confidence" in response.lower():
                analysis["confidence"] = "low"
            
            # Try to extract explanation
            explanation_markers = ["explanation:", "because", "reason:"]
            for marker in explanation_markers:
                if marker in response.lower():
                    parts = response.lower().split(marker)
                    if len(parts) > 1:
                        # Take the part after the marker and clean it up
                        explanation = parts[1].strip()
                        # Take just the first sentence or up to 200 chars
                        explanation = explanation.split('.')[0].strip() + '.'
                        explanation = explanation[:200]
                        analysis["explanation"] = explanation
                        break
        
        print(f"[ANALYSIS] Full text needed: {analysis['full_text_needed']}")
        print(f"[ANALYSIS] Confidence: {analysis['confidence']}")
        print(f"[ANALYSIS] Explanation: {analysis['explanation']}")
        
        return analysis
    
    def _get_cache_path(self, item_key: str) -> str:
        """Get the path for cached parsed paper"""
        return os.path.join(self.cache_dir, f"{item_key}.pkl")
    
    def _is_paper_cached(self, item_key: str) -> bool:
        """Check if paper is already parsed and cached"""
        cache_path = self._get_cache_path(item_key)
        return os.path.exists(cache_path)
    
    def _save_parsed_paper(self, item_key: str, full_text: str) -> None:
        """Save parsed paper to cache"""
        cache_path = self._get_cache_path(item_key)
        try:
            with open(cache_path, 'wb') as f:
                pickle.dump(full_text, f)
            print(f"[CACHE] Saved parsed paper to cache: {cache_path}")
        except Exception as e:
            print(f"[CACHE] Error saving to cache: {str(e)}")
    
    def _load_parsed_paper(self, item_key: str) -> Optional[str]:
        """Load parsed paper from cache"""
        cache_path = self._get_cache_path(item_key)
        try:
            with open(cache_path, 'rb') as f:
                full_text = pickle.load(f)
            print(f"[CACHE] Loaded parsed paper from cache: {cache_path}")
            return full_text
        except Exception as e:
            print(f"[CACHE] Error loading from cache: {str(e)}")
            return None
    
    def find_local_attachment_path(self, item_key: str) -> List[str]:
        """
        Find local attachment paths for a Zotero item (if local storage is configured).
        
        Args:
            item_key: Zotero item key
            
        Returns:
            List of paths to local attachment files
        """
        if not self.local_storage_path:
            return []
        
        # Get attachments from Zotero API
        try:
            children = self.zot.children(item_key)
            attachments = [child for child in children if child['data']['itemType'] == 'attachment']
        except Exception as e:
            print(f"[LOCAL STORAGE] Error getting children for item {item_key}: {str(e)}")
            return []
        
        if not attachments:
            return []
        
        local_paths = []
        
        for attachment in attachments:
            attachment_data = attachment['data']
            
            # Only consider file attachments
            if attachment_data.get('linkMode') == 'imported_file':
                attachment_key = attachment['key']
                
                # Try to construct local path from storage key
                if 'key' in attachment and self.local_storage_path:
                    # Zotero storage pattern: storage/ABCD1234.pdf
                    # where ABCD1234 is the attachment key
                    storage_path = os.path.join(
                        self.local_storage_path, 
                        attachment_key
                    )
                    
                    # Handle different Zotero storage patterns
                    possible_paths = [
                        storage_path,  # Direct key
                        os.path.join(storage_path, attachment_data.get('filename', '')),  # Key as directory
                        os.path.join(self.local_storage_path, attachment_key[:2], attachment_key[2:]),  # Key split (newer Zotero)
                    ]
                    
                    for path in possible_paths:
                        if os.path.exists(path):
                            print(f"[LOCAL STORAGE] Found local attachment: {path}")
                            local_paths.append(path)
                            break
        
        return local_paths
    
    def download_attachment(self, item_key: str, attachment_key: str, filename: str) -> Optional[str]:
        """
        Download a specific attachment from Zotero.
        
        Args:
            item_key: Zotero item key
            attachment_key: Zotero attachment key
            filename: Filename to save as
            
        Returns:
            Path to the downloaded file or None if download failed
        """
        print(f"[ATTACHMENT DOWNLOAD] Downloading attachment {attachment_key} for item {item_key}")
        
        try:
            # Get the attachment content
            file_content = self.zot.file(attachment_key)
            
            # Save the file
            file_path = os.path.join(self.download_dir, filename)
            with open(file_path, 'wb') as f:
                f.write(file_content)
            
            print(f"[ATTACHMENT DOWNLOAD] Successfully downloaded to {file_path}")
            return file_path
        except Exception as e:
            print(f"[ATTACHMENT DOWNLOAD] Error downloading attachment: {e}")
            return None
    
    def get_attachments(self, item_key: str) -> List[str]:
        """
        Get attachments for a Zotero item - first check local, then download if needed.
        
        Args:
            item_key: Zotero item key
            
        Returns:
            List of paths to attachment files
        """
        print(f"[ATTACHMENTS] Getting attachments for item {item_key}")
        
        # First check local storage
        local_paths = self.find_local_attachment_path(item_key)
        if local_paths:
            print(f"[ATTACHMENTS] Found {len(local_paths)} local attachments")
            return local_paths
        
        # If no local attachments, download them
        print(f"[ATTACHMENTS] No local attachments found, downloading...")
        
        # Get attachments from Zotero API
        try:
            children = self.zot.children(item_key)
            attachments = [child for child in children if child['data']['itemType'] == 'attachment']
        except Exception as e:
            print(f"[ATTACHMENTS] Error getting children for item {item_key}: {str(e)}")
            return []
        
        if not attachments:
            print(f"[ATTACHMENTS] No attachments found for item {item_key}")
            return []
        
        downloaded_files = []
        
        for attachment in attachments:
            attachment_data = attachment['data']
            
            # Check if it's a file attachment (not just a link)
            if attachment_data.get('linkMode') == 'imported_file' or attachment_data.get('contentType') == 'application/pdf':
                attachment_key = attachment['key']
                filename = attachment_data.get('filename', f"attachment_{attachment_key}.pdf")
                
                # Download the attachment
                file_path = self.download_attachment(item_key, attachment_key, filename)
                if file_path:
                    downloaded_files.append(file_path)
        
        return downloaded_files
    
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
    
    def get_document_full_text(self, doc: Document) -> Document:
        """
        Get full text for a document, using cache if available.
        
        Args:
            doc: Document object
            
        Returns:
            Updated Document object with full text (if available)
        """
        zotero_key = doc.metadata.get('zotero_key', '')
        
        if not zotero_key:
            print(f"[FULL TEXT] Document has no Zotero key, skipping full text processing")
            return doc
        
        # Check if already processed and cached
        if self._is_paper_cached(zotero_key):
            print(f"[FULL TEXT] Using cached parsed paper for {zotero_key}")
            full_text = self._load_parsed_paper(zotero_key)
            
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
        
        # Get attachments (local or download)
        pdf_paths = self.get_attachments(zotero_key)
        
        if pdf_paths:
            # Process first PDF only (for simplicity)
            pdf_path = pdf_paths[0]
            full_text = self.process_pdf_with_llamaparse(pdf_path)
            
            # Cache the parsed result
            self._save_parsed_paper(zotero_key, full_text)
            
            # Create updated document with full text
            updated_text = doc.text + f"\nFULL TEXT:\n{full_text}\n"
            updated_doc = Document(text=updated_text)
            
            # Copy metadata and update has_full_text flag
            updated_metadata = doc.metadata.copy()
            updated_metadata['has_full_text'] = True
            updated_doc.metadata = updated_metadata
            
            return updated_doc
        else:
            print(f"[FULL TEXT] No attachments found, using abstract only")
            return doc
    
    def process_documents_with_full_text(self, research_question: str, documents: List[Document], max_papers_to_download: int = 3) -> List[Document]:
        """
        Process documents including downloading and parsing PDFs for full text analysis.
        Uses parallel processing for improved efficiency.
        
        Args:
            research_question: The research question
            documents: List of Document objects
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
        
        # Define a worker function for parallel processing
        def process_document(idx):
            doc = documents[idx]
            print(f"[FULL TEXT PROCESSING] Processing document {idx+1}/{len(selected_indices)}: {doc.metadata.get('title', 'Untitled')}")
            return idx, self.get_document_full_text(doc)
        
        # Process documents in parallel using ThreadPoolExecutor
        from concurrent.futures import ThreadPoolExecutor, as_completed
        
        with ThreadPoolExecutor(max_workers=min(len(selected_indices), 5)) as executor:
            # Submit all tasks
            future_to_index = {executor.submit(process_document, idx): idx for idx in selected_indices}
            
            # Process results as they complete
            for future in as_completed(future_to_index):
                try:
                    idx, processed_doc = future.result()
                    updated_documents[idx] = processed_doc
                except Exception as e:
                    idx = future_to_index[future]
                    print(f"[FULL TEXT PROCESSING] Error processing document {idx+1}: {str(e)}")
        
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
                papers_context += f":\n{doc.text}\n"  # Using the full text without truncation
            
            # Create prompt
            prompt = f"""            
            PAPERS:
            {papers_context}

            

            """
            
            print(f"[PROMPT PREPARATION] Prompt prepared successfully")
            return prompt
        
    def get_papers_for_research(
        self,
        research_question: str,
        collection_keys: List[str] = None,
        use_full_text: Optional[bool] = None,
        max_papers_to_download: int = 3,
        max_papers_to_consider: int = None,
        items: List[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Get papers from Zotero to answer a research question and prepare a prompt.
        
        Args:
            research_question: The research question to answer
            collection_keys: List of Zotero collection keys to search within
                            If None, will search the entire library
            use_full_text: Whether to download and parse PDFs or just use abstracts.
                        If None, LLM will analyze and give recommendation.
            max_papers_to_download: Maximum number of papers to download for full text
            max_papers_to_consider: Maximum number of papers to consider before screening
                                If None, will use all retrieved items
            items: Optional pre-retrieved items to use
            
        Returns:
            A dictionary containing the prompt and process details
        """
        print(f"\n[ZOTERO RAG] Processing research question: '{research_question}'")
        start_time = time.time()
        
        # Step 1: Retrieve items from Zotero (if not already provided)
        if items is None:
            if collection_keys:
                items = self.get_collection_items(collection_keys)
            else:
                items = self.get_all_library_items()
            
            # Filter items to include only relevant types
            relevant_types = ['journalArticle', 'book', 'bookSection', 'conferencePaper', 'report', 'thesis', 'Conference Paper']
            items = self.filter_items_by_type(items, relevant_types)
        else:
            print(f"[ZOTERO RAG] Using {len(items)} previously retrieved items")
        
        # Step 2: Process items into documents
        documents = self.process_items(items)
        
        # Step 3: Screen documents for relevance
        if max_papers_to_consider is not None and len(documents) > max_papers_to_consider:
            documents = documents[:max_papers_to_consider]
            print(f"[ZOTERO RAG] Limited to {max_papers_to_consider} documents for consideration")
        
        documents = self.screen_items_with_llm(research_question, documents)
        
        # Step 5: Process documents with full text if needed
        if use_full_text:
            print(f"[ZOTERO RAG] Using full text analysis for up to {max_papers_to_download} papers")
            documents = self.process_documents_with_full_text(research_question, documents, max_papers_to_download)
        else:
            print(f"[ZOTERO RAG] Using abstracts only")
        
        # Step 6: Prepare prompt
        prompt = self.prepare_prompt(research_question, documents)
        
        # Calculate total time
        total_time = time.time() - start_time
        print(f"[ZOTERO RAG] Completed in {total_time:.2f} seconds")
        
        # Return all the details for transparency
        result = {
            "research_question": research_question,
            "items_retrieved": len(items),
            "documents_processed": len(documents),
            "full_text_used": use_full_text,
            "documents": documents,  # Include the documents for reference
            "total_time": total_time,
            "prompt": prompt,
        }
        
        # Add to search history
        self.search_history.append({
            "timestamp": time.ctime(),
            "research_question": research_question,
            "collection_keys": collection_keys,
            "full_text_used": use_full_text,
            "documents_count": len(documents)
        })
        
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
    
    # Save the data to JSON file
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(retrieved_papers['prompt'], f, indent=2, ensure_ascii=False)
    
    print(f"[ZOTERO RAG] Saved retrieved papers to {filepath}")
    return filepath





def main():
    zotero_engine = ZoteroRAG(
        library_id='5310176',
        library_type='group',
        api_key='91Z1BHYMHJonusqkbBP6hE60',
        llm_model='o4-mini',
        max_papers_used=50,
        download_dir='./zotero_downloads',
        cache_dir='./paper_cache',
        local_storage_path='/Users/frankran/Zotero/storage'
    )

    # Example usage
    research_question = "What are the sensitivity of these measurements: Spin Noise Spectroscopy (SNS), Optically Detected Magnetic Resonance (ODMR), Nuclear Magnetic Resonance (NMR) setups?"

    research_question = "what is the relationship between chirality and magnetism?"   

    retrived_papers = zotero_engine.get_papers_for_research(research_question=research_question, use_full_text=True, max_papers_to_download=3, max_papers_to_consider=None)
    
    # download the retrieved papers into json for checking
    json_path = save_papers_to_json(retrived_papers)


if __name__ == "__main__":
    main()