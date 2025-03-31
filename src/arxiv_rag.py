import time
import arxiv
from typing import List, Dict, Any, Optional
from llama_index.llms.openai import OpenAI
from llama_index.core.schema import Document

class TransparentArxivRAG:
    """
    A transparent RAG system for ArXiv papers.
    
    This system:
    1. Uses an LLM to generate optimized ArXiv search queries
    2. Retrieves relevant papers from ArXiv
    3. Processes the papers for RAG
    4. Generates responses with proper citations
    """
    
    def __init__(
        self, 
        llm_model: str = "o3-mini", 
        max_results: int = 100,
        max_papers_used: int = 30
    ):
        """
        Initialize the transparent ArXiv RAG system.
        
        Args:
            llm_model: The OpenAI model to use
            max_results: Maximum number of ArXiv results to retrieve
            max_papers_used: Maximum number of papers to include in the context
        """
        self.llm = OpenAI(model=llm_model)
        self.max_results = max_results
        self.max_papers_used = max_papers_used
        self.search_history = []
        
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
        sort_by: arxiv.SortCriterion = arxiv.SortCriterion.Relevance
    ) -> List[arxiv.Result]:
        """
        Retrieve papers from ArXiv using the provided query.
        
        Args:
            query: The ArXiv search query
            sort_by: How to sort the results
            
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
                "summary_snippet": paper.summary[:200] + "..." if len(paper.summary) > 200 else paper.summary
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
                "categories": categories
            }
            
            documents.append(doc)
        
        print(f"[PAPER PROCESSING] Processed {len(documents)} documents")
        return documents
    
    def generate_response(self, research_question: str, documents: List[Document]) -> str:
        """
        Generate a response to the research question using the retrieved papers.
        
        Args:
            research_question: The original research question
            documents: List of Document objects containing paper information
            
        Returns:
            A comprehensive response with citations
        """
        print(f"\n[RESPONSE GENERATION] Generating response to: '{research_question}'")
        
        # Limit the number of papers to include in the context
        used_documents = documents[:self.max_papers_used]
        print(f"[RESPONSE GENERATION] Using {len(used_documents)} papers in context")
        
        # Format papers as context
        papers_context = ""
        for i, doc in enumerate(used_documents):
            papers_context += f"\nPaper {i+1}:\n{doc.text}\n"
        
        # Create prompt
        prompt = f"""
        You are a research assistant that answers questions based on ArXiv scientific papers.
        Use the following papers to answer the research question. If the papers don't contain relevant information, say so.
        Always cite papers using their arXiv ID (e.g., [2107.05580]) when you use information from them.
        
        PAPERS:
        {papers_context}
        
        RESEARCH QUESTION: {research_question}
        
        Provide a comprehensive answer with proper citations to arXiv papers.
        Include a "REFERENCES" section at the end that lists all papers you cited, including titles and authors.
        Format the references as:
        [ID] Author1, Author2, et al. "Title"
        """
        
        start_time = time.time()
        response = self.llm.complete(prompt).text
        generation_time = time.time() - start_time
        
        print(f"[RESPONSE GENERATION] Response generated in {generation_time:.2f} seconds")
        return response
    
    def answer_question(
        self, 
        research_question: str, 
        use_llm_query: bool = True,
        sort_by: arxiv.SortCriterion = arxiv.SortCriterion.Relevance
    ) -> Dict[str, Any]:
        """
        End-to-end process to answer a research question using ArXiv papers.
        
        Args:
            research_question: The research question to answer
            use_llm_query: Whether to use LLM to generate the query or use the question directly
            sort_by: How to sort ArXiv results
            
        Returns:
            A dictionary containing the response and process details
        """
        print(f"\n[RAG SYSTEM] Processing research question: '{research_question}'")
        start_time = time.time()
        
        # Step 1: Generate optimized search query if requested
        if use_llm_query:
            arxiv_query = self.generate_search_query(research_question)
        else:
            arxiv_query = research_question
            print(f"[RAG SYSTEM] Using research question directly as query: '{arxiv_query}'")
        
        # Step 2: Retrieve papers from ArXiv
        papers = self.retrieve_papers(arxiv_query, sort_by=sort_by)
        
        # Step 3: Process papers into documents
        documents = self.process_papers(papers)
        
        # Step 4: Generate response
        response = self.generate_response(research_question, documents)
        
        # Calculate total time
        total_time = time.time() - start_time
        print(f"[RAG SYSTEM] Completed in {total_time:.2f} seconds")
        
        # Return all the details for transparency
        result = {
            "research_question": research_question,
            "arxiv_query": arxiv_query,
            "papers_retrieved": len(papers),
            "papers_used": min(len(papers), self.max_papers_used),
            "total_time": total_time,
            "response": response,
        }
        
        return result
    
    def get_search_history(self) -> List[Dict[str, Any]]:
        """Get the search history."""
        return self.search_history

def run_transparent_arxiv_rag():
    """Run the transparent ArXiv RAG system interactively."""
    # ASCII art for EQUAL LAB
    def print_equal_lab():
        letters = {
            'E': [
                "#######",
                "#      ",
                "#      ",
                "#####  ",
                "#      ",
                "#######"
            ],
            'Q': [
                " ##### ",
                "#     #",
                "#     #",
                "#     #",
                "#  #  #",
                " ######"
            ],
            'U': [
                "#     #",
                "#     #",
                "#     #",
                "#     #",
                "#     #",
                " ##### "
            ],
            'A': [
                "   #   ",
                "  # #  ",
                " #   # ",
                "#######",
                "#     #",
                "#     #"
            ],
            'L': [
                "#      ",
                "#      ",
                "#      ",
                "#      ",
                "#      ",
                "#######"
            ],
            'B': [
                "###### ",
                "#     #",
                "#     #",
                "###### ",
                "#     #",
                "###### "
            ],
            ' ': [
                "       ",
                "       ",
                "       ",
                "       ",
                "       ",
                "       "
            ]
        }

        text = "EQUAL LAB"
        # Each letter is 6 rows high.
        for row in range(6):
            # Join each letter's row with 4 spaces in between.
            line = "    ".join(letters[ch][row] for ch in text)
            print(line)
    
    print_equal_lab()
    print("\nTransparent ArXiv RAG System")
    print("============================")
    
    # Initialize the RAG system
    rag_system = TransparentArxivRAG(max_results=100, max_papers_used=30)
    
    print("This system uses LLM to optimize your research questions into effective ArXiv queries.")
    print("It transparently shows every step of the process from query generation to paper retrieval.")
    print("Please ask your research question!\n")
    
    # Get the research question
    question = input("Enter your research question: ")
    
    # Ask if they want to use LLM for query generation
    use_llm = input("\nUse LLM to optimize the ArXiv query? (yes/no, default: yes): ")
    use_llm_query = True if use_llm.lower() not in ["no", "n"] else False
    
    # Ask for sort criterion
    sort_options = {
        "1": ("Relevance", arxiv.SortCriterion.Relevance),
        "2": ("Last Updated Date", arxiv.SortCriterion.LastUpdatedDate),
        "3": ("Submission Date", arxiv.SortCriterion.SubmittedDate)
    }
    
    print("\nSort results by:")
    for key, (name, _) in sort_options.items():
        print(f"{key}. {name}")
    
    sort_choice = input("Choose sorting method (default: 1): ")
    sort_by = sort_options.get(sort_choice, sort_options["1"])[1]
    
    # Process the question
    print("\nProcessing your research question...")
    result = rag_system.answer_question(question, use_llm_query=use_llm_query, sort_by=sort_by)
    
    # Display the answer
    print("\n\033[1;31m\nArXiv RAG Answer:\n\033[0m")
    print(f"\033[31m{result['response']}\033[0m")
    
    # Offer to show search details
    show_details = input("\nWould you like to see detailed information about the papers? (yes/no): ")
    if show_details.lower() in ["yes", "y"]:
        history = rag_system.get_search_history()
        if history:
            print("\n=== SEARCH PROCESS DETAILS ===")
            latest_search = history[-1]
            print(f"\nResearch Question: '{question}'")
            print(f"ArXiv Query: '{latest_search['query']}'")
            print(f"Papers Retrieved: {latest_search['num_results']}")
            print(f"Papers Used in Response: {result['papers_used']}")
            print(f"Total Time: {result['total_time']:.2f} seconds")
            
            print("\nPapers Details:")
            for i, paper in enumerate(latest_search['papers'][:result['papers_used']]):
                print(f"\nPaper {i+1}:")
                print(f"Title: {paper['title']}")
                print(f"Authors: {paper['authors']}")
                print(f"ID: {paper['id']}")
                print(f"Published: {paper['published']}")
                print(f"Categories: {paper['categories']}")
                print(f"Summary: {paper['summary_snippet']}")
    
    return rag_system

if __name__ == "__main__":
    run_transparent_arxiv_rag()