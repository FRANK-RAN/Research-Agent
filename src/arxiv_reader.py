from llama_index.readers.papers.arxiv import ArxivReader

# Initialize the reader
reader = ArxivReader()

# Try with explicit parameters and error handling

search_docs, abtracts = reader.load_papers_and_abstracts(
    search_query="ErFeO3",
    papers_dir="/Users/frankran/Repos/Research-Chatbot/data/papers/arxiv",  # Specify an existing directory
    max_results=3,
)
print(f"Successfully retrieved {len(search_docs)} documents")


print("Documents:")
print(search_docs[0])  # Print the text of the first document

print("=====")
print("Abstracts:")
print(abtracts[0])  # Print the abstract of the first document
