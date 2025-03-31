from dotenv import load_dotenv
load_dotenv()
# Import necessary libraries
from llama_parse import LlamaParse
from llama_index.core import SimpleDirectoryReader
from llama_index.core import StorageContext, load_index_from_storage

# Set up the PDF parser
parser = LlamaParse(
    result_type="markdown"  # Options: "markdown" or "text"
)

# Use SimpleDirectoryReader to read and parse the file
file_extractor = {".pdf": parser}
documents = SimpleDirectoryReader(
    input_dir="/Users/frankran/Repos/Research-Chatbot/data/papers/quantum_fluctuation", 
    file_extractor=file_extractor
).load_data()

# Display a summary of parsed documents
print("Parsed Documents Summary:")
print(f"Total Documents: {len(documents)}")


# one extra dep
from llama_index.core import VectorStoreIndex

# create an index from the parsed markdown
index = VectorStoreIndex.from_documents(documents)
index.set_index_id("optically") # Set a unique index ID
index.storage_context.persist(persist_dir="/Users/frankran/Repos/Research-Chatbot/database/indices/quantum_fluctuation")  # Persist the index to disk

