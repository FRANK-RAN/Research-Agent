import os
import shutil
import hashlib

def get_file_hash(filepath):
    """
    Generate a hash for a file to detect duplicates.
    
    Args:
        filepath (str): Path to the file
    
    Returns:
        str: MD5 hash of the file contents
    """
    with open(filepath, 'rb') as f:
        return hashlib.md5(f.read()).hexdigest()

def extract_unique_pdfs(source_dir, destination_dir):
    """
    Extract unique PDF files from source directory to destination directory.
    
    Args:
        source_dir (str): Root directory to start searching for PDFs
        destination_dir (str): Directory to copy unique PDFs to
    """
    # Create destination directory if it doesn't exist
    os.makedirs(destination_dir, exist_ok=True)
    
    # Set to keep track of unique file hashes
    unique_file_hashes = set()
    
    # Set to keep track of unique file names (to handle potential naming conflicts)
    unique_filenames = set()
    
    # Walk through all directories and subdirectories
    for root, dirs, files in os.walk(source_dir):
        for file in files:
            # Check if file is a PDF
            if file.lower().endswith('.pdf'):
                full_path = os.path.join(root, file)
                
                # Generate file hash
                file_hash = get_file_hash(full_path)
                
                # Check if this is a unique file (by content)
                if file_hash not in unique_file_hashes:
                    # Generate a unique filename to prevent overwriting
                    base_name = os.path.splitext(file)[0]
                    ext = os.path.splitext(file)[1]
                    
                    # Find a unique filename
                    counter = 1
                    new_filename = file
                    while new_filename in unique_filenames:
                        new_filename = f"{base_name}_{counter}{ext}"
                        counter += 1
                    
                    # Copy the file to destination
                    destination_path = os.path.join(destination_dir, new_filename)
                    shutil.copy2(full_path, destination_path)
                    
                    # Add to tracking sets
                    unique_file_hashes.add(file_hash)
                    unique_filenames.add(new_filename)
                    
                    print(f"Copied: {full_path} -> {destination_path}")
    
    print(f"\nTotal unique PDFs copied: {len(unique_file_hashes)}")

# Example usage (replace with your actual paths)
# source_directory = '/path/to/.../files'
# destination_directory = '/path/to/quantum_flutatation'
# extract_unique_pdfs(source_directory, destination_directory)

if __name__ == "__main__":
    # Prompt user to input paths
    source_directory = input("Enter the full path to the source directory containing PDF files: ")
    destination_directory = input("Enter the full path to the destination directory (quantum_flutatation): ")
    
    # Validate paths
    if not os.path.isdir(source_directory):
        print("Error: Source directory does not exist.")
    else:
        # Run the extraction
        extract_unique_pdfs(source_directory, destination_directory)