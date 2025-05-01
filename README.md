# Research Agent

A powerful research assistant that helps you analyze and interact with academic papers from various sources including arXiv and Zotero.

## Features

- Fetch and analyze papers from arXiv
- Import and process papers from Zotero
- automatically generate literature review survey to answer research questions
- PDF processing and text extraction
- Streamlit-based web interface
- RESTful API for programmatic access

## Prerequisites

- Python 3.11 or higher
- OpenAI API key
- LlamaParse API Key

## Installation

1. Clone the repository:
```bash
git clone https://github.com/FRANK-RAN/Research-Agent
cd Research-Agent
```

2. Create and activate a virtual environment:
```bash
python -m venv .literature
source .literature/bin/activate  # On Windows, use: .literature\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set up environment variables:
Create a `.env` file in the root directory with:
```
OPENAI_API_KEY=your_api_key_here
LLAMA_CLOUD_API_KEY=your_api_key_here
```

## Usage

### Starting the Application

1. Make the start script executable:
```bash
chmod +x start_app.sh
```

2. Start the application:
```bash
./start_app.sh
```

This will start:
- Backend API at http://localhost:8000
- Frontend UI at http://localhost:8501

### Stopping the Application

```bash
./stop_app.sh
```

## Project Structure

- `src/` - Source code directory
- `research_output/` - Research analysis outputs
- `streamlit_app.py` - Streamlit frontend
- `web_api.py` - FastAPI backend

## API Documentation

The backend API provides endpoints for:
- Paper fetching and processing
- Chat interactions
- Document analysis
- Search functionality


## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the terms of the LICENSE file in the root directory of this repository.

## Support

For support, please open an issue in the repository or contact the maintainers.
