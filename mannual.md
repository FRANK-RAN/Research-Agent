# Research Agent - User Manual

## Setup

### 1. Navigate to Research-Agent Directory
Find the directory where your Research-Agent is installed and navigate to it, for example:
```bash
cd Research-Agent
```

### 2. Activate the Virtual Environment

**On Linux/Mac:**
```bash
source .literature/bin/activate
```

**On Windows:**
```bash
.literature\Scripts\activate
```

## Running the Application

### 1. Starting the Application

**On Linux/Mac:**
```bash
./start_app.sh
```

**On Windows:**
```bash
.\start_app.bat
```

This will launch:
- Backend API server at http://localhost:8000
- Frontend UI interface at http://localhost:8501

### 2. Stopping the Application

**On Linux/Mac:**
```bash
./stop_app.sh
```

**On Windows:**
```bash
.\stop_app.bat
```

## Troubleshooting

If you encounter any issues:
1. Ensure you're in the correct directory
2. Verify the virtual environment is properly activated
3. Check that all dependencies are installed
4. Confirm ports 8000 and 8501 are not in use by other applications

## Additional Resources

For more information, visit the project documentation or contact support.