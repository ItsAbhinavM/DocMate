# DocMate
Your own LLM agent for generating dataset from documents.
## Overview
**DocMate** is an Autonomous Document Intelligent tool that lets you chat with both structured and unstrured documents in your dataset. You can upload files, ask questions and get structured responses instantly. It support schema generation, data extraction, refinement and dynamic follow-ups. The front-end features voice input, multiple file upload and chat history.
## Features and Technical Highlights
- **Multi Modal Support** <br> 
Works with structure and unstructured files and parsed using `unstructured` library,
- **RAG based workflow**<br>
Uses Retrieval Augmented Geneation to enhance accuracy, retrieving releavant documents chunks before extraction.
- **LangGraph**<br>
Conditional graph-based flow for managing complex state transitions.
- **FAISS Vector Store**<br>
Similarity check for document and chunk retrieval.
- **Dataset Wide Statistics**<br>
Generates statistics of the whole dataset which displays and saves locally.
- **Speech-to-Text Input**
- **Multiple File upload**

## Getting Started
Follow these steps to setup DOCMATE on your local machine.
### Prerequisite
- **Tauri** and **npm**
- API keys for **Azure STT** and **Gemini**
### Installation
1. Clone the repository:

   ```bash
   git clone https://github.com/ItsAbhinavM/DocMate.git
   cd DocMate
   ```

2. Install dependencies:

   ```bash
   npm install
   pip install requirements.txt
   ```

3. Set up environment variables:

   Create a `.env` file in the root directory with the necessary API keys:

   ```plaintext
   AZURE_API_KEY=<YOUR_AZURE_API_KEY>
   GOOGLE_API_KEY=<YOUR_GOOGLE_GEMINI_API_KEY>
   ```

4. Start the development server:

   ```bash
   npm run tauri build
   fastapi run main.py
   ```

