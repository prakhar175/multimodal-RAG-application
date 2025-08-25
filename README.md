# 🌟 Multimodal RAG: Advanced PDF Question-Answering System

A sophisticated **Retrieval-Augmented Generation (RAG)** system that processes both text and images from PDF documents using CLIP embeddings and large language models for comprehensive document understanding.

## 📋 Table of Contents

- [Features](#features)
- [Architecture](#architecture)
- [Installation](#installation)
- [Configuration](#configuration)
- [Usage](#usage)
- [How It Works](#how-it-works)
- [License](#license)

## ✨ Features

- **🔄 Multimodal Processing**: Handles both text and images from PDF documents
- **🧠 CLIP Embeddings**: Uses OpenAI's CLIP model for unified text-image embedding space
- **📊 Vector Search**: FAISS-powered similarity search for efficient retrieval
- **🤖 Smart Chunking**: Intelligent text splitting with RecursiveCharacterTextSplitter
- **🖼️ Image Processing**: Extracts and processes images with base64 encoding
- **💬 Conversational AI**: Powered by Meta's Llama model via Groq API
- **🎯 Context-Aware**: Combines text and visual context for comprehensive answers

## 🏗️ Architecture

```
PDF Document
    ↓
┌─────────────────┬─────────────────┐
│   Text Content  │  Image Content  │
│       ↓         │       ↓         │
│  Text Chunking  │ Image Extraction│
│       ↓         │       ↓         │
│ CLIP Text Embed │ CLIP Image Embed│
└─────────────────┴─────────────────┘
    ↓
FAISS Vector Store
    ↓
Query → Similarity Search → Context Retrieval
    ↓
LLM (Llama-4) → Final Answer
```

## 🚀 Installation

### Prerequisites

- Python 3.10 (recommended)

### Install Dependencies

```bash
# Clone the repository
git clone https://github.com/prakhar175/multimodal-RAG-application.git
cd multimodal-RAG-application

# Create virtual environment
python -m venv myenv
myenv\Scripts\activate

# Install required packages
pip install -r req.txt
```

## ⚙️ Configuration

### Environment Setup

Create a `.env` file in your project root:

```env
# Groq API Key for Llama model
GROQ_API_KEY=your_groq_api_key_here

# Hugging Face Token (optional, for private models)
HF_TOKEN=your_huggingface_token_here
```

### Get API Keys

1. **Groq API**: Sign up at [Groq Console](https://console.groq.com) for Llama model access
2. **Hugging Face**: Create account at [Hugging Face](https://huggingface.co) for model downloads

## 🎯 Usage

### Jupyter Notebook

1. Open `code.ipynb`
2. Update the `pdf_path` variable with your PDF file
3. Run all cells to process the document
4. Use the `multimodal_rag_pipeline()` function to ask questions

### Example Queries

```python
queries = [
    "What is the aim of this guidance?",
    "Explain the chart in Figure 10",
    "What does Anscombe's quartet demonstrate?",
    "Summarize the key findings from the images"
]

for query in queries:
    print(f"\nQuery: {query}")
    print("-" * 50)
    answer = multimodal_rag_pipeline(query)
    print(f"Answer: {answer}")
    print("=" * 70)
```

## 🔧 How It Works

### 1. Document Processing
- **Text Extraction**: Extracts text content from each PDF page
- **Image Extraction**: Identifies and extracts embedded images
- **Chunking**: Splits text into manageable chunks (500 chars with 30 char overlap)

### 2. Embedding Generation
- **CLIP Model**: Uses `openai/clip-vit-base-patch32` for unified embeddings
- **Text Embeddings**: Converts text chunks to 512-dimensional vectors
- **Image Embeddings**: Converts images to same 512-dimensional space

### 3. Vector Storage
- **FAISS Index**: Creates searchable vector database
- **Metadata**: Stores page numbers, content types, and image references
- **Base64 Storage**: Images stored as base64 strings for LLM consumption

### 4. Query Processing
- **Similarity Search**: Finds most relevant text/image chunks (k=5 default)
- **Context Assembly**: Combines retrieved text and images
- **LLM Generation**: Uses Llama-4 to generate contextual answers


## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

**Built with ❤️ for multimodal AI applications**
