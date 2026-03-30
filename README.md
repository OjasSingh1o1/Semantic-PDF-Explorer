# Semantic PDF Explorer

A CLI tool that lets you search through a PDF using natural language queries. Instead of keyword matching, it uses AI to understand the *meaning* of your query and returns the most relevant sections of the document.

## What it does

Extract text from a PDF using a library like pypdf
Split the text into smaller chunks/paragraphs
Use Hugging Face Zero-Shot Classification model (`facebook/bart-large-mnli`)to compare a query with each chunk
Display chunks with high confidence scores (e.g., above 80%)


## Pipeline
```
PDF -> Text Extraction (pypdf) -> Chunking -> Zero-Shot Classification (HuggingFace) -> Filtered Results
```

## NOTE

**1. Create and activate virtual environment**
```
python3 -m venv venv
source venv/bin/activate
```

**2. Install dependencies**
```
pip install -r requirements.txt

## Sample queries and outputs

**Query:** `Assignment tasks`

![Query output](images/query-output.png)

## Approach

1. Used `pypdf` to extract raw text from the PDF page by page
2. Split the full text into chunks of 200 words each to keep inputs within model limits
3. Loaded `facebook/bart-large-mnli` — a transformer model trained on natural language inference
4. For each query, ran zero-shot classification treating the query as a candidate label
5. Filtered and sorted results by confidence score, displaying only chunks above 80%

## Difficulties faced

- **File path with quotes** — drag and drop on Mac wraps paths in single quotes, causing `FileNotFoundError`. Fixed by adding `.strip("'\"")` to clean the input, or if not that then just make sure the name of the pdf you're uploading does not have soaces in it like mine did.


## Tech stack

- Python 3.13
- pypdf
- HuggingFace Transformers
- PyTorch
- facebook/bart-large-mnli
