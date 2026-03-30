from pypdf import PdfReader
from transformers import pipeline

def extract_text(pdf_path):
    reader = PdfReader(pdf_path)
    full_text = ""
    for page in reader.pages:
        full_text += page.extract_text() + "\n"
    return full_text

def split_into_chunks(text, chunk_size=180):
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size):
        chunk = " ".join(words[i:i + chunk_size])
        chunks.append(chunk)
    return chunks

def load_classifier():
    print("\nLoading AI model... (first run takes a minute)\n")
    classifier = pipeline("zero-shot-classification", 
                         model="facebook/bart-large-mnli")
    return classifier

def search(chunks, query, classifier, threshold=0.80):
    print(f"Searching for: '{query}'\n")
    print("-" * 60)
    
    results = []
    for i, chunk in enumerate(chunks):
        output = classifier(chunk, candidate_labels=[query])
        score = output["scores"][0]
        if score >= threshold:
            results.append((score, chunk))
    
    results.sort(reverse=True)
    
    if not results:
        print("No chunks found above 80% confidence.")
        print("Try a broader query or lower the threshold.")
    else:
        for score, chunk in results:
            print(f"Confidence: {score:.2%}")
            print(f"Chunk: {chunk[:300]}...")
            print("-" * 60)
    
    return results

def main():
    pdf_path = input("Enter path to your PDF file: ").strip().strip("'\"")
    
    print("\nExtracting text from PDF...")
    text = extract_text(pdf_path)
    
    print(f"Splitting into chunks...")
    chunks = split_into_chunks(text)
    print(f"Total chunks created: {len(chunks)}")
    
    classifier = load_classifier()
    
    print("\nSemantic PDF Explorer ready.")
    print("Type your query and press Enter. Type 'quit' to exit.\n")
    
    while True:
        query = input("Query: ").strip()
        if query.lower() == "quit":
            print("Exiting.")
            break
        if not query:
            continue
        search(chunks, query, classifier)

if __name__ == "__main__":
    main()

