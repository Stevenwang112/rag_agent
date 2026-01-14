import os
import re
import time
from typing import List, Dict, Any
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from supabase import create_client, Client

load_dotenv()

# --- Configuration ---
MD_PATH = "resouce/SU7_data/SU7_mixed_content.md"
SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_SERVICE_KEY")

# Initialize Clients
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")

def classify_document(text_sample: str) -> str:
    """
    Uses DeepSeek to identify the car model from the document text.
    Returns: 'ES9', 'SU7', or 'Unknown'
    """
    llm = ChatOpenAI(
        model="deepseek-chat", 
        api_key=os.environ["DEEPSEEK_API_KEY"], 
        base_url="https://api.deepseek.com",
        temperature=0.0
    )
    
    prompt = f"""
    Read the following text sample from a document and find out the company name you read.
    
    Text Sample:
    {text_sample[:1000]}...
    
    Instructions:
    - Output ONLY the company or model name (e.g. 'ES9', 'SU7', 'Tesla').
    - If you find "StarEra" or "Exeed" or "ES9", output: ES9
    - If you find "Xiaomi" or "SU7", output: SU7
    - Do not output anything else.
    """
    
    try:
        result = llm.invoke(prompt).content.strip()
        # Basic cleanup: remove periods, quotes
        result = result.replace(".", "").replace('"', "").replace("'", "").strip()
        return result
    except Exception as e:
        print(f"⚠️ Classification failed: {e}")
        return "Unknown"

def load_markdown_by_page(file_path: str) -> List[Dict[str, str]]:
    """
    Parses the Markdown file split by '## Page X'.
    Returns a list of dicts: [{'page': 1, 'content': '...'}, ...]
    """
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()

    # Split by "## Page "
    # This regex looks for "## Page " followed by digits, capturing everything until the next match or EOF
    pages = []
    
    # Using regex to find all page sections
    # Pattern: ## Page (\d+)\n(.*?) (followed by next ## Page or end of string)
    # flag DOTALL allows . to match newlines
    pattern = re.compile(r"## Page (\d+)\n(.*?)(?=## Page \d+|$)", re.DOTALL)
    
    matches = pattern.findall(content)
    
    for page_num_str, page_text in matches:
        # Clean up the text
        clean_text = page_text.strip()
        
        # Remove the code block markers if they were wrapper around the whole page content by the LLM
        # e.g. ```markdown ... ```
        if clean_text.startswith("```markdown"):
            clean_text = clean_text.replace("```markdown", "", 1)
        if clean_text.startswith("```html"):
             clean_text = clean_text.replace("```html", "", 1)
        if clean_text.endswith("```"):
            clean_text = clean_text[:-3]
            
        clean_text = clean_text.strip()

        if clean_text:
            pages.append({
                "page": int(page_num_str),
                "content": clean_text
            })
            
    print(f"Loaded {len(pages)} pages from {file_path}")
    return pages

def ingest_parent_child():
    """
    Main logic:
    1. Load full pages (Parents)
    2. Auto-Classify (DeepSeek)
    3. Insert Parent -> Get ID
    4. Split Parent Content -> Child Chunks
    5. Embed & Insert Child Chunks linked to Parent ID + Company Tag
    """
    if not os.path.exists(MD_PATH):
        print(f"File not found: {MD_PATH}. Please run pdf_converter.py first.")
        return

    pages = load_markdown_by_page(MD_PATH)
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=600,
        chunk_overlap=100,
        separators=["\n\n", "\n", ".", " ", ""]
    )

    print("Starting ingestion to Supabase...")
    
    # 0. Auto-Classify Document based on Page 1
    company_name = "Unknown"
    if pages:
        print("🤖 Auto-classifying document using DeepSeek...")
        company_name = classify_document(pages[0]["content"])
        print(f"✅ Identified Company/Model: {company_name}")

    for page_data in pages:
        page_num = page_data["page"]
        full_content = page_data["content"]
        
        print(f"--- Processing Page {page_num} ---")
        
        # 1. Insert Parent Document
        parent_record = {
            "content": full_content,
            "metadata": {"page": page_num, "source": "SU7_mixed_content.md"}
        }
        
        try:
            # Execute insert and return the inserted row to get the ID
            result = supabase.table("document_parents").insert(parent_record).execute()
            parent_id = result.data[0]['id']
            print(f"  Inserted Parent ID: {parent_id}")
        except Exception as e:
            print(f"  Error inserting parent page {page_num}: {e}")
            continue

        # 2. Create Child Chunks
        # We wrap the content in a Document object for the splitter
        doc_obj = Document(page_content=full_content)
        chunks = text_splitter.split_documents([doc_obj])
        
        print(f"  Split into {len(chunks)} chunks. Generating embeddings...")

        chunk_records = []
        for i, chunk in enumerate(chunks):
            chunk_text = chunk.page_content
            
            # Generate Embedding
            # Add simple retry logic for embedding API
            try:
                vector = embeddings.embed_query(chunk_text)
            except Exception as e:
                print(f"    Error embedding chunk {i}: {e}. Retrying once...")
                time.sleep(2)
                vector = embeddings.embed_query(chunk_text)

            chunk_records.append({
                "parent_id": parent_id,
                "content": chunk_text,
                "metadata": {
                    "chunk_index": i, 
                    "page_source": page_num,
                    "company_name": company_name
                },
                "embedding": vector
            })
            
            # Batch text-embedding-004 to avoid rate limits if needed, 
            # effectively we are doing 1 by 1 here but 2.0-flash is fast.
            # Adding a tiny sleep tailored for tier-1 limits if passing many requests
            time.sleep(0.2) 

        # 3. Insert Child Chunks
        if chunk_records:
            try:
                supabase.table("document_chunks").insert(chunk_records).execute()
                print(f"  Successfully stored {len(chunk_records)} chunks for Page {page_num}")
            except Exception as e:
                print(f"  Error inserting chunks for page {page_num}: {e}")

    print("\n✅ Ingestion Complete!")

if __name__ == "__main__":
    ingest_parent_child()
