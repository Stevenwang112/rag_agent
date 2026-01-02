
import os
import base64
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import SupabaseVectorStore
from rag import embeddings, supabase

load_dotenv()

PDF_PATH = "resouce/星辰电动ES9·未来旗舰电动SUV产品介绍.pdf"
OUTPUT_MD_PATH = "resouce/ES9_tables.md"

def extract_tables_with_gemini():
    print(f"Reading PDF: {PDF_PATH}...")
    try:
        with open(PDF_PATH, "rb") as f:
            pdf_data = f.read()
    except FileNotFoundError:
        print("Error: PDF file not found.")
        return False
        
    llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0)
    
    print("Sending to Gemini for Table Extraction...")
    message = HumanMessage(
        content=[
            {
                "type": "text", 
                "text": "Please scan this document and extract ONLY the technical specification tables, configuration lists, and parameter charts. Ignore general marketing paragraphs. Convert all tables into clean Markdown format. Pay special attention to numbers regarding Range (续航), Battery (电池), Dimensions (尺寸), and Performance. If there is a range like '600-850km', ensure it is transcripted exactly."
            },
            {
                "type": "media",
                "mime_type": "application/pdf",
                "data": base64.b64encode(pdf_data).decode("utf-8")
            }
        ]
    )

    try:
        response = llm.invoke([message])
        content = response.content
        
        with open(OUTPUT_MD_PATH, "w") as f:
            f.write(content)
            
        print(f"Tables saved to {OUTPUT_MD_PATH}")
        print(f"Content Preview:\n{content[:200]}...")
        return True
    except Exception as e:
        print(f"Extraction Failed: {e}")
        return False

def ingest_tables_to_supabase():
    print("\nIngesting extracted tables to Supabase...")
    try:
        loader = TextLoader(OUTPUT_MD_PATH)
        documents = loader.load()
        
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        docs = text_splitter.split_documents(documents)
        
        print(f"Splitting into {len(docs)} chunks...")
        
        # Add metadata to distinguish this from the main doc
        for doc in docs:
            doc.metadata["source"] = "ES9 Product Specs (Tables)"
            doc.metadata["is_table_data"] = "true"

        vector_store = SupabaseVectorStore.from_documents(
            docs,
            embeddings,
            client=supabase,
            table_name="documents",
            query_name="match_documents",
            chunk_size=500,
        )
        print("Ingestion Done!")
    except Exception as e:
        print(f"Ingestion failed: {e}")

if __name__ == "__main__":
    if extract_tables_with_gemini():
        ingest_tables_to_supabase()
