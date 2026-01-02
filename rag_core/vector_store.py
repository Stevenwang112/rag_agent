from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv
from langchain_community.vectorstores import SupabaseVectorStore
from supabase.client import Client, create_client
import os
load_dotenv()

embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")



supabase_url = os.environ.get("SUPABASE_URL")
supabase_key = os.environ.get("SUPABASE_SERVICE_KEY")
supabase: Client = create_client(supabase_url, supabase_key)

def get_vector_store():
    return SupabaseVectorStore(
        embedding=embeddings,
        client=supabase,
        table_name="documents",
        query_name="match_documents",
    )

def ingest_documents():
    from langchain_community.document_loaders import PyPDFLoader
    from langchain_text_splitters import CharacterTextSplitter

    loader = PyPDFLoader("resouce/星辰电动ES9·未来旗舰电动SUV产品介绍.pdf")
    documents = loader.load()
    text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    docs = text_splitter.split_documents(documents)

    vector_store = SupabaseVectorStore.from_documents(
        docs,
        embeddings,
        client=supabase,
        table_name="documents",
        query_name="match_documents",
        chunk_size=500,
    )







if __name__ == "__main__":
    ingest_documents()