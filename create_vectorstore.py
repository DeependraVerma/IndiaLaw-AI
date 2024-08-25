import os
import faiss
import pandas as pd
from uuid import uuid4
from dotenv import load_dotenv
from langchain.schema import Document
from langchain_community.vectorstores import FAISS
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.document_loaders import CSVLoader

# Load environment variables
load_dotenv()
api_key = os.getenv("GOOGLE_GENAI_API_KEY")
DB_FAISS_PATH = "vector_store/faiis_cpu"

# Initialize Google Generative AI embeddings
embedding_model = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=api_key)

# Load the CSV file containing case data
df = pd.read_csv("top_judgments.csv")

loader = CSVLoader(file_path="top_judgments.csv", encoding="utf-8", csv_args={
                'delimiter': ','}
)
data = loader.load()

embedding_dim = len(embedding_model.embed_documents([data[0].page_content])[0])
index = faiss.IndexFlatL2(embedding_dim)
#index = faiss.IndexFlatL2(len(embedding_model.embed_query(data)))
# Initialize FAISS vector store
vector_store = FAISS(
    embedding_function=embedding_model,
    index=index,
    docstore= InMemoryDocstore(),
    index_to_docstore_id={}
)
uuids = [str(uuid4()) for _ in range(len(data))]
# Add documents to the vector store
vector_store.add_documents(documents=data, ids=uuids)
# Save the FAISS index to disk
vector_store.save_local(DB_FAISS_PATH)
print("Vector store saved to disk.")