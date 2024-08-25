from langchain_google_genai import GoogleGenerativeAIEmbeddings as OriginalGoogleGenerativeAIEmbeddings
import google.generativeai as genai
from dotenv import load_dotenv
from app import user_api_key
import os

load_dotenv()
api_key_embed = os.getenv("GOOGLE_GENAI_API_KEY")

class GoogleGenerativeAIEmbeddings:
    def __init__(self, model_name, google_api_key):
        # Initialize the embeddings model with the correct model name and API key
        self.model = OriginalGoogleGenerativeAIEmbeddings(model=model_name, google_api_key=user_api_key)

    def embed_documents(self, texts):
        # Embed the document text and return the embeddings
        return self.model.embed_documents(texts)

    def embed_query(self, query):
        # Embed the query and return the embedding
        return self.model.embed_query(query)
