import streamlit as st
import pandas as pd
from dotenv import load_dotenv
from fetch_case_content import fetch_case_content
from your_embeddings_module import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAI
import google.generativeai as genai
import os

# Load environment variables
load_dotenv()
api_key = genai.configure(api_key=os.getenv("GOOGLE_GENAI_API_KEY"))
api_key_embed = os.getenv("GOOGLE_GENAI_API_KEY")
generation_config = {
  "temperature": 1,
  "top_p": 0.95,
  "top_k": 64,
  "max_output_tokens": 8192,
  "response_mime_type": "text/plain",
}
model = genai.GenerativeModel(
  model_name="gemini-1.5-flash",
  generation_config=generation_config,
  # safety_settings = Adjust safety settings
  # See https://ai.google.dev/gemini-api/docs/safety-settings
)

chat_session = model.start_chat(
  history=[
  ]
)


embedding_model = GoogleGenerativeAIEmbeddings(model_name="models/embedding-001", google_api_key=api_key_embed)

# Load the saved FAISS index
vector_store = FAISS.load_local("vector_store/faiis_cpu", embedding_model, allow_dangerous_deserialization=True)

# Load the CSV file containing case data
df = pd.read_csv("top_judgments.csv")

def get_relevant_case(query):
    """Retrieves the most relevant case based on the user's query."""
    query_embedding = embedding_model.embed_query(query)
    search_result = vector_store.similarity_search_by_vector(query_embedding, k=2)
    if search_result:
        relevant_case = search_result[0]
        return relevant_case
    return None

query = 'Rangappa vs Sri Mohan on 7 May, 2010'
case_content = get_relevant_case(query=query)
print(case_content)