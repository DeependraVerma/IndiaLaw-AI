import streamlit as st
import pandas as pd
from dotenv import load_dotenv
from fetch_case_content import fetch_case_content
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
import os
import re

# Load environment variables
load_dotenv()

user_api_key = st.sidebar.text_input("API Key", type="password")
api_key = genai.configure(api_key=user_api_key)
generation_config = {
  "temperature": 0.2,
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


embedding_model = GoogleGenerativeAIEmbeddings(model_name="models/embedding-001", google_api_key=user_api_key)

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
    

def generate_response(case_content, query):
    """Generates a response based on the case content and query."""
    prompt = f"Act as an expert of Indian Legal cases by different courts. Your task is to give full and structured detail based on the following case content, answer the query:\n\n{case_content}\n\nQuery: {query}. Also, make the output response attractive and by removing '\n' and other unnecesary part."
    response = chat_session.send_message(prompt)
    return response.text



# Streamlit UI
st.title("IndiaLaw-AI: Legal Case Chatbot")

query = st.text_input("Ask a legal case question:")

if query:
    relevant_case = get_relevant_case(query)

    if relevant_case:
        page_content = relevant_case.page_content
        match = re.search(r'title:\s*(.*)', page_content)
        if match:
            title = match.group(1).strip()
            case_row = df[df['title'] == title]
            if not case_row.empty:
                print("Case row found:\n", case_row)
            else:
                print("No matching case row found.")
        else:
            print("Title not found in page_content")
        case_url = case_row['url'].values[0]
        case_title = case_row['title'].values[0]

        st.write(f"Fetching content for: **{case_title}**")

        case_content = fetch_case_content(case_url)

        if case_content:
            response = generate_response(case_content, query)

            st.write("**Answer:**")
            st.write(response)

        else:
            st.error("Failed to retrieve case content.")
    else:
        st.error("No relevant case found.")
