import os
from dotenv import load_dotenv
import tempfile
import camelot
import pandas as pd
import streamlit as st
from transformers import pipeline, AutoTokenizer, AutoModelForQuestionAnswering

# Load environment variables from .env file
load_dotenv()

# Retrieve the Hugging Face API token
hf_api_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")

# Ensure the token is loaded
if not hf_api_token:
    raise ValueError("Hugging Face API token not found. Please check your .env file.")

# Function to extract tables from a PDF
def extract_tables_from_pdf(uploaded_pdf):
    # Save the PDF from the upload to a temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as temp_pdf:
        temp_pdf.write(uploaded_pdf.getbuffer())
        temp_pdf_path = temp_pdf.name

    # Now that we have a file path, we can use Camelot to read the PDF
    tables = camelot.read_pdf(temp_pdf_path, pages='all', flavor='lattice')

    return [table.df for table in tables]

# Function to answer questions using Hugging Face's Transformers
def answer_question(question, data):
    # Initialize the tokenizer and model for question answering
    tokenizer = AutoTokenizer.from_pretrained("bert-large-uncased-whole-word-masking-finetuned-squad", use_auth_token=hf_api_token)
    model = AutoModelForQuestionAnswering.from_pretrained("bert-large-uncased-whole-word-masking-finetuned-squad", use_auth_token=hf_api_token)

    # Initialize the pipeline
    qa_pipeline = pipeline("question-answering", model=model, tokenizer=tokenizer)

    # Generate a response
    response = qa_pipeline(question=question, context=data)
    return response['answer']

def main():
    st.title('PDF Table Extractor and Question Answering System')

    # Step 1: User uploads a PDF
    uploaded_pdf = st.file_uploader("Upload a PDF containing tables", type="pdf")
    
    if uploaded_pdf:
        # Step 2: Extract tables from the PDF
        try:
            extracted_tables = extract_tables_from_pdf(uploaded_pdf)
            table_names = [f"Table {i+1}" for i in range(len(extracted_tables))]
            st.success('Tables extracted successfully!')
        except Exception as e:
            st.error(f'An error occurred when extracting tables: {e}')
            return

        # Sidebar to select a table to view
        table_to_view = st.sidebar.selectbox("Select a table to view", table_names)
        view_index = table_names.index(table_to_view)
        st.subheader(f'Viewing {table_to_view}')
        st.dataframe(extracted_tables[view_index])

        # Sidebar to select a table to ask a question on
        table_to_question = st.sidebar.selectbox("Select a table to ask a question on", table_names)
        question_index = table_names.index(table_to_question)

        # User inputs a question
        user_question = st.sidebar.text_input(f"Enter your question related to {table_to_question}")

        if user_question:
            # Utilize Language Model
            data_as_string = extracted_tables[question_index].to_string(index=False)
            answer = answer_question(user_question, data_as_string)

            # Present the answer
            st.subheader('Answer')
            st.write(answer)

if __name__ == "__main__":
    main()
