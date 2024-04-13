import tempfile
import camelot
import pandas as pd
import streamlit as st
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

def extract_tables_from_pdf(uploaded_pdf):
    with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as temp_pdf:
        temp_pdf.write(uploaded_pdf.getbuffer())
        temp_pdf_path = temp_pdf.name

    tables = camelot.read_pdf(temp_pdf_path, pages='all', flavor='lattice')
    return [table.df for table in tables]

def preprocess_dataframe_for_flan(df):
    df = df.astype(str).fillna("N/A")
    df.columns = df.columns.map(str)
    df.reset_index(drop=True, inplace=True)
    return df

def convert_table_to_text(table_df):
    text_representation = []
    columns = table_df.columns.tolist()
    for _, row in table_df.iterrows():
        row_texts = [f"{col} is {row[col]}" for col in columns]
        row_text = ", ".join(row_texts)
        text_representation.append(row_text)
    return " ".join(text_representation)

def answer_question_with_flan_t5(question, table_df):
    table_text = convert_table_to_text(table_df)
    prompt = f"Question: {question}\nTable: {table_text}\nAnswer:"

    # Load the saved tokenizer and model
    model_directory = "model_directory_base"
    tokenizer = AutoTokenizer.from_pretrained(model_directory)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_directory)

    inputs = tokenizer.encode(prompt, return_tensors="pt")
    outputs = model.generate(inputs)
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)

    return answer

def main():
    st.set_page_config(layout="wide")
    st.title('PDF Table Extractor and Question Answering System')

    uploaded_pdf = st.file_uploader("Upload a PDF containing tables", type="pdf")
    
    if uploaded_pdf:
        try:
            extracted_tables = extract_tables_from_pdf(uploaded_pdf)
            table_names = [f"Table {i+1}" for i in range(len(extracted_tables))]
            st.success('Tables extracted successfully!')
        except Exception as e:
            st.error(f'An error occurred when extracting tables: {e}')
            return

        st.sidebar.markdown("## Select Table")
        table_to_view = st.sidebar.selectbox("View Table", table_names, index=0)
        view_index = table_names.index(table_to_view)
        st.subheader(f'Viewing {table_to_view}')
        st.dataframe(extracted_tables[view_index])

        st.sidebar.markdown("## Ask a Question")
        table_to_question = st.sidebar.selectbox("Ask Question On", table_names, index=0)
        question_index = table_names.index(table_to_question)
        user_question = st.sidebar.text_input("Enter your question", key="question_input")

        if user_question:
            preprocessed_table = preprocess_dataframe_for_flan(extracted_tables[question_index])
            answer = answer_question_with_flan_t5(user_question, preprocessed_table)
            st.subheader('Answer')
            st.write(answer)

if __name__ == "__main__":
    main()
