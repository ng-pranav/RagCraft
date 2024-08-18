import os
import pandas as pd
import streamlit as st
from langchain_core.prompts import ChatPromptTemplate
from langchain import PromptTemplate
from langchain_aws import BedrockLLM, BedrockChat
from langchain_core.output_parsers import JsonOutputParser

import boto3

session = boto3.Session(profile_name='default')


# Function to prepare batch input with corresponding file names
def prepare_batch_input(files):
    batch_inputs = []
    filenames = []
    
    for file in files:
        markdown_text = file.read().decode("utf-8")
        batch_input = {
            'markdown_content': markdown_text
        }
        batch_inputs.append(batch_input)
        filenames.append(file.name)
    
    return batch_inputs, filenames

# Function to extract fields from markdown using BedrockChat model
def extract_fields_from_markdown(batch_inputs):
    # Model Import
    model = BedrockChat(
        credentials_profile_name='default',
        model_id='meta.llama3-70b-instruct-v1:0',
        model_kwargs={
            "prompt": "string",
            "temperature": 0,
            "top_p": 0.9,
            "max_gen_len": 2048
        }
    )

    # Prompt
    prompt = PromptTemplate(
        input_variables=["markdown_content"],
        template="""
        Please process the following input text and provide a structured JSON output.

        Input text: {markdown_content}

        Your response should be a JSON object with the following structure:
        {{
            "seller_name": "extracted_seller_name",
            "date_of_purchase": "extracted_date_of_purchase.The extracted_date_of_purchase should be in dd-mm-yyyy format.
                                If the format is different from dd-mm-yyyy then convert into that format then give the output.",
            "address": "extracted_address_of_seller",
            "country": "extracted_country_of_seller",
            "items":[{{"item_name": "extracted_item_name","price": "extracted_price","quantity": "extracted_quantity",
                    "unit_of_quantity":  mention suitable from "pcs", "kg", "meter",
                    "item_category":mention suitable from "Utility","Medical","Lifestyle","Grocery","Transportation","Housing","Education","Financial Obligations","Miscellaneous".]}},....]
            "total_payment": "extracted_total",
            "mode_of_payment": "extracted_mode_of_payment"
        }}
        
        Please ensure the following:
        - The JSON should be properly formatted and includes all the required fields if they are available otherwise give None 
        - Strictly do not add any text to response other than Json object
        """
    )

    # Chain
    chain = prompt | model | JsonOutputParser()

    # Response
    responses = chain.batch(batch_inputs)
    return responses

# Function to update master excel sheet
def update_master_excel(new_data, master_file="master_excel.xlsx"):
    if os.path.exists(master_file):
        master_df = pd.read_excel(master_file)
    else:
        master_df = pd.DataFrame()

    updated_df = pd.concat([master_df, new_data], ignore_index=True)
    updated_df.to_excel(master_file, index=False)
    return updated_df

# Streamlit App
def main():
    st.title("Markdown to Excel Extractor")

    col1, col2 = st.columns(2)

    with col1:
        st.header("Upload Markdown Files")
        uploaded_files = st.file_uploader("Choose markdown files", type="md", accept_multiple_files=True)
        if uploaded_files:
            batch_inputs, filenames = prepare_batch_input(uploaded_files)
            st.write(f"Uploaded {len(uploaded_files)} files")

            # Getting entire markdown
            markdown_full= ('\n\n---\n\n').join([x['markdown_content'] for x in batch_inputs])
            st.markdown(markdown_full)

            responses = extract_fields_from_markdown(batch_inputs)

            # Convert responses to DataFrame
            df = pd.DataFrame(responses)

            # Explode the 'items' column to separate rows
            exploded_df = df.explode('items')

            # Normalize the 'items' column to expand dictionary keys into columns
            items_df = pd.json_normalize(exploded_df['items'])

            # Concatenate the original data (excluding the 'items' column) with the expanded items_df
            result_df = pd.concat([exploded_df.drop(columns=['items']).reset_index(drop=True), items_df.reset_index(drop=True)], axis=1)

            # Convert 'price' and 'quantity' columns to numeric types
            result_df['price'] = pd.to_numeric(result_df['price'], errors='coerce')
            result_df['quantity'] = pd.to_numeric(result_df['quantity'], errors='coerce')

            # Calculate 'price_per_item'
            result_df['price_per_item'] = result_df['price'] * result_df['quantity']

            # Store the DataFrame in session state for further use
            st.session_state['extracted_df'] = result_df
            st.session_state['filenames'] = filenames

    with col2:
        st.header("Review and Correct Extracted Fields")
        if 'extracted_df' in st.session_state:
            st.write("Extracted Fields DataFrame:")

            # Display the DataFrame and allow editing
            edited_df = st.data_editor(st.session_state['extracted_df'], num_rows="dynamic")

            if st.button("Save Corrections"):
                master_df = update_master_excel(edited_df)
                st.write("Master Excel Sheet Updated")
                st.write(master_df)

if __name__ == "__main__":
    main()
