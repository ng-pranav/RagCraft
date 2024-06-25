import json
import os
import pandas as pd
from langchain_core.prompts import ChatPromptTemplate
from langchain import PromptTemplate
from langchain_aws import BedrockLLM, BedrockChat
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser

# Function to prepare batch input with corresponding file names
def prepare_batch_input(input_folder):
    batch_inputs = []
    filenames = []
    
    # Loop through all markdown files in the input folder
    for filename in os.listdir(input_folder):
        if filename.endswith(".md"):
            # Read the content of the markdown file
            file_path = os.path.join(input_folder, filename)
            with open(file_path, 'r', encoding='utf-8') as file:
                markdown_text = file.read()
            
            # Prepare the input dictionary for each file
            batch_input = {
                'markdown_content': markdown_text
            }
            batch_inputs.append(batch_input)
            filenames.append(filename)
    
    return batch_inputs, filenames

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

# Prepare batch inputs
input_folder = 'input_files'
batch_inputs, filenames = prepare_batch_input(input_folder)

# Response
responses = chain.batch(batch_inputs)

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

# Output file path
output_file = 'extracted_fields.xlsx'

# Write the DataFrame to an Excel file
result_df.to_excel(output_file, index=False)

print(f"Output written to {output_file}")
