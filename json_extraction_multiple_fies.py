import json
import os
import pandas as pd
from langchain_core.prompts import ChatPromptTemplate
from langchain_aws import BedrockLLM, BedrockChat
from langchain_core.output_parsers import StrOutputParser,JsonOutputParser

# Load the question from config.json file
with open('config.json', 'r') as config_file:
    config = json.load(config_file)
question = config['question']

# Function to prepare batch input with corresponding file names
def prepare_batch_input(input_folder, question):
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
                'question': question,
                'markdown_text': markdown_text
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
        "file_name": "extracted_file_name",
        "sellers_name": "extracted_sellers_name",
        "item_name": "extracted_item_name",
        "price": "extracted_price",
        "quantity": "extracted_quantity",
        "unit_of_quantity": ["numbers", "kg", "meter"],
        "total_payment": "extracted_total",
        "mode_of_payment": "extracted_mode_of_payment",
        "date_of_purchase": "extracted_date_of_purchase",
        "address": "extracted_address_of_seller",
        "country": "extracted_country_of_seller"
    }}
    
    Please ensure the following:
    - The Item Name, Quantity, and Price are separated into different fields and not combined into one field called "items."
    - The JSON is properly formatted and includes all the required fields.
    """
    )



# Chain
chain=prompt|model|JsonOutputParser()

# Inputs
input_folder = 'input_files'
question = question

# Prepare batch inputs
batch_inputs, filenames = prepare_batch_input(input_folder, question)
print(batch_inputs)
print(filenames)

# Response
responses = chain.batch(batch_inputs)

# Function to process responses and generate output data
def process_responses(responses, filenames):
    output_data = []

    # Ensure responses is always a list
    if not isinstance(responses, list):
        responses = [responses]

    # Ensure that the number of responses and filenames match
    if len(responses) != len(filenames):
        raise ValueError("The number of responses does not match the number of filenames")

    #Loop through responses and filenames to create output data
    for response, filename in zip(responses, filenames):
        output_data.append({
            'Filename': filename,
            'Extracted Information': response 
        })

    return output_data

# Process the responses
final_response = process_responses(responses, filenames)

# Convert the response to a DataFrame
df = pd.DataFrame(final_response)

df_expanded = pd.json_normalize(df['items'])

# Drop the 'items' column from df
df = df.drop(columns=['items'])

# Concatenate df and df_expanded, placing df_expanded columns between 'sellers_name' and 'total'
df_combined = pd.concat([df.iloc[:, :2], df_expanded, df.iloc[:, 2:]], axis=1)

df_combined['total']=df_combined['price']*df_combined['quantity']

# Output file path
output_file = 'extracted_output.xlsx'

# Write the DataFrame to an Excel file
df_combined.to_excel(output_file, index=False)

print(f"Output written to {output_file}")
