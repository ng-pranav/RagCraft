import json
import os
import pandas as pd
from langchain_core.prompts import ChatPromptTemplate
from langchain_aws import BedrockLLM, BedrockChat
from langchain_core.output_parsers import StrOutputParser

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
prompt = ChatPromptTemplate.from_template(
    "Please assist by analyzing the provided markdown text and answering the user's question in a clear, concise, and accurate manner. Ensure that the response directly addresses the inquiry with relevant details.\n\n"
    "User's Question:\n{question}\n\n"
    "Markdown Text:\n{markdown_text}\n\n"
    "Instructions:\n"
    "1. Read the markdown text carefully.\n"
    "2. Understand the context and the details provided in the text.\n"
    "3. Formulate a response that directly answers the user's question.\n"
    "4. Ensure the response is specific, accurate, and relevant to the question.\n"
    "5. Keep the response clear and concise.\n\n"
    "Thank you for your assistance."
)


# Chain
chain = prompt | model | StrOutputParser()

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

# Output file path
output_file = 'extracted_output.xlsx'

# Write the DataFrame to an Excel file
df.to_excel(output_file, index=False)

print(f"Output written to {output_file}")
