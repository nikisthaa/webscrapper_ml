
"""
ML-Code.ipynb

Original file is located at
    https://colab.research.google.com/drive/1daS89TgZZy6kzapdXslRBpsWuuKGOt9_

    
Instructions - 

1 .Make sure you run the following command before -
pip3 install requests
pip3 install langchain
pip3 install openai
pip3 install bs4
pip3 install tiktoken

2. Use run_pipeline(url) function to extract staff data

"""




"""# API Keys"""

OPENAI_API_KEY = "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"

"""# Staff Directories Links"""

static_urls = [
    "https://yescharteracademy.org/school-information-and-contacts/",
    "https://westinghousearts.org/about/staff-directory/",
    "https://www.swanseaschools.org/domain/15",
    "https://www.southridgeprep.com/apps/staff/",
    "https://www.santiagocharterms.org/staff-directory/santiago-staff",
    "https://www.phoenixcharteracademy.org/who-we-are/our-staff/",
    "https://www.ncavts.org/domain/21",
    "https://www.monroevilleschools.org/StaffDirectory.aspx",
    "http://lavidaschool.org/directory/",
    "https://www.konoctiusd.org/District/Staff/",
    "https://www.homercenter.org/apps/staff/",
    "https://www.glenncoe.org/Staff-Directory/index.html",
    "https://www.frsd.info/departments/administrative-staff",
    "https://www.evergreenps.org/Equity/department-staff",
    "https://www.dallask12.org/1/Content2/586",
    "https://www.csd313.org/apps/pages/index.jsp?uREC_ID=1575430&type=d&pREC_ID=2104451",
    "https://www.cfbisd.edu/about-us/district-leadership",
    "https://www.ccsdk12.org/apps/staff/",
    "https://berkscareer.com/meet-our-administrators/",
    "https://www.amethodschools.org/apps/staff/",
    "https://www.albanyschools.org/about/district-administration",
    "https://www.aadusd.k12.ca.us/domain/20",
    "https://www.luskinacademy.org/apps/staff/",
]

dynamic_urls = [
    "https://www.wcs.k12.va.us/staff",
    "https://www.upsd83.org/about_upsd/staff_directory",
    "https://www.sw.wednet.edu/connect/staff-directory",
    "https://www.svpanthers.org/domain/667",
    "https://www.petalumacityschools.org/domain/74",
    "https://www.mossyrockschools.org/district/staff_directory",
    "https://www.iu08.org/staff",
    "https://www.edencisd.net/staff",
    "https://www.crschools.org/parents/staff_directory",
    "https://www.clymercsd.org/staff",
    "https://www.ccusd93.org/domain/142",
    "https://www.broaddusisd.com/staff",
    "https://www.asfa.k12.al.us/domain/34",
    "https://www.ahisd.net/cms/one.aspx?pageId=899355",
    "https://www.agsd.us/staff",
    "https://osd.wednet.edu/our_district/staff_directory",
    "https://libertasacademy.org/our-team/",
    "https://www.tekoasd.org/staff",
    "https://www.vacavilleusd.org/departments/directory_by_department",
    "https://www.westcler.org/our-team/directory/index",
    "https://www.phillyscholars.org/apps/pages/index.jsp?uREC_ID=1046939&type=d&pREC_ID=staff",
    "https://www.ofcs.net/district_staff.aspx?action=search&location=0&department=0",
    "https://www.laurenscs.org/StaffDirectory.aspx",
    "https://www.foxboroughrcs.org/apps/pages/StaffDirectory",
    "https://www.cwctc.org/index.php/staff-directory-2/",
    "https://www.csd49.org/directory",
    "https://www.comfort.txed.net/354240_2",
    "https://www.cimarron.k12.ok.us/vnews/display.v/SEC/Staff",
    "https://brooklyncharter.org/about/leadership-and-staff/",
    "https://bridgehampton.k12.ny.us/staff/Default.aspx?school=276",
    "https://bridge-rayn.org/en-US/staff",
    "https://www.bouseschool.org/Administration-and-Staff",
    "https://www.aobihs.com/administration/",
    "https://www.allenbowden.org/apps/staff/",
    "http://ctec.fcoe.org/our-team",
    "https://www.collinsfamilyjaguars.org/apps/staff/",
    "https://bellinghamschools.org/contact/",
]

"""# Data Collection and Cleaning Function"""

import requests
from bs4 import BeautifulSoup

def collect_data(url):
    """
    Collects data from a given URL by making an HTTP GET request and parsing the HTML content.

    Args:
        url (str): The URL of the webpage to scrape.

    Returns:
        str: The HTML content of the webpage as a string.
    """

    # Send an HTTP GET request to the specified URL
    response = requests.get(url)

    # Create a BeautifulSoup object to parse the HTML content
    soup = BeautifulSoup(response.content, 'html.parser')

    # Remove all 'style' tags from the HTML content
    for style in soup.find_all('style'):
        style.decompose()

    # Remove all 'script' tags from the HTML content
    for script in soup.find_all('script'):
        script.decompose()

    # Convert the modified HTML content back to a string
    data = str(soup)

    # Return the modified HTML content
    return data

"""# Data Chunking Function"""

from langchain.text_splitter import RecursiveCharacterTextSplitter

def create_data_chunks(data):
    """
    Creates chunks of data by splitting the input text using RecursiveCharacterTextSplitter.

    Args:
        data (str): The input text to be split into chunks.

    Returns:
        list: A list of data chunks.

    """

    # Initialize RecursiveCharacterTextSplitter with desired chunk size and overlap
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=50
    )

    # Split the input data into chunks using the text splitter
    data_chunks = text_splitter.create_documents([data])

    data_chunks_contents = []
    for chunk in data_chunks:
        # Extract the content of each chunk
        data_chunks_contents.append(chunk.page_content)

    return data_chunks_contents

"""# Data Embeddings Function"""

import tiktoken
from langchain.embeddings import OpenAIEmbeddings

def create_data_embeddings(data):
  """
  Creates embeddings for a list of documents using OpenAI's text-embedding model.

  Args:
    data (list): A list of documents to be embedded.

  Returns:
    data_embeddings (list): A list of embeddings corresponding to the input documents.
  """

  # Instantiate OpenAIEmbeddings with the desired model and API key
  embeddings = OpenAIEmbeddings(
    model='text-embedding-ada-002',
    openai_api_key=OPENAI_API_KEY
  )

  # Embed the input documents using OpenAI's model
  data_embeddings = embeddings.embed_documents(data)

  return data_embeddings

"""# Top Relevant Chunks Function"""

from sklearn.metrics.pairwise import cosine_similarity

def find_top_chunks(query_embeddings, chunks_embeddings, data_chunks):
    """
    Finds the top-k most similar chunks to a given query using cosine similarity.

    Args:
        query_embeddings (array-like): Embeddings of the query.
        chunks_embeddings (array-like): Embeddings of all the data chunks.
        data_chunks (list): List of data chunks.

    Returns:
        list: Top-k most similar data chunks.

    """

    # Calculate cosine similarity between query and all chunks
    similarities = cosine_similarity(query_embeddings, chunks_embeddings)

    # Get the indices of the most similar documents
    top_k = 10
    top_indices = similarities.argsort()[0][-top_k:][::-1]

    # Sort the indices in ascending order
    top_indices.sort()

    # Retrieve the most similar documents
    top_chunks = [data_chunks[i] for i in top_indices]

    return top_chunks

"""# Prompt Generation Function"""

def generate_prompt(top_chunks):
  """
  Generates a prompt for extracting staff contact data from web contents.

  Args:
    top_chunks (list): A list of strings representing the top chunks of web contents.

  Returns:
    list: A list of prompt objects containing system and user messages.
  """
  # Merge the top chunks into a single string
  merged_chunks = ""

  for i, chunk in enumerate(top_chunks, 1):
    merged_chunks += ("Chunk " + str(i) + ":\n")
    merged_chunks += chunk
    merged_chunks += "\n\n"

  # Define the staff format for the prompt
  staff_format = [
      {
        "name": "",
        "job position": "",
        "email": "",
        "phone": ""
      }
    ]

  # Create the prompt with system and user messages
  prompt = [
      {"role": "system", "content": "You are a helpful assistant who can extract staff contacts data from any web contents and return the output in the valid JSON format with proper indentation - \n" + str(staff_format)},
      {"role": "user", "content": "Here are the web contents - \n" + merged_chunks}
  ]

  return prompt

"""# LLM Output Function"""

import openai
import json

def collect_llm_output(prompt):
  """
  Collects the output from the language model based on the given prompt.

  Args:
    prompt (list): A list of message objects representing the conversation prompt.

  Returns:
    str: The output generated by the language model.

  """

  # Create a chat completion using the OpenAI API
  completion = openai.ChatCompletion.create(
    model="gpt-3.5-turbo",
    messages=prompt,
    api_key=OPENAI_API_KEY,
    temperature=0
  )

  # Get the content of the first message in the completion as the output
  llm_output = completion.choices[0].message.content

  return llm_output

"""# JSON Format Function"""

import json

def create_json_format(llm_output):
    """
    Converts LLM output to a formatted JSON string.

    Args:
        llm_output (str): The LLM output string.

    Returns:
        str: Formatted JSON string.

    Raises:
        ValueError: If the provided llm_output does not contain valid JSON data.
    """

    # Trim the llm_output to extract the JSON data
    left_trim = "[" + llm_output.split("[", 1)[1]
    right_trim = left_trim.split("]", 1)[0] + "]"
    replaced_trim = right_trim.replace("'", "\"")

    try:
        # Parse the trimmed output as JSON
        json_data = json.loads(replaced_trim)
    except ValueError as e:
        raise ValueError("Invalid JSON data in llm_output") from e
        json_data = {}

    # Format the JSON data with indentation
    formatted_json = json.dumps(json_data, indent=4)

    return formatted_json

"""# Main Pipeline Function"""

def run_pipeline(url):
  """
    Runs the data processing pipeline to collect data from a given URL, create data chunks,
    generate data embeddings, find relevant chunks, generate a prompt, collect the output
    from a language model, and create JSON format.

    Args:
        url (str): The URL to collect data from.

    Returns:
        dict: The processed data in JSON format.
  """

  # Collect Data
  print("Collecting Data...\n")
  data = collect_data(url)

  # Create Data Chunks
  print("Creating Data Chunks...\n")
  data_chunks = create_data_chunks(data)

  # Create Data Embeddings
  print("Creating Data Embeddings...\n")
  chunks_embeddings = create_data_embeddings(data_chunks)

  # Create Query Embeddings
  print("Creating Query Embeddings...\n")
  query = "Find documents with these info: name, email, phone, job position"
  query_embeddings = create_data_embeddings([query])

  # Find Top Chunks
  print("Finding Top Chunks...\n")
  top_chunks = find_top_chunks(query_embeddings, chunks_embeddings, data_chunks)

  # Generate Prompt
  print("Generating Prompt...\n")
  prompt = generate_prompt(top_chunks)

  # Collect LLM Output
  print("Collecting LLM Output...\n")
  llm_output = collect_llm_output(prompt)
  print(llm_output)

  # Create JSON Format
  print("Creating JSON Format...\n")
  jsonData = create_json_format(llm_output)

  return jsonData

"""# Run Code"""

url = static_urls[6]
print("URL: " + url + '\n')

result = run_pipeline(url)
print(result)