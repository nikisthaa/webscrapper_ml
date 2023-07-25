import os
import csv
import json
import time
import openai
import requests
import argparse
from bs4 import BeautifulSoup
from dotenv import load_dotenv

from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter

from sklearn.metrics.pairwise import cosine_similarity

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import NoSuchElementException, ElementClickInterceptedException, StaleElementReferenceException, ElementNotInteractableException, TimeoutException


load_dotenv(".env.local")


def is_email_protected(soup):
    # Search for email protection indicators in the HTML

    # Option 1: Check for Cloudflare email protection message
    cloudflare_message = soup.find("div", string="Email Protection")
    if cloudflare_message:
        return True

    # Option 2: Check for email obfuscation techniques
    email_elements = soup.select('a[href^="mailto:"]')
    if email_elements:
        return True

    # If no email protection indicators are found, assume email is not protected
    return False


def fetch_html(url):
    """
    Collects data from a given URL by making an HTTP GET request and parsing the HTML content.

    Args:
        url (str): The URL of the webpage to scrape.
        method (str): The method to scrape the data. (Allowed: bs4 or selenium)

    Returns:
        str: The HTML content of the webpage as a string.
    """
    page_content = []
    css_selectors = ['em.fa.fa-arrow-right', 'li.next a', 'a.fsNextPageLink', '.paginate_button.next']
    link_texts = ['Next >>']
    x_paths = ['//a[span="Next page"]', '//a[@title="Next"]']
    page=1
    pagination_urls = ["/page_no={0}", "?page={0}"]
    
    # Set up the Selenium WebDriver in headless mode
    chrome_options = Options()
    chrome_options.add_argument("--headless")
    driver = webdriver.Chrome(options=chrome_options)  # Replace with the appropriate driver for your browser
        
    driver.get(url)

    # Wait 3 seconds for html content to load
    time.sleep(2)

    previous_content = None
    while True:
        print("Scrapping content from: " + url + " : pageno: " + str(page))

        # Retrieve the protected HTML content
        protected_html = driver.page_source

        # Pass the protected HTML content to BeautifulSoup for parsing
        soup = BeautifulSoup(protected_html, "html.parser")
        # Remove all 'style' tags from the HTML content
        for style in soup.find_all('style'):
            style.decompose()

        # Remove all 'script' tags from the HTML content
        for script in soup.find_all('script'):
            script.decompose()

        # Convert the modified HTML content back to a string
        current_content = str(soup)
        if current_content == previous_content:
            print('Content is the same as before.')
        else:
            page_content.append(current_content)

        page=page+1

        # Try to find the "next" button and click it if css selectors
        next_button = None
        for selector in css_selectors:
            try:
                next_button = driver.find_element(By.CSS_SELECTOR, selector)
                if next_button:
                    break
            except NoSuchElementException:
                continue

        # Try to find the "next" button and click it if link texts
        if next_button is None:
            for text in link_texts:
                try:
                    next_button = driver.find_element(By.LINK_TEXT, text)
                    if next_button:
                        break
                except NoSuchElementException:
                    continue

        # Try to find the "next" button and click it if xpaths
        if next_button is None:
            for x_path in x_paths:
                try:
                    next_button = driver.find_element(By.XPATH, x_path)
                    if next_button:
                        break
                except NoSuchElementException:
                    continue

        # Try to find the "number" button and click it if only numbers/paginations
        if next_button is None:
            pagination_list = None
            try:
                pagination_list = driver.find_elements(By.CSS_SELECTOR, "ul.ui-pagination-list li.ui-page-number.ui-small-page")
                if pagination_list: 
                    # Click each page link
                    for page in pagination_list:
                        # Click the page link
                        page_link = page.find_element(By.TAG_NAME, "a")
                        page_link.click()

                        # Wait for the page to load
                        time.sleep(2)
                        # Get the HTML of the page
                        html_content = driver.page_source
                        page_content.append(html_content)
            except (NoSuchElementException, StaleElementReferenceException):
                # If the pagination list can't be found, we've reached the last page
                break

        # If the website has standard pagination link format    
        if next_button is None:
            page = 2
            for pagination_url in pagination_urls:
                try:
                    response = requests.get(url+pagination_url.format(page))
                    
                    if response.status_code != 200:
                        # If status code is not 200, the page was not successfully received
                        continue

                    driver.get(url)
                    # Retrieve the protected HTML content
                    protected_html = driver.page_source
                    # Pass the protected HTML content to BeautifulSoup for parsing
                    soup = BeautifulSoup(protected_html, "html.parser")
                    # Remove all 'style' tags from the HTML content
                    for style in soup.find_all('style'):
                        style.decompose()

                    # Remove all 'script' tags from the HTML content
                    for script in soup.find_all('script'):
                        script.decompose()

                    # Convert the modified HTML content back to a string
                    data = str(soup)
                    
                    page_content.append(data)
                    page = page + 1
                except Exception as e:
                    print(e)
        try:
            # Try to find the "next" button and click it
            if next_button is None: 
                break         
            else:
                next_button.click()
                time.sleep(2)
        except (NoSuchElementException, ElementClickInterceptedException, ElementNotInteractableException):
            # If the "next" button can't be found, we've reached the last page
            break
    driver.quit()
    # Return the modified HTML content
    return page_content


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
        openai_api_key=os.environ.get("OPENAI_API_KEY", "")
    )

    # Embed the input documents using OpenAI's model
    data_embeddings = embeddings.embed_documents(data)

    return data_embeddings


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
        {"role": "system", "content": "You are a helpful assistant who can extract staff contacts (especially teacher, teaching assistant, ) data from any web contents and return the output in the valid JSON format with proper indentation - \n" + str(staff_format)},
        {"role": "user", "content": "Here are the web contents - \n" + merged_chunks + "Please remove any duplicates."}
    ]

    return prompt


def is_valid_json(data):
    try:
        json.loads(data)
        return True
    except ValueError:
        return False
    

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
        api_key=os.environ.get("OPENAI_API_KEY", ""),
        temperature=0
    )

    # Get the content of the first message in the completion as the output
    llm_output = completion.choices[0].message.content

    # Trim the llm_output to extract the JSON data
    left_trim = "[" + llm_output.split("[", 1)[1]
    right_trim = left_trim.split("]", 1)[0] + "]"
    replaced_trim = right_trim.replace("'", "\'")
    if not is_valid_json(replaced_trim):
        print("Inside another api calls")
        # Create the prompt with system and user messages
        next_prompt = [
            {"role": "system", "content": "You are a helpful assistant who can extract staff contacts data from any web contents and return the output in the valid JSON format with proper indentation."},
            {"role": "user", "content": replaced_trim + "\n Just return a valid JSON data without duplicates. I am not asking for codes."}
        ]

        # Create a chat completion using the OpenAI API
        completion1 = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=next_prompt,
            api_key=os.environ.get("OPENAI_API_KEY", ""),
            temperature=0
        )

        # Get the content of the first message in the completion as the output
        llm_output = completion1.choices[0].message.content
        return llm_output
    return replaced_trim


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
    if llm_output=='[]':
        return []
    try:
        # Parse the trimmed output as JSON
        json_data = json.loads(llm_output)
        return json_data
    except ValueError as e:
        raise ValueError("Invalid JSON data in llm_output")


def run_pipeline(html):
    """
        Runs the data processing pipeline to collect data from a given URL, create data chunks,
        generate data embeddings, find relevant chunks, generate a prompt, collect the output
        from a language model, and create JSON format.

        Args:
            hrml (str): The HTML content of webpage.

        Returns:
            dict: The processed data in JSON format.
    """
    # Create Data Chunks
    print("Creating Data Chunks...\n")
    data_chunks = create_data_chunks(html)

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
    
    # Create JSON Format
    print("Creating JSON Format...\n")
    jsonData = create_json_format(llm_output)
    return jsonData


def read_file(path:str) -> list([dict]):
    school_staff_directories = []
    # Open the CSV file
    with open(path, 'r') as file:
        # Create a CSV reader
        csv_reader = csv.reader(file)

        # Skip the header row
        next(csv_reader)

        # Iterate over each row in the CSV file
        for row in csv_reader:
            # Access data in each row
            school_staff_directories.append(
                {
                    "school_name": row[0],
                    "state": row[1],
                    "staff_directory": row[2]
                }
            )
        return school_staff_directories


def write_to_csv(filename: str, data: list[dict]):
    file_exists = os.path.isfile(filename)
    keys = data[0].keys()

    with open(filename, 'a', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=keys)

        if not file_exists:
            writer.writeheader()

        writer.writerows(data)

    print(f"CSV file '{filename}' created/updated successfully.")


def main(src_file, dest_file):
    start_time = time.perf_counter()
    datas = read_file(src_file)
    for data in datas:
        try:
            html_contents =  fetch_html(data["staff_directory"])
            total_contacts = []
            for html_content in html_contents:
                try:
                    contacts = run_pipeline(html_content)
                    school_info = {"school": data["school_name"], "state": data["state"]}
                    if len(contacts)>0:
                        if "staff_contacts" in contacts:
                            updated_json_data = [{**item, **school_info} for item in contacts['staff_contacts']]
                        else:
                            updated_json_data = [{**item, **school_info} for item in contacts]
                    total_contacts.extend(updated_json_data)
                except Exception as e:
                    print("An exception occurred:", type(e).__name__, "–", e) 
            if len(total_contacts)>0:
                write_to_csv(dest_file, total_contacts)
        except ValueError:
            print(f"Unexpected value for pagination: {data['pagination']}")
    # Measure the end time
    end_time = time.perf_counter()
    elapsed_time = end_time - start_time

    print(f"The function took {elapsed_time:.6f} seconds to run.")


def test_main(src_file=None, dest_file=None):
    # start_time = time.perf_counter()

    # datas = read_file(src_file)
    # for data in datas:
    #     try:
    #         html_contents =  fetch_html(data["staff_directory"])
    #         total_contacts = []
    #         for html_content in html_contents:
    #             try:
    #                 contacts = run_pipeline(html_content)
    #                 school_info = {"school": data["school_name"], "state": data["state"]}
    #                 if len(contacts)>0:
    #                     updated_json_data = [{**item, **school_info} for item in contacts]
    #                     total_contacts.extend(updated_json_data)
    #             except Exception as e:
    #                 print("An exception occurred:", type(e).__name__, "–", e) 
    #         if len(total_contacts)>0:
    #             write_to_csv(dest_file, total_contacts)
    #     except ValueError:
    #         print(f"Unexpected value for pagination: {data['pagination']}")

    # # Measure the end time
    # end_time = time.perf_counter()
    # elapsed_time = end_time - start_time

    # print(f"The function took {elapsed_time:.6f} seconds to run.")
    start_time = time.perf_counter()
    staff_directory = "https://bmhs.eagleschools.net/about-us/staff-directory"

    html_contents =  fetch_html(staff_directory)
    total_contacts = []
    for html_content in html_contents:
        try:
            print("success")
            # contacts = run_pipeline(html_content)
            # school_info = {"school": "grtg", "state": "stEFRWA"}
            # if len(contacts)>0:
            #     if "staff_contacts" in contacts:
            #         updated_json_data = [{**item, **school_info} for item in contacts['staff_contacts']]
            #     else:
            #         updated_json_data = [{**item, **school_info} for item in contacts]
            #     total_contacts.extend(updated_json_data)
        except Exception as e:
            print("An exception occurred:", type(e).__name__, "–", e) 
        
    if len(total_contacts)>0:
        write_to_csv("test.csv", total_contacts)

    # Measure the end time
    end_time = time.perf_counter()

    elapsed_time = end_time - start_time

    print(f"The function took {elapsed_time:.6f} seconds to run.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process a file.")
    parser.add_argument('src_file', type=str, help="The path of the file to process")
    parser.add_argument('dest_file', type=str, help="The path of the file to process")

    args = parser.parse_args()

    main(args.src_file, args.dest_file)
