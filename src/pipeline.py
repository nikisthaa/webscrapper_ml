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
    css_selectors = ['em.fa.fa-arrow-right', 'li.next a', 'a.fsNextPageLink', '.paginate_button.next']
    link_texts = ['Next >>']
    x_paths = ['//a[span="Next page"]', '//a[@title="Next"]']

    # Set up the Selenium WebDriver in headless mode
    chrome_options = Options()
    chrome_options.add_argument("--headless")
    driver = webdriver.Chrome(options=chrome_options)

    page_content = []
    page_no = 1
    previous_content = None
    next_button = None
    driver.get(url)
    while True:
        print(f"Scrapping content from: {url} : page_no: {page_no}")
        try:
            # Use the WebDriverWait
            wait_for_any_element(driver, 2, css_selectors, link_texts, x_paths)
        except TimeoutException:
            print("Timed out waiting for page to load")
            break

        soup = BeautifulSoup(driver.page_source, "html.parser")
        # Remove all 'style' and 'script' tags from the HTML content
        for tag in soup.find_all(['style', 'script']):
            tag.decompose()

        # Convert the modified HTML content back to a string
        current_content = str(soup)

        if current_content == previous_content:
            print("Content didn't change, probably no more pages left, exiting loop.")
            break
        page_content.append(current_content)

        previous_content = current_content

        page_no += 1
        next_button = find_next_button(driver, css_selectors, link_texts, x_paths, page_content)
        if next_button is None:
            break
        try:
            next_button.click()
        except (NoSuchElementException, ElementClickInterceptedException, ElementNotInteractableException):
            break

    driver.quit()

    return page_content


def wait_for_any_element(driver, timeout, css_selectors, link_texts, x_paths):
    wait = WebDriverWait(driver, timeout)
    try:
        for selector in css_selectors:
            wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, selector)))
            return True
    except:
        pass

    try:
        for text in link_texts:
            wait.until(EC.presence_of_element_located((By.LINK_TEXT, text)))
            return True
    except:
        pass

    try:
        for x_path in x_paths:
            wait.until(EC.presence_of_element_located((By.XPATH, x_path)))
            return True
    except:
        pass

    return False



def find_next_button(driver, css_selectors, link_texts, x_paths, page_content):
    next_button = None
    # Try to find the "next" button and click it if css selectors, link texts, xpaths
    for find_method, locator_list in zip(
        [By.CSS_SELECTOR, By.LINK_TEXT, By.XPATH], 
        [css_selectors, link_texts, x_paths]
    ):
        for locator in locator_list:
            try:
                next_button = driver.find_element(find_method, locator)
                if next_button:
                    break
            except NoSuchElementException:
                continue
        if next_button:
            break

    # Try to find the "number" button and click it if only numbers/paginations
    if next_button is None:
        pagination_list = None
        try:
            pagination_list = driver.find_elements(By.CSS_SELECTOR, "ul.ui-pagination-list li")
            if pagination_list: 
                # Click each page link
                page_no = 2
                for page in pagination_list[1:]:
                    try:
                        # Click the page link
                        page_link = page.find_element(By.TAG_NAME, "a")
                        page_link.click()

                        # Wait for the page to load
                        WebDriverWait(driver, 3).until(EC.presence_of_all_elements_located((By.TAG_NAME, 'a')))

                        
                        # Get the HTML of the page
                        html_content = driver.page_source
                        page_content.append(html_content)
                        print(f"Scraping page: {page_no}")
                        page_no += 1
                    except NoSuchElementException:
                        print("NoSuchElementException occured")
                        continue
                    except StaleElementReferenceException:
                        print("StaleElementReferenceException occured")
                        continue
        except NoSuchElementException:
            print('Pagination list not found.')
            # If the pagination list can't be found, we've reached the last page
            return None
    return next_button


def handle_pagination(url, pagination_urls, page_no, page_content, prev_content):
    # If the website has standard pagination link format
    previous_content = prev_content
    last_modified = None
    for pagination_url in pagination_urls:
        print(pagination_url)
        page = page_no
        while True:
            print(f"Scraping next page: {page}")
            try:
                headers = {'If-Modified-Since': last_modified} if last_modified else {}
                response = requests.get(url+pagination_url.format(page), headers=headers)
                if response.status_code != 200:
                    print("Probably no more pages left, exiting loop.")
                    break

                last_modified = response.headers.get('Last-Modified')
                
                soup = BeautifulSoup(response.content, "html.parser")
    
                # Remove all 'style' and 'script' tags from the HTML content
                for tag in soup.find_all(['style', 'script']):
                    tag.decompose()
                
                current_content = str(soup)
                if current_content == previous_content:
                    print("Content didn't change, probably no more pages left, exiting loop.")
                    break

                previous_content = current_content
                page += 1
                # Convert the modified HTML content back to a string
                page_content.append(str(soup))
            except Exception as e:
                print(e)


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
    # if not is_valid_json(replaced_trim):
    #     print("Inside another api calls")
    #     # Create the prompt with system and user messages
    #     next_prompt = [
    #         {"role": "system", "content": "You are a helpful assistant who can extract staff contacts data from any web contents and return the output in the valid JSON format with proper indentation."},
    #         {"role": "user", "content": replaced_trim + "\n Just return a valid JSON data without duplicates. I am not asking for codes."}
    #     ]

    #     # Create a chat completion using the OpenAI API
    #     completion1 = openai.ChatCompletion.create(
    #         model="gpt-3.5-turbo",
    #         messages=next_prompt,
    #         api_key=os.environ.get("OPENAI_API_KEY", ""),
    #         temperature=0
    #     )

    #     # Get the content of the first message in the completion as the output
    #     llm_output = completion1.choices[0].message.content
    #     return llm_output
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
        try:
            reformatted_json = reformat_llm_output(llm_output)
            json_data = [json.loads(obj) for obj in reformatted_json]
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


def reformat_llm_output(llm_output):
    count = 0
    objects = []
    temp_object = ""

    for char in llm_output:
        if char == '{':
            count += 1
        elif char == '}':
            count -= 1
        
        if count > 0:
            temp_object += char
        elif count == 0 and temp_object != "":
            temp_object += char
            objects.append(temp_object.strip())
            temp_object = ""
    return objects

    
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
                    "staff_directory": row[3]
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

# http://www.syracusecityschools.com/getstaff.cfm?building=Public
def main(src_file, dest_file):
    start_time = time.perf_counter()
    datas = read_file(src_file)
    counter = 551
    for data in datas[551:]:
        if counter>950:
            break
        try:
            html_contents =  fetch_html(data["staff_directory"])
            total_contacts = []
            for html_content in html_contents:
                try:
                    contacts = run_pipeline(html_content)
                    school_info = {"school": data["school_name"], "state": data["state"]}
                    if len(contacts)>0:
                        if type(contacts) is dict:
                            for key in contacts.keys():
                                updated_json_data = [{**item, **school_info} for item in contacts[key]]
                        
                        else:
                            updated_json_data = [{**item, **school_info} for item in contacts]
                        total_contacts.extend(updated_json_data)
                except Exception as e:
                    print("An exception occurred:", type(e).__name__, "–", e) 
            if len(total_contacts)>0:
                write_to_csv(dest_file, total_contacts)
        except Exception as e:
            print(f"An error occurred while scraping: {data['staff_directory']}:", e)
        counter = counter + 1
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


'''
Not scraped:
https://www.pceagles.org/about/teachers-staff/
http://www.pobschools.org/staffdirectory
started from 33
Took 39881.875605 sec to scrape 200
Took 32243.252992 sec to scrape 200
Took 6601.737581 sec to scrape 100

# Website with lots of pagingation:
https://maldenps.org/staff-directory/

# Rate time limit reached here: https://marshallhs.fcps.edu/about/staff-directory ; couldnot extract any below this
'''