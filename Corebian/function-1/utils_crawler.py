# Copyright 2023 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import re
import datetime
import requests
from io import StringIO, BytesIO
from html.parser import HTMLParser
from fuzzysearch import find_near_matches
from bs4 import BeautifulSoup
from google import auth
from google.cloud import storage
from google.auth.transport import requests as req
from pypdf import PdfReader
from utils_config import ENGINE_TYPE

client = storage.Client()
headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 6.1; WOW64; rv:20.0) Gecko/20100101 Firefox/20.0'}

def get_file_metadata(url: str) -> dict:
    credentials, project_id = auth.default()
    if credentials.token is None:
        credentials.refresh(req.Request())
    client = storage.Client()
    bucket_name, file_name = url[5:].split('/', 1)
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(file_name)
    try:
        link = blob.generate_signed_url(
            version="v4",
            # This URL is valid for 15 minutes
            service_account_email=credentials.service_account_email,
            access_token=credentials.token,
            expiration=datetime.timedelta(minutes=15),
            # Allow GET requests using this URL.
            method="GET",
        )
    except Exception as e:
        print(e)
        link = f"https://storage.googleapis.com/{bucket_name}/{file_name}"
    return {"bucket": bucket_name, "file_name": file_name, "link": link}

def parse_pdf(contents: BytesIO, page_nuber: int) -> str:
    pdf = PdfReader(contents)
    text = ""
    for page in pdf.pages[page_nuber-1:page_nuber+1]:
        text += page.extract_text()
    return text

def parse_html(contents: str) -> str:
    soup = BeautifulSoup(contents, 'html.parser')
    return ' '.join(soup.stripped_strings)

def extract_text_from_website(url: str) -> str:
    session = requests.Session()
    try:
        response = session.get(url, headers=headers)
        text = parse_html(response.text)
        return text
    except Exception as e:
        print(e)
        return ' '

def extract_text_from_docs(url:str, page_number: int) -> dict:

    metadata = get_file_metadata(url)
    file_name = metadata.get('file_name')
    bucket = client.bucket(metadata.get('bucket'))
    blob = bucket.blob(file_name)
    if file_name.endswith('.txt'):
        text = blob.download_as_string()
    elif file_name.endswith('.html'):
        contents = blob.download_as_string()
        text = parse_html(contents)
    elif file_name.endswith('.pdf'):
        contents = BytesIO(blob.download_as_bytes())
        text = parse_pdf(contents, page_number)

    return text


def generate_reference(result, n_results):
    print("generate start")
    link = result.document.derived_struct_data.get("link")
    snippets = result.document.derived_struct_data.get("snippets")

    if ENGINE_TYPE == "unstructured":
        metadata = get_file_metadata(link)
        title = metadata.get("file_name")
        link = metadata.get("link")
        extractive_answers = result.document.derived_struct_data.get("extractive_answers")
        extractive_segments = result.document.derived_struct_data.get("extractive_segments")
        if extractive_answers:
            long_snippet = " ".join([answer["content"] for answer in extractive_answers])
        elif extractive_segments:
            long_snippet = extractive_segments[0]["content"]
        else:
            return {}
        return {"link": link, "long_snippet": long_snippet, "title": title}
                
    elif ENGINE_TYPE == "website":
        response_text = extract_text_from_website(link)
        title = result.document.derived_struct_data.get("title")

    full_snippet = snippets[0]["snippet"]
    snippet = max(full_snippet.split("..."), key=len)
    snippet = snippet.replace("\n", " ").replace("\r", " ").replace("\t", " ")
    snippet_position = list(find_near_matches(
        snippet[:-5].lower(), response_text.lower(), max_l_dist=10))

    if snippet_position:
        snippet_position = snippet_position[0].start
    else:
        snippet_position = 0
    snippet_size = 15000//(n_results*2)
    if len(response_text) <= snippet_size:
        long_snippet = response_text
    else:
        snippet_beginning = snippet_position - snippet_size//2
        snippet_end = snippet_position + snippet_size//2
        if snippet_beginning < 0:
            snippet_end -= snippet_beginning
            snippet_beginning = 0
        elif snippet_end > len(response_text):
            snippet_beginning -= snippet_end - len(response_text)
            snippet_end = len(response_text)
        long_snippet = response_text[snippet_beginning:snippet_end]
    return {
        "link": link,
        "long_snippet": long_snippet,
        "snippet": full_snippet,
        "title": title}