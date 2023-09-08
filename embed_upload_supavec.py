

import re
import nltk
from typing import List
from config import settings
from dotenv import load_dotenv
from supabase import create_client, Client

from langchain.vectorstores import SupabaseVectorStore
from langchain.embeddings.openai import OpenAIEmbeddings
 
load_dotenv()

# Constants
OPENAI_API_KEY = settings.openai_api_key
EMBEDDING_MODEL = "text-embedding-ada-002"
CHUNK_SIZE = 2048

SUPABASE_URL = settings.supabase_url
SUPABASE_KEY = settings.supabase_key
TABLE_NAME = "hope_communities"

# Instantiate clients
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
embedding = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
model_name = EMBEDDING_MODEL

# Function to generate embeddings
def embedding_function(texts):
    return embedding.embed_documents(texts)

# Define functions to split text into sentences and group sentences into chunks
def split_into_sentences(text):
    tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
    return tokenizer.tokenize(text)

def group_sentences_into_chunks(sentences, chunk_length=2048):
    chunks = []
    chunk = []
    chunk_length_current = 0
    for sentence in sentences:
        sentence_length = len(sentence)
        if chunk_length_current + sentence_length <= chunk_length:
            chunk.append(sentence)
            chunk_length_current += sentence_length
        else:
            chunks.append(' '.join(chunk))
            chunk = [sentence]
            chunk_length_current = sentence_length
    if chunk:
        chunks.append(' '.join(chunk))
    return chunks
 
# Define a Document class with 'page_content' and 'metadata' attributes
class Document:
    def __init__(self, page_content, metadata):
        self.page_content = page_content
        self.metadata = metadata

# Read and parse data.txt
with open('data.txt', 'r') as file:
    text = file.read()
sections = re.split(r'== (.+?) ==', text)[1:]
sections_dict = {sections[i]: sections[i+1] for i in range(0, len(sections), 2)}

# Prepare a list of documents
documents = []

# Process each section and add to documents list
for section_title, section_text in sections_dict.items():
    if section_title == 'Links':
        # Split the 'Links' section into individual links
        links = re.split(r'Label:', section_text)[1:]
        for link in links:
            link_parts = re.split(r'\n', link.strip())
            label = link_parts[0].strip()
            main_link = link_parts[1].replace('Main Link: ', '').strip()
            sub_links = [sub_link.strip() for sub_link in link_parts[3:]]  # Skip the 'Sub-links:' line
            # Add label and links to documents list
            documents.append(Document(label, {'type': 'label'}))
            documents.append(Document(main_link, {'type': 'main_link'}))
            for sub_link in sub_links:
                documents.append(Document(sub_link, {'type': 'sub_link'}))
    else:
        sentences = split_into_sentences(section_text)
        section_chunks = group_sentences_into_chunks(sentences)
        # Add each chunk to documents list
        for chunk in section_chunks:
            documents.append(Document(chunk, {'type': 'section', 'title': section_title}))

def init_vector_store(
  documents: List[Document], 
  embeddings: OpenAIEmbeddings,
  table_name: str,
  client: Client,
) -> SupabaseVectorStore:
  
  return SupabaseVectorStore.from_documents(
    documents, embeddings, table_name=table_name, client=client, content_field="page_content", metadata_field="metadata"
  )

vector_store = init_vector_store(
  documents, embedding, TABLE_NAME, supabase
)
