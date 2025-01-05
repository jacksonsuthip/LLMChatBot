import hashlib
import requests
from app.config import COLLECTION_NAME, PERSIST_DIRECTORY_DB
from app.models.embedding_models import EmbeddingModels
from app.models.llm_models import LlmModels
from bs4 import BeautifulSoup
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_community.document_loaders import WebBaseLoader
from uuid import uuid4
from urllib.parse import urljoin, urlparse

embeddingModels = EmbeddingModels()
nomicEmbeddings = embeddingModels.nomic_embeddings_ollama

chunk_size = 512
chunk_overlap = 50

# Chroma vector store
vector_store = Chroma(
    collection_name=COLLECTION_NAME,
    embedding_function=nomicEmbeddings,
    persist_directory=PERSIST_DIRECTORY_DB,
)

def get_sub_urls(base_url):
    """Scrape all sub-URLs from a website."""
    try:
        response = requests.get(base_url)
        # print("response --- " + response.text) 
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')

        # Extract all <a> tags with href attributes
        links = [a['href'] for a in soup.find_all('a', href=True)]
        print(f"Extracted links: {links}")
    
        # Normalize links: filter and convert to absolute URLs
        sub_urls = []
        for link in links:
            image_extensions = ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.svg', '.webp', '.tiff']
            if any(link.lower().endswith(ext) for ext in image_extensions):
                print(f"Ignored image link: {link}")
                continue  # Skip image links

            if "@" in link:
                print(f"Ignored email: {link}")
                continue 
            
            # Handle anchor links (e.g., '#section1')
            full_url = ""
            if link.startswith('#'):
                print(f"Ignored anchor link: {link}")
                full_url = base_url + link
                # continue  # Skip anchor links
            else:
                # Handle relative URLs and path traversal (e.g., ../about)
                print(f"urljoin link: {link}")
                full_url = urljoin(base_url, link)

            # Check if the link is an absolute link within the domain
            if link.startswith(base_url):  # Absolute link within the domain
                print(f"Absolute URL: {link}")
                sub_urls.append(link)
            else:  # Handle relative URLs and links with `../`
                print(f"Resolved relative URL: {full_url}")
                sub_urls.append(full_url)

        # Remove duplicates
        sub_urls = list(set(sub_urls))
        return sub_urls

    except Exception as e:
        print(f"Error fetching sub-URLs: {e}")
        return []

def process_url(url):
    try:
        isNewUrl = True
        first_unique_id = ""
        new_chunks = []
        existing_chunks = []

        loader = WebBaseLoader(url)
        documents = loader.load()

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size, 
            chunk_overlap=chunk_overlap, 
            separators=["\n", " ", ""]
        )

        split_documents = text_splitter.split_documents(documents)

        chunks_with_ids = calculate_chunk_ids(split_documents)

        if chunks_with_ids:
            first_unique_id = next(iter(chunks_with_ids)).metadata["unique_id"]

        existing_items = vector_store.get(where={"unique_id": first_unique_id})
        existing_ids = set(existing_items["ids"])

        for chunk in chunks_with_ids:
            if chunk.metadata["id"] not in existing_ids:
                new_chunks.append(chunk)
                isNewUrl = True
            else:
                existing_chunks.append(chunk)
                isNewUrl = False

        if len(new_chunks):
            new_chunk_ids = [chunk.metadata["id"] for chunk in new_chunks]
            vector_store.add_documents(documents=new_chunks, ids=new_chunk_ids)
            # vector_store.persist()

        return {"documents": len(new_chunks), "isNewUrl": isNewUrl}

    except Exception as e:
        print(f"Error processing URL {url}: {e}")
        return f"Error processing URL {url}: {e}"


def calculate_chunk_ids(chunks):
    last_url_hash = None
    current_chunk_index = 0

    for chunk in chunks:
        url = chunk.metadata.get("source")
        url_hash = hashlib.sha256(url.encode("utf-8")).hexdigest()
        
        # If the URL hash is the same as the last one, increment the index.
        if url_hash == last_url_hash:
            current_chunk_index += 1
        else:
            current_chunk_index = 0

        # Calculate the chunk ID using the URL hash and chunk index.
        chunk_id = f"{url_hash}:{current_chunk_index}"
        last_url_hash = url_hash

        # Add it to the chunk meta-data.
        chunk.metadata["id"] = chunk_id
        chunk.metadata["unique_id"] = chunk_id.split(":")[0]

    return chunks

def delete_url_db(url):
    try:
        url_hash = hashlib.sha256(url.encode("utf-8")).hexdigest()

        # Fetch documents from the vector store where the unique_id matches
        existing_items = vector_store.get(where={"unique_id": url_hash})
        
        # Check if any records exist with the provided unique_id
        if not existing_items or not existing_items["ids"]:
            return {"message": "No records found with the provided unique ID.", "unique_id": url_hash, "source": url}

        # Delete the documents from the vector store
        vector_store.delete(ids=existing_items["ids"])
        
        return {"message": "Records successfully deleted.", "unique_id": url_hash, "source": url}
    
    except Exception as e:
        return {"message": f"Error deleting records", "unique_id": url_hash, "source": url}

def get_all_urls_db():
    try:
        response = vector_store.get(
            limit=1000,
            offset=0
        )

        seen_sources = set()
        unique_data = []

        for metadata in response["metadatas"]:
            if metadata['unique_id'] not in seen_sources:
                unique_data.append(metadata)
                seen_sources.add(metadata['unique_id'])

        return unique_data
    
    except Exception as e:
        return {"message": f"Error retrieving documents: {str(e)}"}
