import os
import sys
import json
import time
import psycopg2
import matplotlib.pyplot as plt
from PIL import Image, UnidentifiedImageError
from unstructured.partition.pdf import partition_pdf
from unstructured.documents.elements import Title, NarrativeText, Image as UnstructuredImage, Table
from unstructured.documents.elements import Title, Image
from google.generativeai import GenerativeModel
import nltk
import matplotlib.pyplot as plt
from tenacity import retry, stop_after_attempt, wait_exponential
import argparse
import re
import math
from typing import List, Dict, Any
import base64
import threading
import random
import google.generativeai as genai #Import genai
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Retrieve API Key from environment variable
API_KEY = os.getenv("GEMINI_API_KEY")
if not API_KEY:
    raise ValueError("API key not found. Please set the GEMINI_API_KEY environment variable.")

genai.configure(api_key=API_KEY)  # Configure the Gemini API

# Set up environment paths dynamically
def setup_environment():
    poppler_path = os.getenv("POPPLER_PATH", r"C:\\Program Files\\poppler-24.08.0\\Library\\bin")
    tess_path = os.getenv("TESSDATA_PREFIX", r"C:\\Program Files\\Tesseract-OCR\\tessdata")
    os.environ['PATH'] += os.pathsep + poppler_path
    os.environ["TESSDATA_PREFIX"] = tess_path

setup_environment()




def main(chapter_path=None):
    if chapter_path is None:
        if len(sys.argv) > 1:
            chapter_path = sys.argv[1]  # Get argument from command line
        else:
            print("❌ Error: Missing chapter_path. Provide a valid PDF file path.")
            return

    # Ensure the path exists
    if not os.path.exists(chapter_path):
        print(f"❌ Error: The file {chapter_path} does not exist.")
        return
    
    print(f"✅ Processing file: {chapter_path}")



def extract_data(chapter_path: str) -> list:
    """Extract data from the PDF file."""
    return partition_pdf(
        filename=chapter_path,
        strategy="hi_res",
        extract_images_in_pdf=True,
        extract_image_block_to_payload=False,
        extract_image_block_output_dir="C:/OCR_Piepline/images8"
    )

chapter_path = r"C:\Users\HP\ocr_pipeline\ocr_pipeline\data\Physics\PDF's\chapter_9\Mechanical_Properties_of_Fluids.pdf"  

chapter_path_raw_data = extract_data(chapter_path)  # Now the function gets the required argument


# Extract Textual Component
def extract_text_with_metadata(chapter_path_raw_data, source_document):
    text_data = []
    paragraph_counters = {}
    current_topic_number = None
    current_topic_title = None

    for element in chapter_path_raw_data:
        if isinstance(element, Title):  # Detect and store the latest topic title
            match = re.match(r"(\d+(\.\d+)*)\s+(.*)", element.text.strip())
            if match:
                current_topic_number = match.group(1)  # Extract topic number (e.g., "1.1")
                current_topic_title = match.group(3)   # Extract topic title (e.g., "Introduction")

        elif isinstance(element, NarrativeText):
            page_number = element.metadata.page_number

            if page_number not in paragraph_counters:
                paragraph_counters[page_number] = 1
            else:
                paragraph_counters[page_number] += 1

            paragraph_number = paragraph_counters[page_number]

            text_data.append({
                "source_document": source_document,
                "page_number": page_number,
                "paragraph_number": paragraph_number,
                "topic_number": current_topic_number,  # Add topic number
                "topic_title": current_topic_title,    # Add topic title
                "text": element.text
            })

    return text_data

extracted_data = extract_text_with_metadata(chapter_path_raw_data, chapter_path)

# Extract Image Component
def extract_image_metadata(chapter_path_raw_data, source_document):
    image_data = []
    topic_hierarchy = {}  # Store topic structure
    current_topic_number = None
    current_topic_title = None
    paragraph_counters = {}

    for element in chapter_path_raw_data:
        if isinstance(element, Title):  # Track latest topic info
            match = re.match(r"(\d+(\.\d+)*)\s+(.*)", element.text.strip())
            if match:
                current_topic_number = match.group(1)  # Extract topic number
                current_topic_title = match.group(3)   # Extract topic title

        elif isinstance(element, Image):
            page_number = element.metadata.page_number
            image_path = element.metadata.image_path if hasattr(element.metadata, 'image_path') else None

            if not image_path:
                print(f"[WARNING] No image path found for Page {page_number}!")

            elif not os.path.exists(image_path):
                print(f"[ERROR] Image path {image_path} does not exist!")

            else:
                print(f"[INFO] Extracted image: {image_path} from Page {page_number}")


            # Maintain paragraph counters per page
            if page_number not in paragraph_counters:
                paragraph_counters[page_number] = 1
            else:
                paragraph_counters[page_number] += 1

            paragraph_number = paragraph_counters[page_number]

            image_data.append({
                "source_document": source_document,
                "page_number": page_number,
                "paragraph_number": paragraph_number,  # Keep numbering consistent
                "topic_number": current_topic_number,  # Attach topic number
                "topic_title": current_topic_title,    # Attach topic title
                "image_path": image_path  # Store image path if available
            })

    return image_data

extracted_image_data = extract_image_metadata(chapter_path_raw_data, chapter_path)



def display_images_from_metadata(extracted_image_data, images_per_row=4):
    valid_images = [img for img in extracted_image_data if img['image_path'] and os.path.exists(img['image_path'])]
    
    if not valid_images:
        print("No valid image data available.")
        return

    num_images = len(valid_images)
    num_rows = math.ceil(num_images / images_per_row)
    
    fig, axes = plt.subplots(num_rows, images_per_row, figsize=(20, 5 * num_rows))
    axes = axes.flatten() if num_rows > 1 else [axes]

    for ax, img_data in zip(axes, valid_images):
        try:
            with Image.open(img_data['image_path']) as img:
                img.verify()  # Ensure it's not corrupted
                img = Image.open(img_data['image_path'])  # Reopen since .verify() closes the file
                ax.imshow(img)
                ax.axis('off')
                ax.set_title(f"Page {img_data['page_number']}", fontsize=10)
        except UnidentifiedImageError:
            print(f"Skipping corrupted image: {img_data['image_path']}")
            ax.text(0.5, 0.5, "Corrupted image", ha='center', va='center')
        except Exception as e:
            print(f"Error loading image {img_data['image_path']}: {str(e)}")
            ax.text(0.5, 0.5, f"Error loading image\n{str(e)}", ha='center', va='center')
        ax.axis('off')

    for ax in axes[num_images:]:
        fig.delaxes(ax)

    plt.tight_layout()
    plt.show()

# Extract Table Components
def extract_table_metadata(chapter_path_raw_data, source_document):
    table_data = []
    topic_hierarchy = {}  # Stores topic hierarchy
    current_topic_number = None
    current_topic_title = None

    for element in chapter_path_raw_data:
        if isinstance(element, Title):  # Detect and store latest topic info
            match = re.match(r"(\d+(\.\d+)*)\s+(.*)", element.text.strip())
            if match:
                current_topic_number = match.group(1)  # Extract topic number
                current_topic_title = match.group(3)   # Extract topic title

        elif isinstance(element, Table):
            page_number = element.metadata.page_number

            # Convert table content to string format (modify if needed)
            table_content = str(element)

            table_data.append({
                "source_document": source_document,
                "page_number": page_number,
                "topic_number": current_topic_number,  # Attach topic number
                "topic_title": current_topic_title,    # Attach topic title
                "table_content": table_content  # Store table data
            })

    return table_data

extracted_table_data = extract_table_metadata(chapter_path_raw_data, chapter_path)

tables_summarizer_prompt = """  
As an experienced physics educator, analyze the table provided from a Class 11 Physics textbook.  
Summarize the table's contents in a way that helps students clearly understand what is represented in the data, such as variables, units, and key relationships between quantities.  
Additionally, explain what the table reveals about the topic, including any patterns, trends, or implications for understanding the underlying physics concepts.  
Ensure your explanation is precise and educational, helping students connect the information in the table to broader principles in physics.  

Table: {table_content}  

Limit your summary to 3-4 sentences, maintaining clarity, accuracy, and relevance to the topic of study.  
"""


# ... existing code ...

# Example usage of genai
genai.configure(api_key="AIzaSyCLk0MhIxbr2vOyT7B0jpSalHMnwHihPTU")
model = GenerativeModel("gemini-1.5-flash")

# ... existing code ...

model = genai.GenerativeModel("gemini-1.5-flash")

def extract_table_metadata_with_summary(chapter_path_raw_data, source_document, tables_summarizer_prompt):
    table_data = []
    topic_hierarchy = {}  # Stores topic hierarchy
    current_topic_number = None
    current_topic_title = None

    for element in chapter_path_raw_data:
        if isinstance(element, Title):  # Detect and store latest topic info
            match = re.match(r"(\d+(\.\d+)*)\s+(.*)", element.text.strip())
            if match:
                current_topic_number = match.group(1)  # Extract topic number
                current_topic_title = match.group(3)   # Extract topic title

        elif isinstance(element, Table):
            page_number = element.metadata.page_number
            table_content = str(element)  # Convert table content to string

            # Create a prompt for summarization
            prompt = f"{table_content}\n{tables_summarizer_prompt}"

            try:
                # Generate summary using the Gemini model
                response = model.generate_content(prompt)
                description = response.text.strip() if response.text else "No summary available"
            except Exception as e:
                description = f"Error generating summary: {str(e)}"

            table_data.append({
                "source_document": source_document,
                "page_number": page_number,
                "topic_number": current_topic_number,  # Attach topic number
                "topic_title": current_topic_title,    # Attach topic title
                "table_content": table_content,  # Store table data
                "description": description  # Store generated summary
            })

    return table_data

extracted_table_data_with_summary = extract_table_metadata_with_summary(chapter_path_raw_data, chapter_path, tables_summarizer_prompt)

# Get the first key-value pair in the dictionary
first_table_details = extracted_table_data_with_summary[0]

# Extract the transcription from the first item
first_description = first_table_details

# Image summarization
images_summarizer_prompt = """  
As an experienced physics educator, analyze the image provided from a Class 11 Physics textbook.  
Summarize the image's contents to help students clearly understand what is depicted, including key components such as variables, units, diagrams, or relationships between physical quantities.  
Additionally, explain the insights or principles the image conveys, including any patterns, trends, or implications for understanding the physics topic being studied.  
Provide a coherent explanation that connects the image's content to broader concepts in physics, ensuring it is accessible and educational for students.  

Image: {image_element}  

Limit your summary to 3-4 sentences, focusing on clarity, accuracy, and relevance to the topic.  
"""


def extract_image_metadata_with_summary(chapter_path_raw_data, chapter_path, images_summarizer_prompt):
    model = GenerativeModel('gemini-1.5-flash')

    requests_per_minute = 30  
    delay = 60.0 / requests_per_minute
    last_request_time = time.time()
    
    image_data = []
    current_topic_number = None
    current_topic_title = None

    for element in chapter_path_raw_data:
        if isinstance(element, Title):  
            match = re.match(r"(\d+(\.\d+)*)\s+(.*)", element.text.strip())
            if match:
                current_topic_number = match.group(1)
                current_topic_title = match.group(3)

        elif isinstance(element, UnstructuredImage):
            page_number = element.metadata.page_number if hasattr(element.metadata, 'page_number') else None
            image_path = element.metadata.image_path if hasattr(element.metadata, 'image_path') else None

            if image_path and os.path.exists(image_path):
                try:
                    img = Image.open(image_path)
                    img.verify()  # Detect corrupted images
                    img = Image.open(image_path)  # Reopen for processing
                    if img.size == (0, 0):  # Skip empty images
                        print(f"⚠️ Skipping empty image: {image_path}")
                        continue
                except Exception as e:
                    print(f"⚠️ Skipping corrupted image {image_path}: {str(e)}")
                    continue


                    with open(image_path, "rb") as image_file:
                        image_bytes = image_file.read()
                        encoded_string = base64.b64encode(image_bytes).decode('utf-8')

                    time_since_last_request = time.time() - last_request_time
                    if time_since_last_request < delay:
                        time.sleep(delay - time_since_last_request)

                    retry_attempts = 0
                    max_retries = 5
                    wait_time = 30  # Start with 30s

                    while retry_attempts < max_retries:
                        try:
                            response = model.generate_content([
                                images_summarizer_prompt,
                                {'mime_type': 'image/jpeg', 'data': image_bytes}
                            ])
                            description = response.text.strip() if hasattr(response, 'text') else "No description available"
                            last_request_time = time.time()
                            break  # Success, exit loop

                        except Exception as e:
                            if "429" in str(e):  
                                print(f"⚠️ Rate limit reached, retrying in {wait_time} seconds... (Attempt {retry_attempts+1}/{max_retries})")
                                time.sleep(wait_time)
                                wait_time *= 2  # Exponential backoff
                                retry_attempts += 1
                            else:
                                print(f"⚠️ Error processing image {image_path}: {str(e)}")
                                description = "Error generating description"
                                break  # Exit loop on other errors
  

                    image_data.append({
                        "source_document": chapter_path,
                        "page_number": page_number,
                        "image_path": image_path,
                        "topic_number": current_topic_number,
                        "topic_title": current_topic_title,
                        "description": description,
                        "base64_encoding": encoded_string
                    })

                except UnidentifiedImageError:
                    print(f"Skipping corrupted image: {image_path}")
                except Exception as e:
                    print(f"Unexpected error processing image {image_path}: {str(e)}")

            else:
                print(f"Warning: Image file not found or path missing on page {page_number}")

    return image_data


# Chunking 
class ContentChunker:
    def __init__(
        self,
        chunk_size: int = 512,
        chunk_overlap: int = 50,
        min_chunk_size: int = 100
    ):
        """
        Initialize the ContentChunker with configuration parameters.
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.min_chunk_size = min_chunk_size
        
        # Ensure required NLTK data is available
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt')

    def preprocess_text(self, text: str) -> str:
        """Clean and normalize text before chunking."""
        text = re.sub(r'\s+', ' ', text.strip())  # Remove extra spaces
        text = text.replace('\n', ' ')  # Normalize line breaks
        return text

    def create_chunks(self, text: str) -> List[str]:
        """Split text into overlapping chunks."""
        text = self.preprocess_text(text)
        sentences = nltk.sent_tokenize(text)
        
        chunks = []
        current_chunk = []
        current_length = 0
        
        for sentence in sentences:
            sentence_length = len(sentence)
            
            if current_length + sentence_length > self.chunk_size and current_chunk:
                # Save the current chunk
                chunks.append(' '.join(current_chunk))
                
                # Retain the overlap
                overlap_chunk = []
                overlap_length = 0
                for prev_sentence in reversed(current_chunk):
                    if overlap_length + len(prev_sentence) <= self.chunk_overlap:
                        overlap_chunk.insert(0, prev_sentence)
                        overlap_length += len(prev_sentence)
                    else:
                        break
                
                current_chunk = overlap_chunk
                current_length = overlap_length
            
            current_chunk.append(sentence)
            current_length += sentence_length
        
        # Add the final chunk if it's large enough
        if current_chunk and current_length >= self.min_chunk_size:
            chunks.append(' '.join(current_chunk))
            
        return chunks

    def chunk_text_content(self, extracted_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Process regular text content with metadata."""
        chunked_data = []

        for item in extracted_data:
            text = item.get('text', '').strip()
            if not text:
                continue  # Skip empty entries
                
            chunks = self.create_chunks(text)
            
            for chunk_idx, chunk in enumerate(chunks):
                chunked_data.append({
                    'source_document': item.get('source_document', ''),
                    'page_number': item.get('page_number', None),
                    'paragraph_number': item.get('paragraph_number', None),
                    'topic_number': item.get('topic_number', None),
                    'topic_title': item.get('topic_title', ''),
                    'chunk_index': chunk_idx,
                    'total_chunks': len(chunks),
                    'text': chunk,
                    'content_type': 'text'
                })
        
        return chunked_data

    def chunk_image_content(self, image_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Process image descriptions with metadata."""
        chunked_data = []

        for item in image_data:
            description = item.get('description', '').strip()
            if not description:
                continue  # Skip empty descriptions
            
            chunks = self.create_chunks(description)
            
            for chunk_idx, chunk in enumerate(chunks):
                chunked_data.append({
                    'source_document': item.get('source_document', ''),
                    'page_number': item.get('page_number', None),
                    'image_path': item.get('image_path', ''),
                    'base64_encoding': item.get('base64_encoding', ''),
                    'topic_number': item.get('topic_number', None),
                    'topic_title': item.get('topic_title', ''),
                    'chunk_index': chunk_idx,
                    'total_chunks': len(chunks),
                    'text': chunk,
                    'content_type': 'image',
                    'original_description': description
                })
        
        return chunked_data

    def chunk_table_content(self, table_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Process table content with metadata."""
        chunked_data = []

        for item in table_data:
            description = item.get('description', '').strip()
            if not description:
                continue  # Skip empty tables
            
            chunks = self.create_chunks(description)
            
            for chunk_idx, chunk in enumerate(chunks):
                chunked_data.append({
                    'source_document': item.get('source_document', ''),
                    'page_number': item.get('page_number', None),
                    'table_content': item.get('table_content', ''),
                    'topic_number': item.get('topic_number', None),
                    'topic_title': item.get('topic_title', ''),
                    'chunk_index': chunk_idx,
                    'total_chunks': len(chunks),
                    'text': chunk,
                    'content_type': 'table',
                    'original_description': description  
                })
        
        return chunked_data

def process_all_content_with_chunking(
    text_data: List[Dict[str, Any]] = None,
    image_data: List[Dict[str, Any]] = None,
    table_data: List[Dict[str, Any]] = None,
    chunk_size: int = 512,
    chunk_overlap: int = 50,
    min_chunk_size: int = 100
) -> Dict[str, List[Dict[str, Any]]]:
    """
    Process all types of content with appropriate chunking.
    
    Args:
        text_data: List of text content items.
        image_data: List of image content items.
        table_data: List of table content items.
        chunk_size: Target chunk size.
        chunk_overlap: Overlap between chunks.
        min_chunk_size: Minimum chunk size.
        
    Returns:
        Dictionary with chunked data for each content type.
    """
    chunker = ContentChunker(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        min_chunk_size=min_chunk_size
    )
    
    result = {}
    
    if text_data:
        result['text'] = chunker.chunk_text_content(text_data)
    
    if image_data:
        result['image'] = chunker.chunk_image_content(image_data)
    
    if table_data:
        result['table'] = chunker.chunk_table_content(table_data)
    
    return result

chunked_content = process_all_content_with_chunking(
    text_data=extracted_data,
    image_data=extracted_image_data,
    table_data=extracted_table_data_with_summary
)

# First, let's debug what we have in chunked_content
print("Contents in chunked_content:")
for content_type, data in chunked_content.items():
    print(f"{content_type}: {len(data)} items")

# Extract text, image, and table chunks properly
text_chunks = chunked_content.get("text", [])
image_chunks = chunked_content.get("image", [])
table_chunks = chunked_content.get("table", [])

# Now let's check what we're actually getting in our chunks
print("\nChecking chunks before ingestion:")
print(f"Text chunks: {len(text_chunks)}")
print(f"Image chunks: {len(image_chunks)}")
print(f"Table chunks: {len(table_chunks)}")

# Show chunked content
chunked_content

print("\nExamining chunked content structure:")
for content_type, data in chunked_content.items():
    print(f"\n{content_type} data structure example:")
    if data:
        example_item = data[0]
        print("Keys available:", example_item.keys())
        print("Content type field:", example_item.get('content_type'))
        
        # Print full example item for inspection
        print("\nFull example item:")
        for key, value in example_item.items():
            print(f"{key}: {value[:100] if isinstance(value, str) else value}")

class PhysicsContextGenerator:
    def __init__(self, api_key: str):
        #genai.configure(api_key=api_key)  # Ensure this is correct
        self.token_counts = {
            'input': 0,
            'output': 0,
            'cache_read': 0,
            'cache_creation': 0
        }
        self.token_lock = threading.Lock()

    @retry(
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        reraise=True
    )
    def generate_context_prompt(self, doc_content: str, chunk_content: str, content_type: str) -> str:
        """Generate appropriate prompt based on content type"""
        base_prompt = f"""
        <document>
        {doc_content}
        </document>
        """
        
        if content_type == 'text':
            chunk_prompt = f"""
            Here is the textual chunk we want to situate within the physics document:
            <chunk>
            {chunk_content}
            </chunk>
            Provide a brief context (2-3 sentences) that explains how this chunk relates to the physics concepts in the document.
            Focus on key physics principles, mathematical relationships, or experimental setups mentioned.
            Answer only with the succinct context and nothing else.
            """
        elif content_type == 'image':
            chunk_prompt = f"""
            Here is the description of an image from the physics document:
            <chunk>
            {chunk_content}
            </chunk>
            Provide a brief context (2-3 sentences) that explains what physics concepts this image illustrates
            and how it relates to the main topic. Focus on the physics principles being visualized.
            Answer only with the succinct context and nothing else.
            """
        else:  # table
            chunk_prompt = f"""
            Here is the content of a table from the physics document:
            <chunk>
            {chunk_content}
            </chunk>
            Provide a brief context (2-3 sentences) that explains what physics data/relationships this table represents
            and its significance. Focus on the quantities being compared or measured.
            Answer only with the succinct context and nothing else.
            """

        return base_prompt + chunk_prompt

    def generate_chunk_context(self, doc_content: str, chunk_content: str, content_type: str) -> tuple[str, Any]:
        """Generate context for a single chunk using Gemini with retry logic"""
        try:
            # Add a small random delay to prevent burst requests
            time.sleep(random.uniform(0.5, 1.5))
            
            prompt = self.generate_context_prompt(doc_content, chunk_content, content_type)
            
            response = model.generate_content(
                contents=[prompt],
                generation_config={"temperature": 0.1}
            )
            
            # Extract token usage if available
            usage = getattr(response, 'usage', None)
            if usage:
                with self.token_lock:
                    self.token_counts['input'] += usage.prompt_tokens
                    self.token_counts['output'] += usage.completion_tokens
                    
            return response.text, usage
        except Exception as e:
            if "429" in str(e):
                print(f"Rate limit hit, retrying after delay...")
                raise  # This will trigger the retry logic
            raise  # Re-raise other exceptions

    def process_chunk(self, chunk: Dict[str, Any], content_type: str) -> Dict[str, Any]:
        """Process a single chunk to generate context and prepare for embedding."""
        
        chunk_content = chunk.get('text' if content_type == 'text' else 
                                  'description' if content_type == 'image' else 
                                  'table_content', '')
    
        doc_content = chunk.get('source_document', '')
    
        # Extract topic information
        topic_number = chunk.get('topic_number', None)
        topic_title = chunk.get('topic_title', None)
    
        context, usage = self.generate_chunk_context(doc_content, chunk_content, content_type)
        
        return {
            'text_to_embed': f"{chunk_content}\n\nContext: {context}",
            'metadata': {
                'doc_id': chunk.get('source_document', ''),
                'page_number': chunk.get('page_number', 0),
                'paragraph_number': chunk.get('paragraph_number', 0) if content_type == 'text' else None,
                'original_content': chunk_content,
                'contextualized_content': context,
                'content_type': content_type,
                "topic_number": topic_number, 
                "topic_title": topic_title,
            }
        }

class EnhancedContentIngestion:
    def __init__(
        self,
        api_key: str,
        output_dims: int = 768,
        db_params: Dict[str, Any] = None,
        batch_size: int = 10
    ):
        self.context_generator = PhysicsContextGenerator(api_key)
        #genai.configure(api_key=api_key)
        self.output_dims = output_dims
        self.batch_size = batch_size
        
        self.db_params = db_params or {
            "dbname": "physics_db",
            "user": "postgres",
            "password": "002652",
            "host": "localhost",
            "port": "5432"
        }

    def clean_content(self, text: str) -> str:
        """Removes '\n\nContext:' from text content."""
        if isinstance(text, str):
            return re.sub(r'\n\nContext:', '', text).strip()  # Remove extra spaces
        return text

    def process_chunk(self, chunk: Dict[str, Any], content_type: str) -> Dict[str, Any]:
        """Process a single chunk to generate context and prepare for embedding"""
        try:
            chunk_content = chunk.get('text', '') if content_type == 'text' else chunk.get('description', '')

            context, usage = self.context_generator.generate_chunk_context(
                chunk.get('source_document', ''), chunk_content, content_type
            )

            cleaned_text = self.clean_content(f"{chunk_content}\n\nContext: {context}")

            return {
                'text_to_embed': cleaned_text,
                'metadata': {
                    'doc_id': chunk.get('source_document', ''),
                    'page_number': chunk.get('page_number', 0),
                    'content_type': content_type,
                    'original_content': chunk_content,
                    'contextualized_content': context,
                    'topic_number': str(chunk.get('topic_number', '')),  # Convert to TEXT
                    'topic_title': chunk.get('topic_title', '')
                }
            }
        except Exception as e:
            print(f"Error processing chunk: {str(e)}\nChunk data: {chunk}")
            return None

    def process_batch(self, batch: List[Dict], content_type: str, topic_number: int, topic_title: str) -> List[Dict]:
        """Process a batch of chunks"""
        processed_results = []
        for item in batch:
            try:
                result = self.process_chunk(item, content_type)
                if result:
                    processed_results.append(result)
            except Exception as e:
                print(f"Error in batch processing: {str(e)}")
        return processed_results

    def ingest_chapter_content(
        self,
        subject: str,
        chapter_name: str,
        text_data: List[Dict],
        image_data: List[Dict],
        table_data: List[Dict],
        topic_number: int,
        topic_title: str
    ):
        """Enhanced ingestion with context generation using batch processing"""
        print(f"\nIngesting content for chapter: {chapter_name}")

        def process_content_type(data, content_type):
            processed_results = []
            for i in range(0, len(data), self.batch_size):
                batch = data[i:i + self.batch_size]
                print(f"\nProcessing {content_type} batch {(i//self.batch_size)+1}")

                batch_results = self.process_batch(batch, content_type, topic_number, topic_title)
                processed_results.extend(batch_results)

                if batch_results:
                    self._store_in_database(subject, chapter_name, batch_results)
                    print(f"✓ Stored batch of {len(batch_results)} items")

                time.sleep(2)

            return processed_results

        total_processed = 0

        if text_data:
            text_results = process_content_type(text_data, 'text')
            total_processed += len(text_results)
        if image_data:
            image_results = process_content_type(image_data, 'image')
            total_processed += len(image_results)
        if table_data:
            table_results = process_content_type(table_data, 'table')
            total_processed += len(table_results)

        print(f"\n✓ Total items processed and stored: {total_processed}")

    def _store_in_database(self, subject: str, chapter_name: str, processed_content: List[Dict]):
        """Store processed content in PostgreSQL"""
        conn = psycopg2.connect(**self.db_params)
        cur = conn.cursor()
        
        try:
            for item in processed_content:
                cleaned_text = self.clean_content(item['text_to_embed'])  # Apply cleaning here
                embedding = self.generate_embedding(cleaned_text)

                # ✅ **Fix: Skip empty embeddings**
                if not embedding:  # If embedding is empty, skip this item
                    print(f"⚠️ Skipping database insertion: Empty embedding for {item['metadata'].get('topic_title', 'Unknown Topic')}")
                    continue  # Move to the next item

                vector_str = f"[{','.join(map(str, embedding))}]"  # Convert embedding to vector format

                sql = """
                    INSERT INTO physics_topics 
                    (subject, chapter_name, content_type, content, metadata, embedding, topic_number, topic_title)
                    VALUES (%s, %s, %s, %s, %s, %s::vector, %s, %s)
                """
                values = (
                    subject,
                    chapter_name,
                    item['metadata']['content_type'],
                    json.dumps({'text': cleaned_text}),  # Store cleaned text
                    json.dumps(item['metadata']),
                    vector_str,  # ✅ Ensure it's a valid vector
                    str(item['metadata'].get('topic_number', '')),  
                    item['metadata'].get('topic_title', '')
                )

                cur.execute(sql, values)  # Execute the SQL command

            conn.commit()  # Commit all transactions
            print("✅ Data stored successfully.")

        except Exception as e:
            conn.rollback()
            print(f"❌ Error storing in database: {str(e)}")

        finally:
            cur.close()
            conn.close()


    def generate_embedding(self, text: str) -> List[float]:
        """Generate embedding using Gemini model"""
        try:
            time.sleep(0.5)  # Prevent rate limiting
            result = genai.embed_content(
                model="models/text-embedding-004",  # Correct model name format
                content=text,
                output_dimensionality=self.output_dims
            )

            return result["embedding"]
        except Exception as e:
            print(f"Error generating embedding: {str(e)}")
            return []

# Initialize ingestion with smaller batch size
ingestion = EnhancedContentIngestion(
    api_key=API_KEY,
    batch_size=5 
)


# Get the chunked data
text_chunks = chunked_content.get('text', [])
image_chunks = chunked_content.get('image', [])
table_chunks = chunked_content.get('table', [])

# Get topic metadata
topic_number = chunked_content.get("topic_number", None)
topic_title = chunked_content.get("topic_title", None)

# Print counts before ingest
print("\nData to be ingested:")
print(f"Text chunks: {len(text_chunks)}")
print(f"Image chunks: {len(image_chunks)}")
print(f"Table chunks: {len(table_chunks)}")

# Use the enhanced ingestion
ingestion.ingest_chapter_content(
    subject="Physics",
    chapter_name="Mechanical_Properties_of_Fluids",
    text_data=text_chunks,
    image_data=image_chunks,
    table_data=table_chunks,
    topic_number=topic_number,
    topic_title=topic_title
)

def setup_environment():
    os.environ['PATH'] += os.pathsep + r'C:\Program Files\poppler-24.08.0\Library\bin'
    os.environ["TESSDATA_PREFIX"] = r"C:\Program Files\Tesseract-OCR\tessdata"
    os.environ["PATH"] += os.pathsep + r"C:\Program Files\Tesseract-OCR"

def main():
    setup_environment()
    
    chapter_path = r"C:\Users\HP\ocr_pipeline\ocr_pipeline\data\Physics\PDF's\chapter_9\Mechanical_Properties_of_Fluids.pdf"
    
    # Ensure the path is valid and the file exists
    if not os.path.exists(chapter_path):
        print(f"Error: The file {chapter_path} does not exist.")
        return
    
    chapter_path_raw_data = extract_data(chapter_path)
    
    extracted_data = extract_text_with_metadata(chapter_path_raw_data, chapter_path)
    extracted_image_data = extract_image_metadata(chapter_path_raw_data, chapter_path)
    extracted_table_data = extract_table_metadata(chapter_path_raw_data, chapter_path)
    
    # Display images
    display_images_from_metadata(extracted_image_data)
    
    # Process and chunk content
    chunked_content = process_all_content_with_chunking(
        text_data=extracted_data,
        image_data=extracted_image_data,
        table_data=extracted_table_data
    )
    
    # Ingest content into the database
    ingestion = EnhancedContentIngestion(api_key=os.getenv("GEMINI_API_KEY"), batch_size=5)
    ingestion.ingest_chapter_content(
        subject="Physics",
        chapter_name="Mechanical_Properties_of_Solids",
        text_data=chunked_content.get('text', []),
        image_data=chunked_content.get('image', []),
        table_data=chunked_content.get('table', []),
        topic_number=None,  # Set appropriately
        topic_title=None    # Set appropriately
    )

# # Command-line execution
# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(description="Process a physics textbook chapter PDF.")
#     parser.add_argument("chapter_path", type=str, help="Path to the PDF file to process.")
#     args = parser.parse_args()
#     main(args.chapter_path)

# Run the script
if __name__ == "__main__":
    main()