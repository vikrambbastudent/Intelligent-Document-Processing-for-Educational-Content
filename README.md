# **📘 Intelligent Document Processing for Educational Content**

This project showcases an end-to-end pipeline for extracting, structuring, and storing educational content from unstructured textbook PDFs using **OCR**, **AI-based text processing**, and **semantic search**. The system is built for handling noisy scans, segmenting content by chapters and topics, and preparing it for downstream search and analytics tasks.

---

## **🧠 Problem Statement**

Educational textbooks, especially in scanned PDF format, often lack structured data, making it challenging to retrieve specific content such as definitions, tables, or illustrations. The goal of this project was to **build an intelligent document processing pipeline** that could accurately extract, clean, segment, and store the educational content from these PDFs in a **database-ready format**, including vector embeddings for semantic search.

---

## **🔍 Project Objectives**

- Extract text, images, and tables from scanned textbook PDFs using OCR and partitioning.
- Structure the content by chapter, topic title, and topic number.
- Generate metadata, learning objectives, and summaries using **Gemini AI**.
- Store all extracted content in a PostgreSQL database with **pgvector** for semantic search.
- Ensure the pipeline can handle real-world noisy and skewed data with minimal manual intervention.

---

## **🛠️ Technical Approach**

The solution was modularized into Python scripts across the following components:

- **Text & Media Extraction**: Used `unstructured.partition.pdf` to extract raw text, tables, and images. OCR fallback was implemented for scanned pages.
- **Image & Table Processing**: Applied Gemini API to generate contextual captions and observations for non-text elements.
- **Content Structuring**: Applied regex-based parsing to structure content into chapters, topics, and learning objectives.
- **Vector Embedding & Storage**: Generated 768-dimensional embeddings via Google Gemini and stored them using **PostgreSQL + pgvector**.
- **Database Management**: Created normalized schema (`documents`, `structured_content`, `content_elements`) to store processed content for search and analytics.

---


---

## **🧪 Key Results**

- Successfully processed >90% of scanned textbook content with minimal OCR errors.
- Automatically generated meaningful learning objectives and metadata for each topic.
- Enabled semantic search across textbook chapters using vector embeddings.
- Reduced manual processing time by 80% using automated chunking and tagging.

---

## **📚 Technologies Used**

- Python, Pandas, Regex
- **Google Gemini API** for AI text generation and embedding
- **PostgreSQL + pgvector** for vector storage and search
- Tesseract OCR (fallback)
- `unstructured.partition.pdf` for content segmentation

---

## **🚀 Future Enhancements**

- Build a **Streamlit-based search interface** for students and educators.
- Enable **multi-language support** for regional textbook content.
- Integrate PDF upload from front-end with real-time processing.

---

## **📌 How to Run**

```bash
# Clone the repo
git clone https://github.com/vikrambbastudent/intelligent-doc-processing.git

cd intelligent-doc-processing

# Install dependencies
pip install -r requirements.txt

# Run the pipeline
python ocr.py --input ./data/input.pdf --output ./outputs/
