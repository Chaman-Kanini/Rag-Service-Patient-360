import tiktoken
from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader
 
 
def extract_text_from_pdf(pdf_path: str) -> str:
    """
    Extract text from PDF file using PyPDFLoader.
   
    Args:
        pdf_path: Path to the PDF file
       
    Returns:
        Extracted text
    """
    text = ""
    try:
        loader = PyPDFLoader(pdf_path)
        documents = loader.load()
       
        for i, doc in enumerate(documents):
            if doc.page_content:
                text += f"\n[Page {i + 1}]\n{doc.page_content}"
 
        return text
    except Exception as e:
        raise Exception(f"Error extracting text from PDF: {str(e)}")
 
 
def extract_text_from_docx(docx_path: str) -> str:
    """
    Extract text from DOC/DOCX file using Docx2txtLoader.
   
    Args:
        docx_path: Path to the DOC/DOCX file
       
    Returns:
        Extracted text
    """
    text = ""
    try:
        loader = Docx2txtLoader(docx_path)
        documents = loader.load()
       
        for i, doc in enumerate(documents):
            if doc.page_content:
                text += f"\n[Section {i + 1}]\n{doc.page_content}"
 
        return text
    except Exception as e:
        raise Exception(f"Error extracting text from DOCX: {str(e)}")
 
 
def extract_text_from_document(file_path: str) -> str:
    """
    Extract text from a document file (PDF, DOC, or DOCX).
    Automatically detects file type based on extension.
   
    Args:
        file_path: Path to the document file
       
    Returns:
        Extracted text
    """
    file_path_obj = Path(file_path)
    extension = file_path_obj.suffix.lower()
   
    if extension == '.pdf':
        return extract_text_from_pdf(file_path)
    elif extension in ['.doc', '.docx']:
        return extract_text_from_docx(file_path)
    else:
        raise Exception(f"Unsupported file type: {extension}. Supported types: .pdf, .doc, .docx")
 
 
def tokenize_and_chunk(text: str, chunk_size: int = 1000, overlap: int = 200) -> list:
    """Chunk text into overlapping chunks based on token count."""
    try:
        enc = tiktoken.get_encoding("cl100k_base")
        tokens = enc.encode(text)
 
        chunks = []
        start = 0
        total_tokens = len(tokens)
 
        while start < total_tokens:
            end = start + chunk_size
            chunk_tokens = tokens[start:end]
            chunk_text = enc.decode(chunk_tokens)
            chunks.append(chunk_text)
            start = end - overlap
            if start < 0:
                start = 0
 
        return chunks
    except Exception as e:
        raise Exception(f"Error chunking text: {str(e)}")