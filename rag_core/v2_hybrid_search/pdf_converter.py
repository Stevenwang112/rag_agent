import os
import google.generativeai as genai
from pdf2image import convert_from_path
from dotenv import load_dotenv
import typing

load_dotenv()

# Configure Gemini API
genai.configure(api_key=os.environ.get("GOOGLE_API_KEY"))

model = genai.GenerativeModel('gemini-2.0-flash-exp')

def pdf_to_images(pdf_path: str) -> list:
    """
    Convert PDF pages to a list of PIL Images.
    Requires poppler installed on the system.
    """
    print(f"Converting PDF: {pdf_path}...")
    images = convert_from_path(pdf_path)
    return images

def extract_content_mixed_format(image, page_num: int) -> str:
    """
    Use Gemini Vision to extract content from an image page.
    - Regular Text -> Markdown
    - Tables -> HTML (<table>...</table>)
    """
    prompt = """
    You are a document layout analysis expert. Your task is to transcribe the content of this document page exactly as you see it, while strictly adhering to the following formatting rules:

    1. **Text Content**: 
       - Convert all headers, paragraphs, lists, and regular text into standard **Markdown**.
       - Preserve the hierarchy (H1, H2, bullet points).
    
    2. **Tables**:
       - Identify ALL tables in the page.
       - Convert tables strictly into **HTML format** (`<table>`, `<thead>`, `<tbody>`, `<tr>`, `<th>`, `<td>`).
       - Do NOT use Markdown tables.
       - Preserve cell spans (rowspan, colspan) if visible.
       - Ensure the content within the table cells is accurate.

    3. **General**:
       - Do not add any introductory or concluding remarks. (e.g., "Here is the transcription...").
       - Output ONLY the mixed Markdown and HTML content.
       - Do not miss any numbers or technical specifications.
    """
    
    print(f"Processing Page {page_num} with Gemini VLM...")
    try:
        response = model.generate_content([prompt, image])
        return response.text
    except Exception as e:
        print(f"Error processing page {page_num}: {e}")
        return ""

import time

def process_document(pdf_path: str, output_md_path: str):
    """
    Full Pipeline: PDF -> Images -> VLM Extraction -> Markdown File
    """
    images = pdf_to_images(pdf_path)
    full_content = []
    
    for i, img in enumerate(images):
        page_content = extract_content_mixed_format(img, i+1)
        full_content.append(f"## Page {i+1}\n\n{page_content}\n")
        time.sleep(2) # Respect rate limits
    
    # Save to file
    with open(output_md_path, "w") as f:
        f.write("\n".join(full_content))
    
    print(f"✅ Extraction complete! Saved to {output_md_path}")

if __name__ == "__main__":
    # Example usage
    pdf_path = "resouce/SU7 参数配置表(202511).pdf"
    
    # Create output directory
    output_dir = "resouce/SU7_data"
    os.makedirs(output_dir, exist_ok=True)
    
    output_path = f"{output_dir}/SU7_mixed_content.md"
    
    if os.path.exists(pdf_path):
        process_document(pdf_path, output_path)
    else:
        print(f"PDF file not found: {pdf_path}")
