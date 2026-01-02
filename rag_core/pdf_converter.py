import os
import google.generativeai as genai
from pdf2image import convert_from_path
import typing

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

def extract_tables_from_image(image, page_num: int) -> str:
    """
    Use Gemini Vision to extract tables and text from an image page.
    Returns structured Markdown.
    """
    prompt = """
    You are a data extraction specialist. 
    1. Analyze this image of a car specification sheet.
    2. Transcribe ALL tables into clean Markdown format.
    3. If there is text outside tables, summarize it briefly.
    4. Do not miss any numbers, especially regarding Battery, Range, and Motor.
    """
    
    print(f"Processing Page {page_num} with Gemini VLM...")
    try:
        response = model.generate_content([prompt, image])
        return response.text
    except Exception as e:
        print(f"Error processing page {page_num}: {e}")
        return ""

def process_document(pdf_path: str, output_md_path: str):
    """
    Full Pipeline: PDF -> Images -> VLM Extraction -> Markdown File
    """
    images = pdf_to_images(pdf_path)
    full_content = []
    
    for i, img in enumerate(images):
        page_content = extract_tables_from_image(img, i+1)
        full_content.append(f"## Page {i+1}\n\n{page_content}\n")
    
    # Save to file
    with open(output_md_path, "w") as f:
        f.write("\n".join(full_content))
    
    print(f"✅ Extraction complete! Saved to {output_md_path}")

if __name__ == "__main__":
    # Example usage
    pdf_path = "resouce/ES9_specs.pdf"
    output_path = "resouce/ES9_tables_extracted.md"
    
    if os.path.exists(pdf_path):
        process_document(pdf_path, output_path)
    else:
        print("PDF file not found. Please place your PDF in the resource folder.")
