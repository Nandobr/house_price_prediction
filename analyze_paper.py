from pypdf import PdfReader
import os

pdf_path = "Using_Artificial_Intelligence-Based_Machine_Learni.pdf"

try:
    reader = PdfReader(pdf_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text() + "\n"
    
    # Save to a text file for review
    with open("paper_content.txt", "w", encoding="utf-8") as f:
        f.write(text)
        
    print("Paper content extracted to paper_content.txt")
    print("Preview of first 1000 chars:")
    print(text[:1000])
except Exception as e:
    print(f"Error reading PDF: {e}")
