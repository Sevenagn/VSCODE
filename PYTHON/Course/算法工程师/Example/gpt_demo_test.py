import fitz  # PyMuPDF

def read_first_10_lines(input_path):
    pdf_document = fitz.open(input_path)

    first_10_lines = []
    max_pages = min(10, pdf_document.page_count)  # Read up to 10 pages or fewer if the PDF has fewer pages

    for page_number in range(max_pages):
        page = pdf_document[page_number]
        page_text = page.get_text("text")
        lines = page_text.strip().split("\n")
        first_10_lines.extend(lines[:10])  # Add the first 10 lines from each page to the list

    pdf_document.close()

    return first_10_lines

def main():
    input_file = r"D:\wechatbills\2019.pdf"   # 输入的PDF文件路径

    first_10_lines = read_first_10_lines(input_file)
    for line_number, line in enumerate(first_10_lines, start=1):
        print(f"Line {line_number}: {line}")

if __name__ == "__main__":
    main()
