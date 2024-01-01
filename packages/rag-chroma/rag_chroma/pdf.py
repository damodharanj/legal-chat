# import pdfkit
# pdf_file = open("/Users/damodharan-2579/Downloads/FOOD-ACT.pdf", "rb")

# html_file = pdfkit.from_file(pdf_file, "my_html_file.html")

# pdf_file.close()

# ================================================================


# import fitz  # PyMuPDF

# def pdf_to_html(pdf_file_path, html_output_path):
#     # Open the PDF file
#     pdf_document = fitz.open(pdf_file_path)

#     # Initialize HTML content
#     html_content = ''

#     # Iterate through each page in the PDF
#     for page_number in range(pdf_document.page_count):
#         page = pdf_document.load_page(page_number)
#         # Get the text content from the page
#         html_content += page.get_text("html")

#     # Write the HTML content to a file
#     with open(html_output_path, "w", encoding="utf-8") as html_file:
#         html_file.write(html_content)

#     # Close the PDF document
#     pdf_document.close()

# # Replace 'input.pdf' and 'output.html' with your file paths
# pdf_to_html('/Users/damodharan-2579/Downloads/FOOD-ACT.pdf', 'output.html')

# ================================================================

# from pdfminer.high_level import extract_text_to_fp
# from io import StringIO

# def pdf_to_html(pdf_file_path, html_output_path):
#     output_string = StringIO()  # Create a StringIO object to store extracted text

#     with open(html_output_path, "rb") as html_file:
#         # Use extract_text_to_fp to extract text to the StringIO object
#         extract_text_to_fp(pdf_file_path, output_string, output_type='html', mode="rb", codec='utf-8')

#         # Write the extracted text (in HTML format) to the output file
#         html_file.write(output_string.getvalue().encode('utf-8'))

# # Replace 'input.pdf' and 'output.html' with your file paths
# pdf_to_html('/Users/damodharan-2579/Downloads/FOOD-ACT.pdf', 'output1.html')

# ================================================================

