

# # =====
# from langchain.document_loaders import PDFMinerPDFasHTMLLoader
# loader = PDFMinerPDFasHTMLLoader("/Users/damodharan-2579/Downloads/FOOD-ACT.pdf")
# data = loader.load()[0]   # entire PDF is loaded as a single Document
# print(data)

# from bs4 import BeautifulSoup
# soup = BeautifulSoup(data.page_content,'html.parser')
# content = soup.find_all('div')

# import re
# cur_fs = None
# cur_text = ''
# snippets = []   # first collect all snippets that have the same font size
# for c in content:
#     sp = c.find('span')
#     if not sp:
#         continue
#     st = sp.get('style')
#     if not st:
#         continue
#     fs = re.findall('font-size:(\d+)px',st)
#     if not fs:
#         continue
#     fs = int(fs[0])
#     if not cur_fs:
#         cur_fs = fs
#     if fs == cur_fs:
#         cur_text += c.text
#     else:
#         snippets.append((cur_text,cur_fs))
#         cur_fs = fs
#         cur_text = c.text
# snippets.append((cur_text,cur_fs))



# # Note: The above logic is very straightforward. One can also add more strategies such as removing duplicate snippets (as
# # headers/footers in a PDF appear on multiple pages so if we find duplicates it's safe to assume that it is redundant info)

# from langchain.docstore.document import Document
# cur_idx = -1
# semantic_snippets = []
# # Assumption: headings have higher font size than their respective content
# for s in snippets:
#     # if current snippet's font size > previous section's heading => it is a new heading
#     if not semantic_snippets or s[1] > semantic_snippets[cur_idx].metadata['heading_font']:
#         metadata={'heading':s[0], 'content_font': 0, 'heading_font': s[1]}
#         metadata.update(data.metadata)
#         semantic_snippets.append(Document(page_content='',metadata=metadata))
#         cur_idx += 1
#         continue

#     # if current snippet's font size <= previous section's content => content belongs to the same section (one can also create
#     # a tree like structure for sub sections if needed but that may require some more thinking and may be data specific)
#     if not semantic_snippets[cur_idx].metadata['content_font'] or s[1] <= semantic_snippets[cur_idx].metadata['content_font']:
#         semantic_snippets[cur_idx].page_content += s[0]
#         semantic_snippets[cur_idx].metadata['content_font'] = max(s[1], semantic_snippets[cur_idx].metadata['content_font'])
#         continue

#     # if current snippet's font size > previous section's content but less than previous section's heading than also make a new
#     # section (e.g. title of a PDF will have the highest font size but we don't want it to subsume all sections)
#     metadata={'heading':s[0], 'content_font': 0, 'heading_font': s[1]}
#     metadata.update(data.metadata)
    
#     semantic_snippets.append(Document(page_content='',metadata=metadata))
#     cur_idx += 1

# print(semantic_snippets[4])



# ================================================================================================================================

from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import  chroma
from langchain.document_loaders import PyPDFLoader

loader = PyPDFLoader("/Users/damodharan-2579/Downloads/FOOD-ACT.pdf")
data = loader.load()

# f = data[0].page_content

# f.json

with open('pdf1.txt', 'w') as file:
    index = 0
    for i in range(len(data)):
        # print(i.dict.page_content)
        file.writelines([f"""Index: {i}:     {x.strip()} \n""" for x in data[i].page_content.split('\n\n')])
        index+=1
# Split
from langchain.text_splitter import RecursiveCharacterTextSplitter
text_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=0, separators=[". "])
all_splits = text_splitter.split_documents(data)

# Open a file in write mode ('w' signifies write mode)
with open('pdf.txt', 'w') as file:
    for i in all_splits:
        file.writelines(i.json())
        file.writelines("\n\n\n\n")


# # Add to vectorDB
# vectorstore = chroma.Chroma.from_documents(documents=all_splits, 
#                                     collection_name="rag-chroma",
#                                     embedding=OpenAIEmbeddings(),
#                                     persist_directory="/Users/damodharan-2579/food-act-embedding"
#                                     )

# vectorstore.persist()

