from langchain.schema import format_document
from langchain_core.messages import AIMessage, HumanMessage, get_buffer_string
from langchain_core.runnables import RunnableParallel
from langchain.chat_models import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.prompts.prompt import PromptTemplate
from langchain.vectorstores import Chroma, weaviate, chroma
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from operator import itemgetter
from googletrans import Translator
import streamlit as st

st.title("💬 தமிழ் சட்ட உதவியாளர் - Food legal assistant")

translator = Translator()

vectorstore = chroma.Chroma(
    collection_name="rag-chroma",
    persist_directory="./food-act-embedding/",
    embedding_function=OpenAIEmbeddings(),
)


retriever = vectorstore.as_retriever()

_template = """Assume you are legal assistant who helps food related laws in India. 
Help the user by explaining in a detailed answer

Chat History:
{chat_history}
Follow Up Input: {question}
Standalone question:"""
CONDENSE_QUESTION_PROMPT = PromptTemplate.from_template(_template)

template = """
* List down all the points that are in the context
* For each point in markdown explain it in detail with example
Answer the question based on the following context:
{context}

Question: {question}
"""
ANSWER_PROMPT = ChatPromptTemplate.from_template(template)

DEFAULT_DOCUMENT_PROMPT = PromptTemplate.from_template(template="{page_content}")


def _combine_documents(
    docs, document_prompt=DEFAULT_DOCUMENT_PROMPT, document_separator="\n\n"
):
    doc_strings = [format_document(doc, document_prompt) for doc in docs]
    print(docs)
    return document_separator.join(doc_strings)

_inputs = RunnableParallel(
    standalone_question=RunnablePassthrough.assign(
        chat_history=lambda x: get_buffer_string(x["chat_history"])
    )
    | CONDENSE_QUESTION_PROMPT
    | ChatOpenAI(temperature=0, api_key=st.secrets["OPENAI_API_KEY"])
    | StrOutputParser(),
)
_context = {
    "context": itemgetter("standalone_question") | retriever | _combine_documents,
    "question": lambda x: x["standalone_question"],
}
conversational_qa_chain = _inputs | _context | ANSWER_PROMPT | ChatOpenAI(api_key=st.secrets["OPENAI_API_KEY"])

chain = conversational_qa_chain


def translate(arg):
    return translator.translate(arg, 'ta').text



if "messages" not in st.session_state:
    st.session_state["messages"] = []

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

prompt_str = st.chat_input()


# chain.invoke(input=prompt_str)
if prompt_str:
    st.session_state.messages.append({"role": "user", "content": prompt_str})
    st.chat_message("user").write(prompt_str)
    msg1 = chain.invoke(
        {
            "question": prompt_str,
            "chat_history": [],
        }
    ).content
    tamil = translate(msg1)
    st.session_state.messages.append({"role": "assistant", "content": tamil})
    st.chat_message("assistant").write(tamil)