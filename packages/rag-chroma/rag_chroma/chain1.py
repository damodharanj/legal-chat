from langchain.chat_models import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.vectorstores import Chroma, weaviate, chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.pydantic_v1 import BaseModel
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain_core.runnables import ConfigurableField
from langchain.memory import ConversationBufferMemory
from operator import itemgetter
from googletrans import Translator
from langchain_core.messages import AIMessage, HumanMessage, get_buffer_string


translator = Translator()
# Example for document loading (from url), splitting, and creating vectostore


vectorstore = chroma.Chroma(
    collection_name="rag-chroma",
    persist_directory="/Users/damodharan-2579/food-act-embedding/",
    embedding_function=OpenAIEmbeddings(),
)


retriever = vectorstore.as_retriever()
# retriever.aget_relevant_documents()

# Embed a single document as a test
# vectorstore = chroma.Chroma.from_texts(
#     ["harrison worked at kensho"],
#     collection_name="rag-chroma",
#     embedding=OpenAIEmbeddings(),
# )
# retriever = vectorstore.as_retriever()

# RAG prompt
template = """Assume you are legal assistant who helps food related laws in India. 
Answer the question elaborately point wise based only on the following context:
Country: India
State: {state}
{context}

"""
# prompt = ChatPromptTemplate.from_template(template)

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", template),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{input}")
    ]
)

# print(prompt.messages)

memory = ConversationBufferMemory(return_messages=True)

memory.load_memory_variables({})


# LLM
model = ChatOpenAI(temperature=0).configurable_fields(
    temperature=ConfigurableField(
        name='temperature',
        id='temperature'
    )
)

from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.llms import Ollama

llm = Ollama(
    model="llama2",
    callback_manager=CallbackManager([StreamingStdOutCallbackHandler()])
)


def translate(arg):
    return translator.translate(arg, 'ta').text

 

# {
#     "context": retriever, 
#     "question_da": RunnablePassthrough(),
#     "state": RunnableLambda(lambda x: 'tamil_nadu')
# }

def getMemory():
    print('memory.chat_memory.messages', memory.chat_memory.messages)
    return RunnableLambda(memory.load_memory_variables) | itemgetter("history")

# RAG chain
chain = (
    {
        "context": retriever, 
        "input": RunnablePassthrough(),
        "state": RunnableLambda(lambda x: 'tamil_nadu')
    }
    | RunnablePassthrough.assign(
        history=getMemory(),
    )
    | prompt
    | model
    | StrOutputParser()
)



msg1 = chain.invoke(input="hello!", history=[HumanMessage(content="My name is Damo")])
# print(memory.chat_memory)

# print()

# Add typing for input
class Question_D(BaseModel):
    __root__: str


chain = chain.with_types(input_type=Question_D)

# import streamlit as st

# if "messages" not in st.session_state:
#     st.session_state["messages"] = []

# for msg in st.session_state.messages:
#     st.chat_message(msg["role"]).write(msg["content"])

# prompt_str = st.chat_input()


# # chain.invoke(input=prompt_str)
# if prompt_str:
#     msg1 = chain.invoke(input=prompt_str)
#     st.session_state.messages.append({"role": "user", "content": prompt_str})
#     st.chat_message("user").write(prompt_str)
#     st.session_state.messages.append({"role": "assistant", "content": msg1})
#     st.chat_message("assistant").write(translate(msg1))