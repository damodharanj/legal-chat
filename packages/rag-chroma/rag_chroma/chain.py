from langchain.chat_models import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.prompts import ChatPromptTemplate
from langchain.vectorstores import Chroma, weaviate, chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.pydantic_v1 import BaseModel
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain_core.runnables import ConfigurableField
from operator import itemgetter
from googletrans import Translator

translator = Translator()
# Example for document loading (from url), splitting, and creating vectostore


vectorstore = chroma.Chroma(
    collection_name="rag-chroma",
    persist_directory="/Users/damodharan-2579/food-act-embedding/",
    embedding_function=OpenAIEmbeddings(),
)


retriever = vectorstore.as_retriever().configurable_fields(
    search_type=ConfigurableField(
        id="search_type",
        name="search_type"
    ),
    search_kwargs=ConfigurableField(
        id="search_k",
        name="search_k"
    )
)
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

Question: {question_da}
"""
prompt = ChatPromptTemplate.from_template(template)

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


# RAG chain
chain = (
    {
        "context": retriever, 
        "question_da": RunnablePassthrough(),
        "state": RunnableLambda(lambda x: 'tamil_nadu')
    }
    | prompt
    | llm
    | StrOutputParser()
    | translate
)



# Add typing for input
class Question_D(BaseModel):
    __root__: str


chain = chain.with_types(input_type=Question_D)
