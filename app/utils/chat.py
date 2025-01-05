from itertools import islice
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_chroma import Chroma
from app.config import COLLECTION_NAME, PERSIST_DIRECTORY_DB
from app.models.embedding_models import EmbeddingModels
from app.models.llm_models import LlmModels
from app.schemas.chat_schema import QueryUrlSchema

embeddingModels = EmbeddingModels()
nomicEmbeddings = embeddingModels.nomic_embeddings_ollama

# Initialize the vector store
vector_store = Chroma(
    collection_name=COLLECTION_NAME,
    embedding_function=nomicEmbeddings,
    persist_directory=PERSIST_DIRECTORY_DB,
)

def chatUrl(query:QueryUrlSchema):

    llmModels = LlmModels(query.model, query.temperature)
    llm = llmModels.model_ollama

    chat_history_text = ""
    combined_input = f"Input: {query.query}"

    if query.history:
        for chat in islice(query.history, max(0, len(query.history) - 3), len(query.history)):
            chat_history_text += f"User: {chat.User} \nAssistant: {chat.Assistant}\n"
            combined_input += f" \nHistory Input: {chat.User}"

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "You are a helpful assistant that provides answers based on the context provided."),
            ("human", "Input and historical input are here: {input}.\n Use the following chat history and context to answer. \nChat History: {history}. \nContext: {context}")
        ]
    )

    retriever = vector_store.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={
            "k": 10,
            "score_threshold": 0.1,
        },
    )

    combine_docs_chain = create_stuff_documents_chain(
        llm, prompt
    )

    retrieval_chain = create_retrieval_chain(retriever, combine_docs_chain)
            
    result = retrieval_chain.invoke({"input": combined_input, "history": chat_history_text})

    result["input"] = query.query
    result["combined_input"] = combined_input
    # print("Assistant: ", result["answer"], "\n\n")
    return result

async def chatUrlStream(query, model, temperature):

    llmModels = LlmModels(model, temperature)
    llm = llmModels.model_ollama

    # chat_history_text = ""
    # combined_input = f"Input: {query}"

    # if query.history:
    #     for chat in islice(query.history, max(0, len(query.history) - 3), len(query.history)):
    #         chat_history_text += f"User: {chat.User} \nAssistant: {chat.Assistant}\n"
    #         combined_input += f" \nHistory Input: {chat.User}"

    retriever = vector_store.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={
            "k": 10,
            "score_threshold": 0.1,
        },
    )

    relevant_documents = retriever.get_relevant_documents(query)

    context = "\n".join([doc.page_content for doc in relevant_documents])

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "You are a helpful assistant that provides answers based on the context provided."),
            ("human", "Input are here: {input}.\n Use the following context to answer. \nContext: {context}")
        ]
    )

    chain = prompt | llm | StrOutputParser()

    # analysis_prompt = ChatPromptTemplate.from_template("Check the message with history")
    # composed_chain_with_lambda = (
    #     chain
    #     | (lambda input: {"output": input})
    #     | analysis_prompt
    #     | llm
    #     | StrOutputParser()
    # )
    # print(composed_chain_with_lambda.stream({"topic": query.query}))
    
    # for chunk in chain.stream({"input": combined_input, "history": chat_history_text, "context": context}):
        # chunks.append(chunk)
        # print(chunk)


    # async def event_stream():
    async for chunk in chain.astream({"input": query, "context": context}):
        content = chunk #.replace("\n", "<br>")
        print(content)
        yield f"data: {content}\n\n"
