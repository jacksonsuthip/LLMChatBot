from langchain.utilities import SQLDatabase
from langchain_chroma import Chroma
from langchain_experimental.sql import SQLDatabaseChain
from langchain.prompts import SemanticSimilarityExampleSelector
from langchain.schema import Document
from langchain.prompts import FewShotPromptTemplate
from langchain.chains.sql_database.prompt import PROMPT_SUFFIX, _postgres_prompt
from langchain.prompts.prompt import PromptTemplate
from app.config import COLLECTION_NAME_SQL, DATABASE_URL_LANGCHAIN, GROQ_CLOUD_API, PERSIST_DIRECTORY_DB
from app.models.embedding_models import EmbeddingModels
from app.models.llm_models import LlmModels, LlmModelsGroq
from app.utils.db_helper import get_table_info

embeddingModels = EmbeddingModels()
# HuggingFaceEmbeddings = embeddingModels.HuggingFaceEmbeddings_sentence_transformers
nomicEmbeddings = embeddingModels.nomic_embeddings_ollama

vector_store = Chroma(
    collection_name=COLLECTION_NAME_SQL,
    embedding_function=nomicEmbeddings,
    persist_directory=PERSIST_DIRECTORY_DB,
)

async def save_query_sql(saveQuery):

    documents = []
    for item in saveQuery:
        # Use dot notation to access model attributes
        doc_text = f"Question: {item.Question} | SQL Query: {item.SQLQuery} | SQL Result: {item.SQLResult} | Answer: {item.Answer}"
        
        # Convert the item to a dictionary to store as metadata
        doc = Document(page_content=doc_text, metadata=item.dict())  # Convert Pydantic model to dict
        documents.append(doc)

    # Add the documents to the Chroma vector store
    saved_ids = vector_store.add_documents(documents)

    # Persist the vector store
    # vector_store.persist()

    return saved_ids

async def query_sql_1(query: str):

    llmModels = LlmModelsGroq("gemma2-9b-it", 0.1, GROQ_CLOUD_API)
    llm = llmModels.model_Groq
    
    db = SQLDatabase.from_uri(DATABASE_URL_LANGCHAIN)

    db_chain = SQLDatabaseChain.from_llm(llm, db, verbose=True)

    result = await db_chain.arun(query)
    
    return result

async def query_sql(query):
    llmModels = LlmModels("llama3:8b", 9)
    llm = llmModels.model_ollama

    db = SQLDatabase.from_uri(DATABASE_URL_LANGCHAIN)
    
    # table_info = await get_table_info(DATABASE_URL_LANGCHAIN)
    # top_k = 2

    similar_queries = vector_store.similarity_search(query, k=2)
    
    examples = []
    for item in similar_queries:
        question = item.metadata['Question']
        sql_query = item.metadata['SQLQuery']
        sql_result = item.metadata['SQLResult']
        answer = item.metadata['Answer']

        examples.append({
            "Question": question,
            "SQLQuery": sql_query,
            "SQLResult": sql_result,
            "Answer": answer
        })

    example_prompt = PromptTemplate(
        input_variables=["Question", "SQLQuery", "SQLResult", "Answer"],
        template="\nQuestion: {Question}\nSQLQuery: {SQLQuery}\nSQLResult: {SQLResult}\nAnswer: {Answer}",
    )

    few_shot_prompt = FewShotPromptTemplate(
        examples=examples,
        example_prompt=example_prompt,
        prefix=_postgres_prompt,
        suffix=PROMPT_SUFFIX,
        input_variables=["input", "table_info", "top_k"],
    )

    db_chain = SQLDatabaseChain.from_llm(llm, db, verbose=True, prompt=few_shot_prompt)

    # result = await db_chain.acall({"query": query, "table_info": table_info})
    # result = await db_chain.acall(query)
    result = await db_chain.ainvoke(query)

    return result, few_shot_prompt

async def delete_query_sql():
    try:
        # vector_store.delete_collection()
        vector_store.reset_collection()
        
        return {"message": "All records successfully deleted from the collection.", "collection_name": COLLECTION_NAME_SQL}
    
    except Exception as e:
        return {"message": f"Error deleting records: {str(e)}", "collection_name": COLLECTION_NAME_SQL}
    
async def get_db_query_sql_endpoint(count):
    try:
        results = vector_store.get()
        
        # metadata_list = [result.metadata for result in results]

        return results

    except Exception as e:
        print(f"An error occurred while fetching metadata: {e}")
        return None