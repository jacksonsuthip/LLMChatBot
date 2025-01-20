from langchain.utilities import SQLDatabase
from langchain_chroma import Chroma
from langchain_experimental.sql import SQLDatabaseChain
from langchain.prompts import SemanticSimilarityExampleSelector
from langchain.schema import Document
from langchain.prompts import FewShotPromptTemplate
# from langchain.chains.sql_database.prompt import PROMPT_SUFFIX, _mysql_prompt
from langchain.prompts.prompt import PromptTemplate
from app.config import COLLECTION_NAME_SQL, DATABASE_URL_LANGCHAIN, PERSIST_DIRECTORY_DB
from app.models.embedding_models import EmbeddingModels
from app.models.llm_models import LlmModels
from app.utils.db_helper import get_table_info_async

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

    llmModels = LlmModels("llama3.2:1b", 0)
    llm = llmModels.model_ollama

    query_embedding = nomicEmbeddings.embed(query)

    example_selector = SemanticSimilarityExampleSelector(
        vectorstore=vector_store,
        k=2,
    )

    # selector = example_selector.select_examples(query=query)

    print(example_selector)
    
    # db = SQLDatabase.from_uri(DATABASE_URL_LANGCHAIN)

    # db_chain = SQLDatabaseChain.from_llm(llm, db, verbose=True)

    # result = await db_chain(query)
    
    return "example_selector"


postgresql_prompt = """You are a PostgreSQL expert. Given an input question, first create a syntactically correct PostgreSQL query to run, 
then look at the results of the query and return the answer to the input question.
Unless the user specifies in the question a specific number of examples to obtain, query for at most {top_k} results using the LIMIT clause as per PostgreSQL.
You can order the results to return the most informative data in the database.
Never query for all columns from a table. You must query only the columns that are needed to answer the question. Wrap each column name in double quotes (") to denote them as delimited identifiers.
Pay attention to use only the column names you can see in the tables below. Be careful to not query for columns that do not exist. Also, pay attention to which column is in which table.
Pay attention to use CURRENT_DATE function to get the current date, if the question involves "today".

Use the following format:

Question: Question here
SQLQuery: Query to run with no pre-amble
SQLResult: Result of the SQLQuery
Answer: Final answer here

No pre-amble.
"""


async def query_sql(query):
    
    llmModels = LlmModels("llama3.2:1b", 0)
    llm = llmModels.model_ollama

    db = SQLDatabase.from_uri(DATABASE_URL_LANGCHAIN)
    table_info = get_table_info_async()

    print(table_info)

    top_k = 2

    similar_queries = vector_store.similarity_search(query, k=top_k)
    
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
        prefix=postgresql_prompt,  # Use the PostgreSQL-specific prompt
        suffix="END OF SQL OUTPUT",
        input_variables=["input", "table_info", "top_k"],
    )

    db_chain = SQLDatabaseChain.from_llm(llm, db, verbose=True, prompt=few_shot_prompt)

    # result = await db_chain(query)

    # generated_sql = await chain.run({
    #     "input": query,
    #     "table_info": "users (id INT, name VARCHAR, age INT)",  # Example table info, adjust as needed
    #     "top_k": top_k
    # })

    print(db)

    return "db"
