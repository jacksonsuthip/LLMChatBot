from langchain.utilities import SQLDatabase
from langchain_experimental.sql import SQLDatabaseChain
from app.config import DATABASE_URL

async def query_sql(query: str):
    print(DATABASE_URL)
    db = SQLDatabase.from_uri(DATABASE_URL)

    db_chain = SQLDatabaseChain(llm=None, database=db)

    result = await db_chain.invoke(query)
    
    return result  # Return the result of the query