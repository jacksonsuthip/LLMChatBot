from sqlalchemy import create_engine, MetaData
from sqlalchemy.engine import reflection
import asyncpg
import asyncio

async def get_table_info_async(DATABASE_URL):
    # Using asyncpg to fetch the table info directly from PostgreSQL
    conn = await asyncpg.connect(DATABASE_URL)
    query = """
    SELECT table_name, column_name, data_type
    FROM information_schema.columns
    WHERE table_schema = 'public';
    """
    rows = await conn.fetch(query)
    await conn.close()

    # Format the results into a dictionary
    table_info = {}
    for row in rows:
        table_name = row["table_name"]
        if table_name not in table_info:
            table_info[table_name] = []
        table_info[table_name].append({
            "name": row["column_name"],
            "type": row["data_type"]
        })
    
    return table_info
