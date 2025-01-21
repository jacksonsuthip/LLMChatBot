from sqlalchemy import create_engine, MetaData
from sqlalchemy.engine import reflection
import asyncpg

async def get_table_info_1(DATABASE_URL):
    # Using asyncpg to fetch the table info directly from PostgreSQL
    conn = await asyncpg.connect(DATABASE_URL)
    query = """
    SELECT table_name, column_name, data_type
    FROM information_schema.columns
    WHERE table_schema = 'public';
    """
    rows = await conn.fetch(query)
    await conn.close()

    # Build the string to return
    table_info_str = ""
    
    # Iterate over rows to create a formatted string
    for row in rows:
        table_name = row["table_name"]
        column_name = row["column_name"]
        data_type = row["data_type"]
        
        # Format each table's information
        if table_info_str == "" or not table_info_str.endswith(f"\n{table_name}"):
            table_info_str += f"\n\nTable: {table_name}\n"
        
        table_info_str += f"  - {column_name} ({data_type})\n"
    
    return table_info_str.strip()


async def get_table_info(DATABASE_URL):
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
