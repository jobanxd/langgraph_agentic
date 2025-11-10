import time
import logging
from typing import Dict, Any
from langchain_core.tools import tool
from database.database import db

logger = logging.getLogger(__name__)

@tool
def execute_query(sql_query: str) -> Dict[str, Any]:
    """
    Execute a SQL query against the insurance database.
    
    Args:
        sql_query: The SQL query string to execute
        
    Returns:
        Dictionary containing query results, execution time, and status
    """
    try:
        start_time = time.time()
        results = db.execute_query(sql_query)
        execution_time = (time.time() - start_time) * 1000
        
        return {
            "query_successful": True,
            "data": results,
            "record_count": len(results),
            "execution_time_ms": round(execution_time, 2),
            "error": None
        }
    except Exception as e:
        logger.error("SQL query execution error: %s", e)
        logger.error("Query: %s", sql_query)
        return {
            "query_successful": False,
            "data": [],
            "record_count": 0,
            "execution_time_ms": 0,
            "error": f"Query execution failed: {str(e)}"
        }