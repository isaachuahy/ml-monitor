# eval/db_utils.py - Utility functions for database operations
# Used to get a database connection to avoid repeating code for evaluation scripts.
# Also in case we need pooled connections, we can use this utility function to get a connection.

import os
import psycopg2

def get_db_conn():
    return psycopg2.connect(os.getenv("DATABASE_URL"))