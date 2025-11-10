import sqlite3
import pandas as pd
import os
import argparse
import sys


def create_db(db_name: str):
    if not db_name:
        raise ValueError("Input a database name!")

    conn = sqlite3.connect(db_name)
    conn.close()
    print(f"Database '{db_name}' created successfully.")


def add_table(csv_file: str, db_name: str, table_name: str = None):
    if not os.path.exists(csv_file):
        raise FileNotFoundError(f"CSV file '{csv_file}' not found.")
    
    if not os.path.exists(db_name):
        raise FileNotFoundError(f"Database '{db_name}' not found. Please create it first!")
    
    if table_name is None:
        table_name = os.path.splitext(os.path.basename(csv_file))[0]

    df = pd.read_csv(csv_file)
    conn = sqlite3.connect(db_name)

    df.to_sql(table_name, conn, if_exists="replace", index=False)
    print(f"Table '{table_name}' added to '{db_name}' successfully.")

    # Show sample rows
    sample = conn.execute(f"SELECT * FROM {table_name} LIMIT 5;").fetchall()
    for row in sample:
        print(row)

    conn.close()


def main():
    parser = argparse.ArgumentParser(
        description="SQLite DB Utility: Create DB or add tables from CSV files."
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    # Subcommand: create-db
    create_parser = subparsers.add_parser("create-db", help="Create a new SQLite database")
    create_parser.add_argument("db_name", help="Name of the database to create")

    # Subcommand: add-table
    add_parser = subparsers.add_parser("add-table", help="Add a table to an existing database from a CSV file")
    add_parser.add_argument("csv_file", help="Path to the CSV file")
    add_parser.add_argument("db_name", help="Path to the SQLite database")
    add_parser.add_argument("--table", help="Optional table name (defaults to CSV filename)", default=None)

    args = parser.parse_args()

    if args.command == "create-db":
        create_db(args.db_name)

    elif args.command == "add-table":
        add_table(args.csv_file, args.db_name, args.table)


if __name__ == "__main__":
    main()
