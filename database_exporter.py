#!/usr/bin/env python3
"""
PostgreSQL Database Exporter Utility
Connects to PostgreSQL, lists databases and tables, and exports data to CSV files
"""

import os
import csv
import json
import logging
from typing import List, Dict, Any
from pathlib import Path
import psycopg2
from psycopg2.extras import RealDictCursor
from dotenv import load_dotenv

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv(override=True)

class DatabaseExporter:
    """Handles PostgreSQL database listing and data export"""
    
    def __init__(self):
        self.connection_string = os.getenv('AZURE_POSTGRES_CONNECTION_STRING')
        if not self.connection_string:
            raise ValueError("AZURE_POSTGRES_CONNECTION_STRING not found in environment")
        
        # Parse connection string
        self.connection_params = self._parse_connection_string(self.connection_string)
        
        # Create export directory
        self.export_dir = Path("exported_data")
        self.export_dir.mkdir(exist_ok=True)
    
    def _parse_connection_string(self, conn_string: str) -> Dict:
        """Parse PostgreSQL connection string"""
        params = {}
        parts = conn_string.split(';')
        
        for part in parts:
            if '=' in part:
                key, value = part.split('=', 1)
                key = key.strip().lower()
                
                if key == 'host':
                    params['host'] = value
                elif key == 'database':
                    params['database'] = value
                elif key == 'username':
                    params['user'] = value
                elif key == 'password':
                    params['password'] = value
                elif key == 'port':
                    params['port'] = int(value)
        
        # Set defaults
        params.setdefault('port', 5432)
        params.setdefault('database', 'postgres')
        
        return params
    
    def get_connection(self, database_name: str = None):
        """Get database connection"""
        conn_params = self.connection_params.copy()
        if database_name:
            conn_params['database'] = database_name
        return psycopg2.connect(**conn_params)
    
    def list_databases(self) -> List[str]:
        """List all accessible databases"""
        try:
            with self.get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute("""
                        SELECT datname FROM pg_database 
                        WHERE datistemplate = false 
                        AND datname NOT IN ('postgres', 'template0', 'template1')
                        ORDER BY datname
                    """)
                    databases = [row[0] for row in cur.fetchall()]
            
            # Add the default database from connection string
            if self.connection_params['database'] not in databases:
                databases.append(self.connection_params['database'])
            
            return sorted(databases)
        except Exception as e:
            logger.error(f"Error listing databases: {e}")
            # Return the default database if listing fails
            return [self.connection_params['database']]
    
    def list_tables(self, database_name: str) -> List[Dict[str, Any]]:
        """List all tables in a database with row counts"""
        try:
            with self.get_connection(database_name) as conn:
                with conn.cursor(cursor_factory=RealDictCursor) as cur:
                    # Get table information
                    cur.execute("""
                        SELECT 
                            schemaname,
                            tablename,
                            tableowner
                        FROM pg_tables 
                        WHERE schemaname NOT IN ('information_schema', 'pg_catalog')
                        ORDER BY schemaname, tablename
                    """)
                    tables = cur.fetchall()
                    
                    # Get row counts for each table
                    table_info = []
                    for table in tables:
                        try:
                            cur.execute(f"SELECT COUNT(*) FROM {table['schemaname']}.{table['tablename']}")
                            row_count = cur.fetchone()[0]
                            
                            table_info.append({
                                'schema': table['schemaname'],
                                'table': table['tablename'],
                                'owner': table['tableowner'],
                                'row_count': row_count,
                                'full_name': f"{table['schemaname']}.{table['tablename']}"
                            })
                        except Exception as e:
                            logger.warning(f"Could not get row count for {table['schemaname']}.{table['tablename']}: {e}")
                            table_info.append({
                                'schema': table['schemaname'],
                                'table': table['tablename'],
                                'owner': table['tableowner'],
                                'row_count': 'N/A',
                                'full_name': f"{table['schemaname']}.{table['tablename']}"
                            })
            
            return table_info
        except Exception as e:
            logger.error(f"Error listing tables: {e}")
            return []
    
    def export_table_to_csv(self, database_name: str, schema_name: str, table_name: str, limit: int = None) -> str:
        """Export a single table to CSV"""
        try:
            with self.get_connection(database_name) as conn:
                with conn.cursor(cursor_factory=RealDictCursor) as cur:
                    # Build query
                    query = f"SELECT * FROM {schema_name}.{table_name}"
                    if limit:
                        query += f" LIMIT {limit}"
                    
                    cur.execute(query)
                    rows = cur.fetchall()
                    
                    if not rows:
                        logger.warning(f"Table {schema_name}.{table_name} is empty")
                        return None
                    
                    # Create filename
                    filename = f"{database_name}_{schema_name}_{table_name}.csv"
                    filepath = self.export_dir / filename
                    
                    # Write to CSV
                    with open(filepath, 'w', newline='', encoding='utf-8') as csvfile:
                        fieldnames = rows[0].keys()
                        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                        
                        writer.writeheader()
                        for row in rows:
                            # Handle special data types
                            processed_row = {}
                            for key, value in row.items():
                                if isinstance(value, (dict, list)):
                                    processed_row[key] = json.dumps(value)
                                elif value is None:
                                    processed_row[key] = ''
                                else:
                                    processed_row[key] = str(value)
                            writer.writerow(processed_row)
                    
                    logger.info(f"Exported {len(rows)} rows to {filepath}")
                    return str(filepath)
                    
        except Exception as e:
            logger.error(f"Error exporting table {schema_name}.{table_name}: {e}")
            return None
    
    def export_database(self, database_name: str, selected_tables: List[str] = None, limit_per_table: int = None) -> List[str]:
        """Export all or selected tables from a database"""
        exported_files = []
        
        # Get table list
        tables = self.list_tables(database_name)
        
        if not tables:
            logger.error(f"No tables found in database: {database_name}")
            return []
        
        # Filter tables if selection provided
        if selected_tables:
            tables = [t for t in tables if t['full_name'] in selected_tables or t['table'] in selected_tables]
        
        # Export each table
        for table_info in tables:
            logger.info(f"Exporting table: {table_info['full_name']} ({table_info['row_count']} rows)")
            
            filepath = self.export_table_to_csv(
                database_name, 
                table_info['schema'], 
                table_info['table'],
                limit_per_table
            )
            
            if filepath:
                exported_files.append(filepath)
        
        return exported_files
    
    def clear_table_data(self, database_name: str, schema_name: str, table_name: str) -> bool:
        """Clear all data from a table (TRUNCATE)"""
        try:
            with self.get_connection(database_name) as conn:
                with conn.cursor() as cur:
                    # Use TRUNCATE for better performance and to reset sequences
                    cur.execute(f"TRUNCATE TABLE {schema_name}.{table_name} RESTART IDENTITY CASCADE")
                conn.commit()
                logger.info(f"Cleared data from table: {schema_name}.{table_name}")
                return True
        except Exception as e:
            logger.error(f"Error clearing table {schema_name}.{table_name}: {e}")
            return False
    
    def drop_table(self, database_name: str, schema_name: str, table_name: str) -> bool:
        """Drop a table completely"""
        try:
            with self.get_connection(database_name) as conn:
                with conn.cursor() as cur:
                    cur.execute(f"DROP TABLE IF EXISTS {schema_name}.{table_name} CASCADE")
                conn.commit()
                logger.info(f"Dropped table: {schema_name}.{table_name}")
                return True
        except Exception as e:
            logger.error(f"Error dropping table {schema_name}.{table_name}: {e}")
            return False
    
    def clear_selected_tables(self, database_name: str, selected_tables: List[str], operation: str) -> Dict[str, bool]:
        """Clear data or drop selected tables"""
        results = {}
        
        # Get table list to get schema info
        all_tables = self.list_tables(database_name)
        table_lookup = {t['full_name']: t for t in all_tables}
        table_lookup.update({t['table']: t for t in all_tables})  # Also allow table name only
        
        for table_identifier in selected_tables:
            if table_identifier in table_lookup:
                table_info = table_lookup[table_identifier]
                schema_name = table_info['schema']
                table_name = table_info['table']
                
                if operation == 'clear':
                    success = self.clear_table_data(database_name, schema_name, table_name)
                elif operation == 'drop':
                    success = self.drop_table(database_name, schema_name, table_name)
                else:
                    success = False
                    logger.error(f"Unknown operation: {operation}")
                
                results[f"{schema_name}.{table_name}"] = success
            else:
                logger.error(f"Table not found: {table_identifier}")
                results[table_identifier] = False
        
        return results

def display_databases(exporter: DatabaseExporter):
    """Display available databases"""
    databases = exporter.list_databases()
    
    print("\n" + "="*60)
    print("AVAILABLE DATABASES")
    print("="*60)
    
    if not databases:
        print("No databases found!")
        return None
    
    for i, db in enumerate(databases, 1):
        print(f"{i}. {db}")
    
    return databases

def display_tables(exporter: DatabaseExporter, database_name: str):
    """Display tables in selected database"""
    tables = exporter.list_tables(database_name)
    
    print(f"\n" + "="*80)
    print(f"TABLES IN DATABASE: {database_name}")
    print("="*80)
    
    if not tables:
        print("No tables found!")
        return None
    
    print(f"{'#':<3} {'Schema':<15} {'Table':<25} {'Rows':<10} {'Owner':<15}")
    print("-" * 80)
    
    for i, table in enumerate(tables, 1):
        print(f"{i:<3} {table['schema']:<15} {table['table']:<25} {table['row_count']:<10} {table['owner']:<15}")
    
    return tables

def get_user_choice(prompt: str, max_value: int) -> int:
    """Get user choice with validation"""
    while True:
        try:
            choice = input(f"\n{prompt} (1-{max_value}, or 0 to exit): ")
            
            if choice == '0':
                print("Exiting...")
                return 0
            
            choice = int(choice)
            if 1 <= choice <= max_value:
                return choice
            else:
                print(f"Please enter a number between 1 and {max_value}")
        except ValueError:
            print("Please enter a valid number")
        except KeyboardInterrupt:
            print("\nExiting...")
            return 0

def get_table_selection(tables: List[Dict], operation: str = 'export') -> List[str]:
    """Get user selection of tables for the specified operation"""
    operation_verb = {
        'export': 'export',
        'clear': 'clear data from',
        'drop': 'drop'
    }.get(operation, 'process')
    
    print(f"\nTable Selection Options:")
    print(f"1. {operation_verb.upper()} ALL tables")
    print(f"2. {operation_verb.upper()} SELECTED tables")
    
    while True:
        try:
            choice = input("\nChoose option (1-2): ")
            if choice == '1':
                return [t['full_name'] for t in tables]
            elif choice == '2':
                break
            else:
                print("Please enter 1 or 2")
        except KeyboardInterrupt:
            print("\nExiting...")
            return []
    
    # Selected tables mode
    selected = []
    print(f"\nSelect tables to {operation_verb} (enter table numbers, separated by commas):")
    print("Example: 1,3,5 or press Enter for all tables")
    
    selection = input("Your selection: ").strip()
    
    if not selection:
        return [t['full_name'] for t in tables]
    
    try:
        indices = [int(x.strip()) for x in selection.split(',')]
        for idx in indices:
            if 1 <= idx <= len(tables):
                selected.append(tables[idx-1]['full_name'])
            else:
                print(f"Warning: Table index {idx} is invalid, skipping...")
        
        return selected
    except ValueError:
        print(f"Invalid selection format, selecting all tables...")
        return [t['full_name'] for t in tables]

def get_row_limit() -> int:
    """Get optional row limit for export"""
    print(f"\nRow Limit Options:")
    print("1. Export ALL rows")
    print("2. Limit rows per table")
    
    while True:
        try:
            choice = input("\nChoose option (1-2): ")
            if choice == '1':
                return None
            elif choice == '2':
                limit = input("Enter row limit per table: ")
                return int(limit) if limit.strip() else None
            else:
                print("Please enter 1 or 2")
        except ValueError:
            print("Invalid number, using no limit...")
            return None
        except KeyboardInterrupt:
            print("\nExiting...")
            return None

def get_operation_choice() -> str:
    """Get user choice for operation type"""
    print(f"\n" + "="*60)
    print("DATABASE OPERATIONS")
    print("="*60)
    print("1. Export tables to CSV")
    print("2. Clear table data (TRUNCATE - removes all rows but keeps table structure)")
    print("3. Drop tables (DELETE - removes tables completely)")
    
    while True:
        try:
            choice = input("\nSelect operation (1-3, or 0 to exit): ")
            
            if choice == '0':
                return 'exit'
            elif choice == '1':
                return 'export'
            elif choice == '2':
                return 'clear'
            elif choice == '3':
                return 'drop'
            else:
                print("Please enter 1, 2, or 3")
        except KeyboardInterrupt:
            return 'exit'

def confirm_destructive_operation(operation: str, database: str, tables: List[str]) -> bool:
    """Get confirmation for destructive operations"""
    if operation == 'clear':
        op_desc = "CLEAR ALL DATA from"
        warning = "⚠️  This will permanently delete all rows from the selected tables!"
    elif operation == 'drop':
        op_desc = "DROP (DELETE COMPLETELY)"
        warning = "⚠️  This will permanently delete the tables and all their data!"
    else:
        return True
    
    print(f"\n" + "⚠️ "*20)
    print("DESTRUCTIVE OPERATION WARNING")
    print("⚠️ "*20)
    print(f"\nYou are about to {op_desc} the following tables:")
    print(f"Database: {database}")
    print(f"Tables: {len(tables)} selected")
    for table in tables[:10]:  # Show first 10
        print(f"  • {table}")
    if len(tables) > 10:
        print(f"  ... and {len(tables) - 10} more tables")
    
    print(f"\n{warning}")
    print("This action CANNOT be undone!")
    
    # Double confirmation
    confirm1 = input(f"\nAre you absolutely sure? Type 'yes' to proceed: ")
    if confirm1.lower() != 'yes':
        return False
    
    confirm2 = input(f"Final confirmation - type 'DELETE' to proceed: ")
    if confirm2 != 'DELETE':
        return False
    
    return True

def main():
    """Main interactive function"""
    try:
        print("PostgreSQL Database Manager")
        print("="*40)
        
        # Initialize exporter
        exporter = DatabaseExporter()
        
        # Step 1: Display and select database
        databases = display_databases(exporter)
        if not databases:
            return
        
        db_choice = get_user_choice("Select database", len(databases))
        if db_choice == 0:
            return
        
        selected_database = databases[db_choice - 1]
        
        # Step 2: Display tables in selected database
        tables = display_tables(exporter, selected_database)
        if not tables:
            return
        
        # Step 3: Get operation choice
        operation = get_operation_choice()
        if operation == 'exit':
            return
        
        # Step 4: Get table selection
        selected_tables = get_table_selection(tables, operation)
        if not selected_tables:
            return
        
        # Step 5: Handle based on operation
        if operation == 'export':
            # Get row limit for export
            row_limit = get_row_limit()
            
            print(f"\n" + "="*60)
            print("STARTING EXPORT")
            print("="*60)
            print(f"Database: {selected_database}")
            print(f"Tables: {len(selected_tables)} selected")
            print(f"Row limit: {'No limit' if row_limit is None else f'{row_limit} rows per table'}")
            print(f"Export directory: {exporter.export_dir}")
            
            confirm = input(f"\nProceed with export? (y/n): ")
            if confirm.lower() not in ['y', 'yes']:
                print("Export cancelled")
                return
            
            # Perform export
            exported_files = exporter.export_database(selected_database, selected_tables, row_limit)
            
            # Summary
            print(f"\n" + "="*60)
            print("EXPORT COMPLETE")
            print("="*60)
            print(f"Exported {len(exported_files)} files:")
            for file in exported_files:
                if os.path.exists(file):
                    file_size = os.path.getsize(file) / 1024  # KB
                    print(f"  • {os.path.basename(file)} ({file_size:.1f} KB)")
            
            print(f"\nAll files saved to: {exporter.export_dir.absolute()}")
        
        elif operation in ['clear', 'drop']:
            # Confirm destructive operation
            if not confirm_destructive_operation(operation, selected_database, selected_tables):
                print("Operation cancelled")
                return
            
            print(f"\n" + "="*60)
            print(f"STARTING {operation.upper()} OPERATION")
            print("="*60)
            
            # Perform operation
            results = exporter.clear_selected_tables(selected_database, selected_tables, operation)
            
            # Summary
            print(f"\n" + "="*60)
            print(f"{operation.upper()} OPERATION COMPLETE")
            print("="*60)
            
            success_count = sum(results.values())
            total_count = len(results)
            
            print(f"Successfully processed: {success_count}/{total_count} tables")
            
            if success_count < total_count:
                print("\nFailed operations:")
                for table, success in results.items():
                    if not success:
                        print(f"  ❌ {table}")
            
            if success_count > 0:
                print("\nSuccessful operations:")
                for table, success in results.items():
                    if success:
                        print(f"  ✅ {table}")
        
    except KeyboardInterrupt:
        print("\n\nOperation interrupted by user")
    except Exception as e:
        logger.error(f"Operation failed: {e}")
        print(f"Error: {e}")

if __name__ == "__main__":
    main()