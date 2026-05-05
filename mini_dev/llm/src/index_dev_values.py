import argparse
import json
import sqlite3
import time
from concurrent.futures import ThreadPoolExecutor

from openai import OpenAI as OpenAIClient
from opensearchpy import OpenSearch
from tqdm import tqdm


class DevValueIndexer:
    def __init__(
        self,
        opensearch_url: str = "http://localhost:9200",
        index_name: str = "bird_dev_values",
        embedder_url: str = "https://api.openai.com/v1",
        embedder_model: str = "bge-m3",
        embedder_api_key: str = "",
    ):
        self.opensearch_url = opensearch_url
        self.index_name = index_name
        self.embedder_url = embedder_url
        self.embedder_model = embedder_model
        self.embedder_api_key = embedder_api_key

        self.client = OpenSearch(
            hosts=[opensearch_url],
            http_compress=True,
            timeout=30,
            max_retries=3,
            retry_on_timeout=True,
        )

        self.embedding_client = OpenAIClient(api_key=embedder_api_key, base_url=embedder_url)

    def create_index(self):
        """Create the OpenSearch index with appropriate mappings"""
        if self.client.indices.exists(index=self.index_name):
            print(f"Deleting existing index '{self.index_name}'...")
            self.client.indices.delete(index=self.index_name)

        index_body = {
            "settings": {
                "index": {
                    "knn": True,
                }
            },
            "mappings": {
                "properties": {
                    "value": {"type": "text"},
                    "value_embedding": {
                        "type": "knn_vector",
                        "dimension": 1024,
                    },
                    "db_id": {"type": "keyword"},
                    "table_name": {"type": "keyword"},
                    "column_name": {"type": "keyword"},
                }
            },
        }

        try:
            self.client.indices.create(index=self.index_name, body=index_body)
            print(f"Created index '{self.index_name}'")
        except Exception as e:
            print(f"Error creating index: {e}")
            raise

    def generate_embedding(self, text: str) -> list[float]:
        """
        Generate embedding for a text using the embedding model
        """
        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = self.embedding_client.embeddings.create(
                    model=self.embedder_model, input=text
                )
                return response.data[0].embedding
            except Exception as e:
                if attempt < max_retries - 1:
                    wait_time = 2**attempt
                    print(f"Error generating embedding (attempt {attempt + 1}/{max_retries}): {e}")
                    print(f"Retrying in {wait_time} seconds...")
                    time.sleep(wait_time)
                else:
                    print(f"Failed to generate embedding after {max_retries} attempts: {e}")
                    raise

    def generate_embeddings_batch(self, texts: list[str]) -> list[list[float]]:
        """
        Generate embeddings for a batch of texts using the embedding model
        """
        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = self.embedding_client.embeddings.create(
                    model=self.embedder_model, input=texts
                )
                return [item.embedding for item in response.data]
            except Exception as e:
                if attempt < max_retries - 1:
                    wait_time = 2**attempt
                    print(
                        f"Error generating embeddings batch (attempt {attempt + 1}/{max_retries}): {e}"
                    )
                    print(f"Retrying in {wait_time} seconds...")
                    time.sleep(wait_time)
                else:
                    print(f"Failed to generate embeddings batch after {max_retries} attempts: {e}")
                    raise

    def load_dev_tables(self, dev_tables_path: str) -> list[dict]:
        """
        Load dev database table schemas from JSON file
        """
        with open(dev_tables_path, encoding="utf-8") as f:
            data = json.load(f)

        return data

    def get_text_columns(self, db_schema: dict) -> list[tuple[str, str]]:
        """
        Identify text-based columns that should be indexed
        Returns list of (table_name, column_name) tuples
        """
        text_columns = []

        for table_idx, table_name in enumerate(db_schema.get("table_names_original", [])):
            column_types = db_schema.get("column_types", [])
            column_names = db_schema.get("column_names_original", [])

            table_columns = []
            for col_idx, col_info in enumerate(column_names):
                if col_info[0] == table_idx:
                    col_name = col_info[1]
                    col_type = column_types[col_idx] if col_idx < len(column_types) else "text"

                    # Skip columns that are likely IDs or numeric
                    if col_type == "text" and not any(
                        keyword in col_name.lower()
                        for keyword in ["id", "date", "time", "number", "count", "amount", "price"]
                    ):
                        table_columns.append((table_name, col_name))

            text_columns.extend(table_columns)

        return text_columns

    def extract_unique_values(self, db_path: str, table_name: str, column_name: str) -> set[str]:
        """
        Extract unique values from a specific table and column
        """
        try:
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()

            # SQLite reserved keywords that need escaping
            reserved_keywords = {
                "order",
                "group",
                "by",
                "select",
                "from",
                "where",
                "insert",
                "update",
                "delete",
                "create",
                "drop",
                "alter",
                "table",
                "index",
                "join",
                "on",
                "and",
                "or",
                "not",
                "in",
                "is",
                "null",
                "like",
                "between",
                "having",
                "limit",
                "offset",
                "union",
                "distinct",
                "as",
                "asc",
                "desc",
                "case",
                "when",
                "then",
                "else",
                "end",
            }

            # Escape identifiers that contain spaces, special characters, or are reserved keywords
            # Use double quotes for SQLite compatibility
            def escape_identifier(identifier: str) -> str:
                needs_escaping = (
                    " " in identifier
                    or "-" in identifier
                    or identifier.lower() in reserved_keywords
                    or not identifier.replace("_", "").isalnum()
                )
                return f'"{identifier}"' if needs_escaping else identifier

            safe_table_name = escape_identifier(table_name)
            safe_column_name = escape_identifier(column_name)

            cursor.execute(f"SELECT DISTINCT {safe_column_name} FROM {safe_table_name}")
            values = cursor.fetchall()

            unique_values = set()
            for value_tuple in values:
                value = value_tuple[0]
                if value is not None and str(value).strip():
                    unique_values.add(str(value).strip())

            conn.close()
            return unique_values

        except Exception as e:
            print(f"Error extracting values from {db_path}.{table_name}.{column_name}: {e}")
            return set()

    def index_database_values(self, db_root_path: str, db_schema: dict, batch_size: int = 1024):
        """
        Index values from a single database
        """
        db_id = db_schema.get("db_id", "")
        db_path = f"{db_root_path}{db_id}/{db_id}.sqlite"
        print(f"\nProcessing database: {db_id}")
        text_columns = self.get_text_columns(db_schema)
        if not text_columns:
            print(f"No text columns found for {db_id}")
            return

        print(f"Found {len(text_columns)} text columns to index")
        for table_name, column_name in text_columns:
            print(f"  Extracting values from {table_name}.{column_name}...")

            unique_values = self.extract_unique_values(db_path, table_name, column_name)
            unique_values = [v[:10000] for v in unique_values]

            if not unique_values:
                print("    No values found")
                continue

            print(f"    Found {len(unique_values)} unique values")
            self._index_values_batch(
                db_id=db_id,
                table_name=table_name,
                column_name=column_name,
                values=list(unique_values),
                batch_size=batch_size,
            )

    def _index_values_batch(
        self,
        db_id: str,
        table_name: str,
        column_name: str,
        values: list[str],
        batch_size: int = 1024,
    ):
        """
        Index a batch of values to OpenSearch
        """

        def process_batch(batch):
            try:
                embeddings = self.generate_embeddings_batch(batch)
                documents = []
                for value, embedding in zip(batch, embeddings):
                    doc = {
                        "value": value,
                        "value_embedding": embedding,
                        "db_id": db_id,
                        "table_name": table_name,
                        "column_name": column_name,
                    }
                    documents.append(doc)

                if documents:
                    self._index(documents)

            except Exception as e:
                print(f"Error processing batch of {len(batch)} values: {e}")
                # Fallback to processing one by one if batch fails
                for value in batch:
                    try:
                        embedding = self.generate_embedding(value)
                        doc = {
                            "value": value,
                            "value_embedding": embedding,
                            "db_id": db_id,
                            "table_name": table_name,
                            "column_name": column_name,
                        }
                        self._index([doc])
                    except Exception as single_error:
                        print(f"Error processing value '{value}': {single_error}")
                        continue

        # Use 2 threads for parallel embedding generation and indexing
        with ThreadPoolExecutor(max_workers=8) as executor:
            batches = [values[i : i + batch_size] for i in range(0, len(values), batch_size)]
            list(
                tqdm(
                    executor.map(process_batch, batches),
                    total=len(batches),
                    desc="Process batches of values...",
                )
            )

    def _index(self, documents: list[dict]):
        """
        Batch index documents to OpenSearch
        """
        body = []
        for doc in documents:
            body.append({"index": {"_index": self.index_name}})
            body.append(doc)

        try:
            response = self.client.bulk(body=body)
            if response.get("errors"):
                print(f"Errors in bulk indexing: {response}")
        except Exception as e:
            print(f"Error in bulk indexing: {e}")

    def index_all_databases(self, dev_tables_path: str, db_root_path: str, batch_size: int = 2048):
        """
        Index values from all dev databases
        """
        db_schemas = self.load_dev_tables(dev_tables_path)
        print(f"Found {len(db_schemas)} databases to process")
        for db_schema in tqdm(db_schemas, desc="Indexing databases"):
            self.index_database_values(db_root_path, db_schema, batch_size)


def main():
    parser = argparse.ArgumentParser(
        description="Index dev database values with embeddings into OpenSearch"
    )
    parser.add_argument(
        "--dev_tables_path", type=str, required=True, help="Path to dev_tables.json file"
    )
    parser.add_argument(
        "--db_root_path", type=str, required=True, help="Root path to dev databases directory"
    )
    parser.add_argument(
        "--opensearch_url",
        type=str,
        default="http://localhost:9200",
        help="OpenSearch endpoint URL",
    )
    parser.add_argument(
        "--index_name", type=str, default="bird_dev_values", help="Name of the OpenSearch index"
    )
    parser.add_argument(
        "--embedder_url",
        type=str,
        default="https://api.openai.com/v1",
        help="Embedding model API URL",
    )
    parser.add_argument(
        "--embedder_model", type=str, default="bge-m3", help="Name of the embedding model"
    )
    parser.add_argument(
        "--embedder_api_key", type=str, required=True, help="API key for embedding service"
    )
    parser.add_argument("--batch_size", type=int, default=512, help="Batch size for indexing")

    args = parser.parse_args()

    indexer = DevValueIndexer(
        opensearch_url=args.opensearch_url,
        index_name=args.index_name,
        embedder_url=args.embedder_url,
        embedder_model=args.embedder_model,
        embedder_api_key=args.embedder_api_key,
    )

    indexer.create_index()
    indexer.index_all_databases(
        dev_tables_path=args.dev_tables_path,
        db_root_path=args.db_root_path,
        batch_size=args.batch_size,
    )
    print("\nIndexing completed successfully!")


if __name__ == "__main__":
    main()
