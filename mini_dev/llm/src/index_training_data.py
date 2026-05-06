import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor

from openai import OpenAI as OpenAIClient
from opensearchpy import OpenSearch
from tqdm import tqdm


class TrainingDataIndexer:
    def __init__(
        self,
        opensearch_url: str = "http://localhost:9200",
        index_name: str = "bird_training_data",
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
                    "question": {"type": "text", "analyzer": "standard"},
                    "question_embedding": {
                        "type": "knn_vector",
                        "dimension": 1024,  # TODO extract to run_gpt.sh variable?
                    },
                    "evidence": {"type": "text"},
                    "sql": {"type": "text"},
                    "db_id": {"type": "keyword"},
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

    def load_training_data(self, train_path: str) -> list[dict]:
        """
        Load training data from JSON file
        """
        with open(train_path, encoding="utf-8") as f:
            data = json.load(f)

        return data

    def index_data(self, data: list[dict], batch_size: int = 10):
        """
        Index training data with embeddings
        """
        print(f"Indexing {len(data)} samples...")

        def process_batch(batch):
            try:
                batch_questions = []
                valid_examples = []
                for example in batch:
                    question = example.get("question", "")
                    if question:
                        batch_questions.append(question)
                        valid_examples.append(example)

                if not batch_questions:
                    return

                embeddings = self.generate_embeddings_batch(batch_questions)
                documents = []
                for example, embedding in zip(valid_examples, embeddings):
                    doc = {
                        "question": example.get("question", ""),
                        "question_embedding": embedding,
                        "evidence": example.get("evidence", ""),
                        "sql": example.get("SQL", ""),
                        "db_id": example.get("db_id", ""),
                    }
                    documents.append(doc)

                if documents:
                    self._index(documents)

            except Exception as e:
                print(f"Error processing batch of {len(batch)} samples: {e}")
                for example in batch:
                    question = example.get("question", "")
                    if not question:
                        continue
                    try:
                        embedding = self.generate_embedding(question)
                        doc = {
                            "question": question,
                            "question_embedding": embedding,
                            "evidence": example.get("evidence", ""),
                            "sql": example.get("SQL", ""),
                            "db_id": example.get("db_id", ""),
                        }
                        self._index([doc])
                    except Exception as single_error:
                        print(f"Error processing example {question}: {single_error}")
                        continue

        with ThreadPoolExecutor(max_workers=8) as executor:  # TODO hardcoded number
            batches = [data[i : i + batch_size] for i in range(0, len(data), batch_size)]
            list(tqdm(executor.map(process_batch, batches), total=len(batches), desc="Indexing"))

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


def main():
    parser = argparse.ArgumentParser(
        description="Index training data with embeddings into OpenSearch"
    )
    parser.add_argument(
        "--train_path", type=str, required=True, help="Path to training data JSON file"
    )
    parser.add_argument(
        "--opensearch_url",
        type=str,
        default="http://localhost:9200",
        help="OpenSearch endpoint URL",
    )
    parser.add_argument(
        "--index_name", type=str, default="bird_training_data", help="Name of the OpenSearch index"
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
    parser.add_argument("--batch_size", type=int, default=50, help="Batch size for indexing")

    args = parser.parse_args()

    indexer = TrainingDataIndexer(
        opensearch_url=args.opensearch_url,
        index_name=args.index_name,
        embedder_url=args.embedder_url,
        embedder_model=args.embedder_model,
        embedder_api_key=args.embedder_api_key,
    )

    indexer.create_index()
    data = indexer.load_training_data(args.train_path)
    indexer.index_data(data, batch_size=args.batch_size)
    print("\nIndexing completed successfully!")


if __name__ == "__main__":
    main()
