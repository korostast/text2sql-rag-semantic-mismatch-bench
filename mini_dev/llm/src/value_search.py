from openai import OpenAI as OpenAIClient
from opensearchpy import OpenSearch


class ValueSearchClient:
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

        self._create_rrf_pipeline()

    def _create_rrf_pipeline(self):
        """
        Create RRF search pipeline for hybrid search
        """
        pipeline_name = "rrf-pipeline-values"

        try:
            if self.client.transport.perform_request("PUT", f"/_search/pipeline/{pipeline_name}"):
                print(f"RRF pipeline '{pipeline_name}' already exists")
                return
        except Exception:
            pass

        pipeline_body = {
            "description": "Post processor for hybrid RRF search for values",
            "phase_results_processors": [
                {"score-ranker-processor": {"combination": {"technique": "rrf"}}}
            ],
        }

        try:
            self.client.transport.perform_request(
                "PUT", f"/_search/pipeline/{pipeline_name}", body=pipeline_body
            )
        except Exception as e:
            print(f"Error creating RRF pipeline: {e}")

    def generate_embedding(self, text: str) -> list[float]:
        """
        Generate embedding for a text using the embedding model
        """
        try:
            response = self.embedding_client.embeddings.create(
                model=self.embedder_model, input=text
            )
            return response.data[0].embedding
        except Exception as e:
            print(f"Error generating embedding: {e}")
            raise

    def search_similar_values(
        self,
        value: str,
        table_name: str,
        column_name: str,
        db_id: str,
        k: int = 5,
    ) -> list[dict]:
        """
        Search for similar values in the same table and column
        """
        query_embedding = self.generate_embedding(value)

        try:
            response = self.client.search(
                index=self.index_name,
                body={
                    "size": k,
                    "query": {
                        "bool": {
                            "must": [
                                {"term": {"db_id": db_id}},
                                {"term": {"table_name": table_name}},
                                {"term": {"column_name": column_name}},
                            ],
                            "should": [
                                {"match": {"value": {"query": value, "operator": "or"}}},
                                {"knn": {"value_embedding": {"vector": query_embedding, "k": k}}},
                            ],
                        }
                    },
                },
                params={"search_pipeline": "rrf-pipeline-values"},
            )

            results = []
            for hit in response["hits"]["hits"]:
                source = hit["_source"]
                results.append(
                    {
                        "value": source.get("value", ""),
                        "db_id": source.get("db_id", ""),
                        "table_name": source.get("table_name", ""),
                        "column_name": source.get("column_name", ""),
                        "score": hit["_score"],
                    }
                )

            return results
        except Exception as e:
            print(f"Error in hybrid search: {e}")
            return []


def search_similar_values_for_pairs(
    column_value_pairs: list[dict],
    opensearch_url: str,
    index_name: str,
    embedder_url: str,
    embedder_model: str,
    embedder_api_key: str,
    k_results: int = 5,
) -> dict[str, list[dict]]:
    """
    Search similar values for multiple column-value pairs
    """
    searcher = ValueSearchClient(
        opensearch_url=opensearch_url,
        index_name=index_name,
        embedder_url=embedder_url,
        embedder_model=embedder_model,
        embedder_api_key=embedder_api_key,
    )

    results = {}

    for pair in column_value_pairs:
        table_name = pair.get("table_name", "")
        column_name = pair.get("column_name", "")
        value = pair.get("value", "")
        db_id = pair.get("db_id", "")

        if not all([table_name, column_name, value, db_id]):
            continue

        key = f"{table_name}.{column_name}.{value}"

        try:
            similar_values = searcher.search_similar_values(
                value=value,
                table_name=table_name,
                column_name=column_name,
                db_id=db_id,
                k=k_results,
            )
            results[key] = similar_values
        except Exception as e:
            print(f"Error searching for {key}: {e}")
            results[key] = []

    return results
