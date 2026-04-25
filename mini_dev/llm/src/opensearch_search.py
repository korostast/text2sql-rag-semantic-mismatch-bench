from typing import List, Dict
from opensearchpy import OpenSearch
from openai import OpenAI as OpenAIClient


class OpenSearchClient:
    def __init__(
        self,
        opensearch_url: str = "http://localhost:9200",
        index_name: str = "bird_training_data",
        embedder_url: str = "https://api.openai.com/v1",
        embedder_model: str = "bge-m3",
        embedder_api_key: str = ""
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
            retry_on_timeout=True
        )
        
        self.embedding_client = OpenAIClient(
            api_key=embedder_api_key,
            base_url=embedder_url
        )
        
        self._create_rrf_pipeline()
    
    def _create_rrf_pipeline(self):
        """Create RRF search pipeline for hybrid search."""
        pipeline_name = "rrf-pipeline"
        
        try:
            if self.client.transport.perform_request(
                "PUT",
                f"/_search/pipeline/{pipeline_name}"
            ):
                print(f"RRF pipeline '{pipeline_name}' already exists")
                return
        except:
            pass  # Pipeline doesn't exist, create it
        
        pipeline_body = {
            "description": "Post processor for hybrid RRF search",
            "phase_results_processors": [
                {
                    "score-ranker-processor": {
                        "combination": {
                            "technique": "rrf"
                        }
                    }
                }
            ]
        }
        
        try:
            self.client.transport.perform_request(
                "PUT",
                f"/_search/pipeline/{pipeline_name}",
                body=pipeline_body
            )
        except Exception as e:
            print(f"Error creating RRF pipeline: {e}")
    
    def generate_embedding(self, text: str) -> List[float]:
        """
        Generate embedding for a text using the embedding model
        """
        try:
            response = self.embedding_client.embeddings.create(
                model=self.embedder_model,
                input=text
            )
            return response.data[0].embedding
        except Exception as e:
            print(f"Error generating embedding: {e}")
            raise
    
    def search_similar_examples(
        self,
        question: str,
        k: int = 5,
        operator: str = "or"
    ) -> List[Dict]:
        """
        Search for k most similar examples
        """
        query_embedding = self.generate_embedding(question)
        try:
            response = self.client.search(
                index=self.index_name,
                body={
                    "size": k,
                    "query": {
                        "hybrid": {
                            "queries": [
                                {
                                    "match": {
                                        "question": {
                                            "query": question,
                                            "operator": operator
                                        }
                                    }
                                },
                                {
                                    "knn": {
                                        "question_embedding": {
                                            "vector": query_embedding,
                                            "k": k
                                        }
                                    }
                                }
                            ]
                        }
                    }
                },
                params={"search_pipeline": "rrf-pipeline"}
            )
            
            results = []
            for hit in response['hits']['hits']:
                source = hit['_source']
                results.append({
                    'question': source.get('question', ''),
                    'evidence': source.get('evidence', ''),
                    'sql': source.get('sql', ''),
                    'db_id': source.get('db_id', ''),
                    'question_id': source.get('question_id', ''),
                    'difficulty': source.get('difficulty', ''),
                    'score': hit['_score']
                })
            
            return results
        except Exception as e:
            print(f"Error in hybrid search: {e}")
            return []


def search_similar_examples(
    question: str,
    k: int = 5,
    operator: str = "or",
    opensearch_url: str = "http://localhost:9200",
    index_name: str = "bird_training_data",
    embedder_url: str = "https://api.openai.com/v1",
    embedder_model: str = "bge-m3",
    embedder_api_key: str = ""
) -> List[Dict]:
    searcher = OpenSearchClient(
        opensearch_url=opensearch_url,
        index_name=index_name,
        embedder_url=embedder_url,
        embedder_model=embedder_model,
        embedder_api_key=embedder_api_key
    )
    return searcher.search_similar_examples(question, k=k, operator=operator)
