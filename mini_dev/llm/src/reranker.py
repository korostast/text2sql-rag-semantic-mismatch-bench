import httpx


class RerankerClient:
    def __init__(
        self,
        reranker_url: str = "https://api.openai.com/v1",
        reranker_model: str = "bge-reranker-v2",
        reranker_api_key: str = "",
    ):
        self.reranker_url = reranker_url.rstrip("/")
        self.reranker_model = reranker_model
        self.reranker_api_key = reranker_api_key

    def rerank(
        self, query: str, documents: list[str], top_k: int = None
    ) -> list[dict]:
        """
        Rerank documents based on their relevance to the query
        """
        try:
            payload = {
                "model": self.reranker_model,
                "query": query,
                "documents": documents,
            }
            if top_k is not None:
                payload["top_n"] = top_k

            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.reranker_api_key}",
            }
            response = httpx.post(
                f"{self.reranker_url}/rerank",
                json=payload,
                headers=headers,
                timeout=30.0,
            )
            response.raise_for_status()
            data = response.json()

            return data.get("results", [])

        except httpx.HTTPStatusError as e:
            print(f"HTTP error in reranking: {e.response.status_code} - {e.response.text}")
            return []
        except Exception as e:
            print(f"Error in reranking: {e}")
            return []

def rerank_search_results(
    query: str,
    search_results: list[dict],
    k_ret: int,
    score_threshold: float,
    reranker_url: str,
    reranker_model: str,
    reranker_api_key: str,
) -> list[dict]:
    """
    Rerank search results using BGE-reranker-v2 model
    """
    if not search_results:
        return []

    documents = [result.get("question", "") for result in search_results]

    reranker = RerankerClient(
        reranker_url=reranker_url,
        reranker_model=reranker_model,
        reranker_api_key=reranker_api_key,
    )

    reranked = reranker.rerank(query=query, documents=documents, top_k=None)
    if not reranked:
        print("Reranking failed, returning empty results")
        return []

    # Filter results by score threshold
    filtered_results = []
    for item in reranked:
        relevance_score = item.get("relevance_score", 0.0)
        if relevance_score >= score_threshold:
            original_index = item.get("index")
            if original_index is not None and original_index < len(search_results):
                original_result = search_results[original_index].copy()
                original_result["reranker_score"] = relevance_score
                filtered_results.append(original_result)

    filtered_results.sort(key=lambda x: x["reranker_score"], reverse=True)
    return filtered_results[:k_ret]
