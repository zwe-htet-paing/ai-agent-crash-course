import numpy as np
from typing import List, Dict, Any, Optional
from tqdm.auto import tqdm
from sentence_transformers import SentenceTransformer

from logger import logger

class SearchTool:
    """
    Advanced search tool with text, vector, and hybrid search capabilities.
    """
    
    def __init__(self, index, use_vector_search: bool = False):
        """
        Initialize the search tool.
        
        Args:
            index: Text search index
            use_vector_search: Whether to enable vector/hybrid search capabilities
        """
        self.index = index
        self.use_vector_search = use_vector_search
        
        # Vector search components
        self.documents: List[Dict] = index.docs
        self.embedding_model: SentenceTransformer = None
        self.embeddings: np.ndarray = None # Normalized document vectors
        
        if use_vector_search:
            logger.info("Loading embedding model...")
            self.embedding_model = SentenceTransformer('multi-qa-distilbert-cos-v1')
            
            # Get documents from index
            self.documents = index.docs  # Assuming index has docs attribute
            
            # Create embeddings
            logger.info("Creating embeddings for vector search...")
            raw_embeddings = self._create_embeddings(self.documents)
            
            # Normalize embeddings
            norms = np.linalg.norm(raw_embeddings, axis=1, keepdims=True)
            self.embeddings = raw_embeddings / norms
    
    def _create_embeddings(self, chunks: List[Dict]) -> np.ndarray:
        """
        Create embeddings for document chunks.
        
        Args:
            chunks: List of document chunks with metadata
            
        Returns:
            numpy array of embeddings
        """
        if self.embedding_model is None:
            logger.info("Loading embedding model...")
            self.embedding_model = SentenceTransformer('multi-qa-distilbert-cos-v1')
        
        embeddings_list = []
        
        logger.info("Encoding document chunks...")
        for d in tqdm(chunks, desc="Creating embeddings",):
            # Create enhanced text for better context
            text_to_encode = d.get('chunk', d.get('section', d.get('content', '')))
            
            if 'title' in d and d['title']:
                text_to_encode = f"{d['title']}. {text_to_encode}"
            
            if 'filename' in d:
                filename_parts = d['filename'].replace('/', ' ').replace('_', ' ')
                filename_parts = filename_parts.replace('.mdx', '').replace('.md', '')
                text_to_encode = f"{filename_parts}. {text_to_encode}"
            
            # Encode text to vector
            v = self.embedding_model.encode(text_to_encode, show_progress_bar=False)
            embeddings_list.append(v)
        
        return np.array(embeddings_list)
    
    def _get_doc_key(self, doc: Dict) -> Any:
        """Helper to get a unique identifier for a document (used in RRF)."""
        # Strip potential score/search_type keys if they exist from a previous search step
        temp_doc = {k: v for k, v in doc.items() if k not in ['score', 'search_type', 'final_rrf_score']}
        return temp_doc.get('id') or hash(str(temp_doc))
    
    def index_search(self, query: str, num_results: int = 5) -> List[Any]:
        """
        Perform a text-based search on the index.
        
        Args:
            query: The search query string
            num_results: Number of results to return
            
        Returns:
            List of search results
        """
        return self.index.search(query, num_results=num_results)
    
    def vector_search(self, query: str, num_results: int = 5) -> List[Dict]:
        """
        Search for most similar documents using cosine similarity.
        
        Args:
            query_embedding: Query vector embedding
            num_results: Number of results to return
            
        Returns:
            List of documents with similarity scores
        """
        if not self.use_vector_search:
            raise RuntimeError("Vector search is not enabled.")
        
        # 1. Encode and normalize the query
        query_vector = self.embedding_model.encode(query)
        query_vector = query_vector / np.linalg.norm(query_vector)
        
        # 2. Calculate scores and get top N indices
        similarities = self.embeddings @ query_vector.T 
        top_indices = np.argsort(similarities)[::-1][:num_results]
        
        # 3. Return documents with similarity scores
        results = []
        for idx in top_indices:
            doc = self.documents[idx].copy()
            doc['score'] = float(similarities[idx])
            doc['search_type'] = 'vector'
            results.append(doc)
        
        return results
    
    def hybrid_search(self, query: str, num_results: int = 5) -> List[Dict]:
        """
        Perform a Hybrid Search combining keyword and vector results using Reciprocal Rank Fusion (RRF)
        
        Args:
            query: Search query string
            alpha: Weight for text search (1-alpha for vector search)
            num_results: Number of results to return
            
        Returns:
            List of ranked documents
        """
        if not self.use_vector_search:
            logger.info("Vector search is disabled. Falling back to Keyword Search.")
            return self.index_search(query, num_results)
        
        # Get more results from each search type to improve RRF effectiveness
        N_SEARCH = num_results * 2
        keyword_results = self.index_search(query, N_SEARCH)
        vector_results = self.vector_search(query, N_SEARCH)
        
        # RRF Constant: k=60 is a common value, higher k smooths rank differences
        RRF_K = 60
        fused_results = {} # Store {doc_id: {'score': float, 'doc': Dict}}

        # --- FUSION STEP ---
        
        # 1. Fuse scores from Keyword Search
        for rank, doc in enumerate(keyword_results):
            doc_id = self._get_doc_key(doc)
            # RRF Formula: 1 / (K + rank + 1)
            rrf_score = 1 / (RRF_K + rank + 1)
            
            if doc_id not in fused_results:
                fused_results[doc_id] = {'score': 0.0, 'doc': doc}
            
            fused_results[doc_id]['score'] += rrf_score

        # 2. Fuse scores from Vector Search
        for rank, doc in enumerate(vector_results):
            doc_id = self._get_doc_key(doc)
            rrf_score = 1 / (RRF_K + rank + 1)
            
            if doc_id not in fused_results:
                fused_results[doc_id] = {'score': 0.0, 'doc': doc}
                
            fused_results[doc_id]['score'] += rrf_score

        # --- RANKING STEP ---
        
        # 3. Sort by the final RRF score
        sorted_results = sorted(
            fused_results.values(), 
            key=lambda x: x['score'], 
            reverse=True
        )

        # 4. Compile and return the top N results
        return [
            {**res['doc'], 'final_rrf_score': res['score'], 'search_type': 'hybrid'} 
            for res in sorted_results[:num_results]
        ]
        
        
if __name__ == "__main__":
    from ingest import index_data
    
    # Example usage
    repo_owner = "pydantic"
    repo_name = "pydantic-ai"
    
    logger.info(f"Indexing data for {repo_owner}/{repo_name}...")
    index = index_data(repo_owner, repo_name, chunk_method="markdown_sections")
    
    logger.info("Initializing SearchTool with hybrid search...")
    search_tool = SearchTool(index=index, use_vector_search=True)
    
    query = "How to install a Pydantic-ai?"
    logger.info(f"Performing hybrid search for query: '{query}'")
    results = search_tool.hybrid_search(query, num_results=5)
    
    for i, res in enumerate(results, 1):
        logger.info(f"Result {i}:")
        logger.info(f"Filename: {res.get('filename', 'N/A')}")
        logger.info(f"Score: {res.get('final_rrf_score', 'N/A'):.4f}")
        logger.info(f"Content Snippet: {res.get('content', '')[:200]}...")
        logger.info("-" * 40)