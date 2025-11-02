"""
Improved Confidence Calculation for RAG System
==============================================

Key improvements:
1. Semantic similarity between query and retrieved docs
2. Better source quality scoring
3. Adaptive answer completeness
4. Citation density (not just count)
5. Query complexity awareness
"""

from typing import List
from langchain_core.documents import Document
import numpy as np


class ImprovedConfidenceCalculator:
    """Enhanced confidence scoring for RAG responses"""

    def __init__(self, embeddings):
        """Initialize with embedding model for semantic similarity"""
        self.embeddings = embeddings

    def calculate_confidence(
        self,
        query: str,
        answer: str,
        retrieved_docs: List[Document],
        used_docs: List[Document],
    ) -> float:
        """
        Calculate confidence score based on multiple factors

        Args:
            query: Original user query
            answer: Generated answer
            retrieved_docs: All documents retrieved from vector store
            used_docs: Documents actually used in generation (top-k after reranking)

        Returns:
            Confidence score between 0.0 and 1.0
        """
        if not used_docs or not answer:
            return 0.0

        # Calculate individual factors
        semantic_score = self._calculate_semantic_similarity(query, used_docs)
        source_quality = self._calculate_source_quality(retrieved_docs, used_docs)
        answer_quality = self._calculate_answer_quality(answer, query)
        citation_quality = self._calculate_citation_quality(answer, used_docs)

        # Weighted combination
        # Semantic similarity is most important for confidence
        confidence = (
            semantic_score * 0.35  # How well docs match query
            + source_quality * 0.25  # Quality/quantity of sources
            + answer_quality * 0.25  # Answer completeness/coherence
            + citation_quality * 0.15  # Proper citation usage
        )

        return round(min(confidence, 1.0), 2)

    def _calculate_semantic_similarity(self, query: str, docs: List[Document]) -> float:
        """
        Calculate semantic similarity between query and retrieved documents
        This is the most important factor for confidence
        """
        if not docs:
            return 0.0

        try:
            # Get query embedding
            query_embedding = self.embeddings.embed_query(query)

            # Get document embeddings
            doc_texts = [doc.page_content for doc in docs]
            doc_embeddings = self.embeddings.embed_documents(doc_texts)

            # Calculate cosine similarities
            similarities = []
            for doc_emb in doc_embeddings:
                # Cosine similarity
                similarity = np.dot(query_embedding, doc_emb) / (
                    np.linalg.norm(query_embedding) * np.linalg.norm(doc_emb)
                )
                similarities.append(similarity)

            # Use weighted average (give more weight to top documents)
            weights = [1.0 / (i + 1) for i in range(len(similarities))]
            weighted_sim = sum(s * w for s, w in zip(similarities, weights)) / sum(
                weights
            )

            # Normalize to 0-1 range (cosine similarity is typically 0.7-0.95 for relevant docs)
            # Map 0.6 → 0.0 and 0.9 → 1.0
            normalized = (weighted_sim - 0.6) / 0.3
            return max(0.0, min(1.0, normalized))

        except Exception as e:
            # Fallback to keyword overlap if embedding fails
            return self._fallback_similarity(query, docs)

    def _fallback_similarity(self, query: str, docs: List[Document]) -> float:
        """Fallback to keyword-based similarity if embeddings fail"""
        query_terms = set(query.lower().split())

        similarities = []
        for doc in docs:
            doc_terms = set(doc.page_content.lower().split())
            if not query_terms:
                continue
            overlap = len(query_terms & doc_terms)
            similarity = overlap / len(query_terms)
            similarities.append(similarity)

        if not similarities:
            return 0.0

        # Average of top 3 docs
        top_3 = sorted(similarities, reverse=True)[:3]
        return min(sum(top_3) / len(top_3), 1.0)

    def _calculate_source_quality(
        self, retrieved_docs: List[Document], used_docs: List[Document]
    ) -> float:
        """
        Calculate source quality based on:
        - Number of sources used
        - Proportion of retrieved docs that were useful
        - Diversity of sources
        """
        if not used_docs:
            return 0.0

        # Factor 1: Number of sources (more is better, up to 5)
        num_sources = len(used_docs)
        source_count_score = min(num_sources / 5, 1.0)

        # Factor 2: Retrieval precision (what % of retrieved docs were actually used)
        if retrieved_docs:
            precision = len(used_docs) / len(retrieved_docs)
        else:
            precision = 0.0

        # Factor 3: Source diversity (unique documents)
        unique_sources = len(
            set(doc.metadata.get("source_file", "") for doc in used_docs)
        )
        diversity_score = min(unique_sources / 3, 1.0)

        # Combine
        quality = source_count_score * 0.4 + precision * 0.3 + diversity_score * 0.3

        return quality

    def _calculate_answer_quality(self, answer: str, query: str) -> float:
        """
        Calculate answer quality based on:
        - Appropriate length (not too short or too long)
        - Coherence indicators (complete sentences, etc.)
        - Query term coverage
        """
        if not answer:
            return 0.0

        # Factor 1: Length appropriateness (adaptive based on query)
        # Short factual queries expect short answers
        # Complex queries expect longer answers
        query_words = len(query.split())
        expected_min_length = max(
            50, query_words * 10
        )  # At least 10 words per query word
        expected_max_length = query_words * 100  # Up to 100 words per query word

        answer_length = len(answer)

        if answer_length < expected_min_length:
            length_score = answer_length / expected_min_length
        elif answer_length > expected_max_length:
            length_score = 0.9  # Slightly penalize overly verbose answers
        else:
            length_score = 1.0

        # Factor 2: Coherence (has complete sentences, proper structure)
        coherence_score = 1.0
        if not answer.strip().endswith((".", "!", "?", "]")):
            coherence_score *= 0.9  # Incomplete sentence
        if answer.count(".") == 0 and len(answer) > 50:
            coherence_score *= 0.8  # No sentence breaks in long answer

        # Factor 3: Query term coverage
        query_terms = set(query.lower().split())
        answer_terms = set(answer.lower().split())
        coverage = (
            len(query_terms & answer_terms) / len(query_terms) if query_terms else 0
        )
        coverage_score = min(coverage * 1.5, 1.0)  # Boost coverage importance

        quality = length_score * 0.3 + coherence_score * 0.3 + coverage_score * 0.4

        return quality

    def _calculate_citation_quality(self, answer: str, docs: List[Document]) -> float:
        """
        Calculate citation quality based on:
        - Presence of citations
        - Citation density (appropriate amount)
        - All sources cited
        """
        if not docs:
            return 0.0

        # Count citations in answer
        citation_count = sum(1 for i in range(1, len(docs) + 1) if f"[{i}]" in answer)

        # Factor 1: Citation presence (at least one citation)
        presence_score = 1.0 if citation_count > 0 else 0.0

        # Factor 2: Citation coverage (what % of sources are cited)
        coverage = citation_count / len(docs) if docs else 0
        coverage_score = min(coverage * 1.2, 1.0)  # Slightly boost

        # Factor 3: Citation density (not too sparse, not too dense)
        # Ideal: 1 citation per 100-200 characters
        answer_length = len(answer)
        ideal_density = answer_length / 150
        actual_density = citation_count

        if ideal_density == 0:
            density_score = 1.0
        else:
            density_ratio = actual_density / ideal_density
            # Penalize if too few or too many citations
            if 0.5 <= density_ratio <= 2.0:
                density_score = 1.0
            elif density_ratio < 0.5:
                density_score = density_ratio / 0.5
            else:
                density_score = 2.0 / density_ratio

        quality = presence_score * 0.4 + coverage_score * 0.4 + density_score * 0.2

        return quality
