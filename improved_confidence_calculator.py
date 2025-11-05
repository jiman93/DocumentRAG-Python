"""
Improved Confidence Calculation for RAG System - FIXED VERSION
==============================================================

Key fixes:
1. Lower normalization threshold (0.5 → 0.0 instead of 0.6 → 0.0)
2. Adjusted weights to be less aggressive on semantic similarity
3. Better fallback handling
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

        # ✨ ADJUSTED WEIGHTS: Less aggressive on semantic, more on answer quality
        confidence = (
            semantic_score * 0.25  # Reduced from 35%
            + source_quality * 0.25  # Same
            + answer_quality * 0.35  # Increased from 25%
            + citation_quality * 0.15  # Same
        )

        return round(min(confidence, 1.0), 2)

    def _calculate_semantic_similarity(self, query: str, docs: List[Document]) -> float:
        """
        Calculate semantic similarity between query and retrieved documents
        """
        if not docs or not query:
            return 0.5  # Neutral score if no query

        try:
            # Get query embedding
            query_embedding = self.embeddings.embed_query(query)

            # Get document embeddings (first 5 docs only for performance)
            doc_texts = [doc.page_content[:500] for doc in docs[:5]]
            doc_embeddings = self.embeddings.embed_documents(doc_texts)

            # Calculate cosine similarities
            similarities = []
            for doc_emb in doc_embeddings:
                similarity = np.dot(query_embedding, doc_emb) / (
                    np.linalg.norm(query_embedding) * np.linalg.norm(doc_emb)
                )
                similarities.append(similarity)

            # Use weighted average (give more weight to top documents)
            weights = [1.0 / (i + 1) for i in range(len(similarities))]
            weighted_sim = sum(s * w for s, w in zip(similarities, weights)) / sum(
                weights
            )

            # 🔧 FIXED NORMALIZATION: Lower threshold
            # OLD: 0.6 → 0.0, 0.9 → 1.0 (too aggressive!)
            # NEW: 0.5 → 0.0, 0.85 → 1.0 (more forgiving)
            normalized = (weighted_sim - 0.5) / 0.35

            # Ensure we don't go below 0.2 for decent matches
            # This prevents good answers from getting too low confidence
            normalized = max(0.2, min(1.0, normalized))

            return normalized

        except Exception as e:
            # Fallback to keyword overlap if embedding fails
            return self._fallback_similarity(query, docs)

    def _fallback_similarity(self, query: str, docs: List[Document]) -> float:
        """Fallback to keyword-based similarity if embeddings fail"""
        query_terms = set(query.lower().split())

        similarities = []
        for doc in docs[:5]:
            doc_terms = set(doc.page_content.lower().split())
            if query_terms:
                overlap = len(query_terms & doc_terms)
                similarity = overlap / len(query_terms)
                similarities.append(similarity)

        if not similarities:
            return 0.5  # Neutral score

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
            precision = 0.5

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
        query_words = len(query.split()) if query else 5

        # 🔧 ADJUSTED: More lenient length expectations
        expected_min = max(30, query_words * 5)  # Reduced from 50 and 10x
        expected_max = query_words * 150  # Increased from 100x

        answer_length = len(answer)

        if answer_length < expected_min:
            length_score = min(answer_length / expected_min, 0.8)  # Cap penalty at 0.8
        elif answer_length > expected_max:
            length_score = 0.95  # Very small penalty for verbosity
        else:
            length_score = 1.0

        # Factor 2: Coherence (has complete sentences, proper structure)
        coherence_score = 1.0
        if not answer.strip().endswith((".", "!", "?", "]")):
            coherence_score *= 0.95  # Reduced from 0.9
        if answer.count(".") == 0 and len(answer) > 50:
            coherence_score *= 0.9  # Reduced from 0.8

        # Factor 3: Query term coverage
        if query:
            query_terms = set(query.lower().split())
            answer_terms = set(answer.lower().split())
            coverage = (
                len(query_terms & answer_terms) / len(query_terms)
                if query_terms
                else 0.5
            )
            coverage_score = min(coverage * 1.3, 1.0)  # Reduced boost from 1.5
        else:
            coverage_score = 0.7

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
        presence_score = 1.0 if citation_count > 0 else 0.3  # Changed from 0.0 to 0.3

        # Factor 2: Citation coverage (what % of sources are cited)
        coverage = citation_count / len(docs) if docs else 0
        coverage_score = min(coverage * 1.2, 1.0)

        # Factor 3: Citation density (not too sparse, not too dense)
        answer_length = len(answer)
        ideal_density = answer_length / 150
        actual_density = citation_count

        if ideal_density == 0:
            density_score = 1.0
        else:
            density_ratio = actual_density / ideal_density
            # More forgiving density check
            if 0.3 <= density_ratio <= 3.0:  # Widened from 0.5-2.0
                density_score = 1.0
            elif density_ratio < 0.3:
                density_score = max(density_ratio / 0.3, 0.5)  # Floor at 0.5
            else:
                density_score = max(3.0 / density_ratio, 0.5)  # Floor at 0.5

        quality = presence_score * 0.4 + coverage_score * 0.4 + density_score * 0.2

        return quality
