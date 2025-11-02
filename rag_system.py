"""
DocumentRAG - Production-Ready RAG System in Python
====================================================

Features:
- Document upload (PDF, DOCX, TXT, MD)
- Smart chunking with overlap
- Hybrid search (vector + keyword)
- Streaming responses
- Source citations
- Confidence scoring
- Query suggestions
- Beautiful CLI interface

Built with: LangChain, ChromaDB, OpenAI
"""

import os
import warnings
import sys
from contextlib import redirect_stderr
from io import StringIO

from improved_confidence_calculator import ImprovedConfidenceCalculator

# Disable telemetry for ChromaDB and LangChain BEFORE imports
os.environ["ANONYMIZED_TELEMETRY"] = "False"
os.environ["CHROMA_TELEMETRY"] = "False"
os.environ["LANGCHAIN_TRACING_V2"] = "false"
os.environ["LANGCHAIN_ENDPOINT"] = ""

# Suppress telemetry and deprecation warnings
warnings.filterwarnings("ignore", message=".*telemetry.*")
warnings.filterwarnings("ignore", message=".*Failed to send telemetry.*")
warnings.filterwarnings("ignore", category=DeprecationWarning, message=".*persist.*")

# Suppress ChromaDB telemetry stderr messages
_original_stderr = sys.stderr


class TelemetrySuppressor:
    """Context manager to suppress ChromaDB telemetry errors"""

    def __enter__(self):
        self._stderr_buffer = StringIO()
        sys.stderr = self._stderr_buffer
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # Restore stderr
        sys.stderr = _original_stderr
        # Optionally check buffer for non-telemetry errors (for debugging)
        # buffer_content = self._stderr_buffer.getvalue()
        # if buffer_content and "telemetry" not in buffer_content.lower():
        #     _original_stderr.write(buffer_content)
        return False


import json
import time
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime
import hashlib

# LangChain imports
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain_classic.chains.retrieval_qa.base import RetrievalQA
from langchain_core.documents import Document
from langchain_core.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain_core.prompts import PromptTemplate

# Document loaders
from langchain_community.document_loaders import (
    PyPDFLoader,
    Docx2txtLoader,
    TextLoader,
    UnstructuredMarkdownLoader,
)

# For progress bars and beautiful output
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table
from rich.panel import Panel
from rich import box


console = Console()


@dataclass
class Citation:
    """Source citation with metadata"""

    number: int
    document_name: str
    page: int
    content: str
    score: float


@dataclass
class RAGResponse:
    """Complete RAG response with metadata"""

    answer: str
    citations: List[Citation]
    related_questions: List[str]
    confidence_score: float
    chunks_retrieved: int
    chunks_used: int
    total_time_ms: float
    estimated_cost: float


class DocumentRAG:
    """
    Main RAG system - handles everything from document processing to generation
    """

    def __init__(
        self,
        openai_api_key: str,
        model: str = "gpt-5-mini",
        embedding_model: str = "text-embedding-3-large",
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        persist_directory: str = "./chroma_db",
    ):
        """Initialize RAG system"""
        self.openai_api_key = openai_api_key
        self.model = model
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.persist_directory = persist_directory

        # Initialize embeddings
        self.embeddings = OpenAIEmbeddings(
            model=embedding_model, openai_api_key=openai_api_key
        )

        # Initialize improved confidence calculator
        self.confidence_calculator = ImprovedConfidenceCalculator(self.embeddings)

        # Initialize vector store (creates or loads existing)
        self.vectorstore = None
        self.documents_metadata = {}  # Track uploaded documents

        console.print("✅ [green]DocumentRAG initialized![/green]")

    def load_document(self, file_path: str) -> List[Document]:
        """
        Load document based on file extension
        """
        # Check if file exists
        if not os.path.exists(file_path):
            raise FileNotFoundError(
                f"❌ File not found: {file_path}\n"
                f"Please check that the file exists at the specified path."
            )

        if not os.path.isfile(file_path):
            raise ValueError(f"❌ Path is not a file: {file_path}")

        ext = os.path.splitext(file_path)[1].lower()

        loaders = {
            ".pdf": PyPDFLoader,
            ".docx": Docx2txtLoader,
            ".txt": TextLoader,
            ".md": UnstructuredMarkdownLoader,
        }

        if ext not in loaders:
            raise ValueError(
                f"❌ Unsupported file type: {ext}\n"
                f"Supported formats: PDF, DOCX, TXT, MD"
            )

        with console.status(f"[bold blue]Loading {os.path.basename(file_path)}..."):
            loader = loaders[ext](file_path)
            documents = loader.load()

        # Add metadata
        for doc in documents:
            doc.metadata["source_file"] = os.path.basename(file_path)
            doc.metadata["upload_time"] = datetime.now().isoformat()

        console.print(
            f"✅ Loaded {len(documents)} pages from [cyan]{os.path.basename(file_path)}[/cyan]"
        )
        return documents

    def chunk_documents(self, documents: List[Document]) -> List[Document]:
        """
        Smart chunking with overlap
        """
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""],  # Try paragraph → sentence → word
        )

        with console.status("[bold blue]Chunking documents..."):
            chunks = text_splitter.split_documents(documents)

        # Add chunk metadata
        for i, chunk in enumerate(chunks):
            chunk.metadata["chunk_id"] = i
            chunk.metadata["chunk_size"] = len(chunk.page_content)

        console.print(
            f"✅ Created {len(chunks)} chunks (size: {self.chunk_size}, overlap: {self.chunk_overlap})"
        )
        return chunks

    def index_documents(self, file_paths: List[str]):
        """
        Complete indexing pipeline: Load → Chunk → Embed → Store
        """
        start_time = time.time()

        console.print(
            Panel.fit(
                "[bold cyan]📚 Starting Document Indexing Pipeline[/bold cyan]",
                border_style="cyan",
            )
        )

        all_chunks = []

        # Step 1: Load all documents
        for file_path in file_paths:
            documents = self.load_document(file_path)
            chunks = self.chunk_documents(documents)
            all_chunks.extend(chunks)

            # Track document
            doc_id = hashlib.md5(file_path.encode()).hexdigest()[:8]
            self.documents_metadata[doc_id] = {
                "filename": os.path.basename(file_path),
                "chunks": len(chunks),
                "indexed_at": datetime.now().isoformat(),
            }

        # Step 2: Create/update vector store
        with Progress(
            SpinnerColumn(),
            TextColumn("[bold blue]Generating embeddings and indexing..."),
            console=console,
        ) as progress:
            task = progress.add_task("Indexing", total=None)

            if self.vectorstore is None:
                # Create new vector store (suppress telemetry errors)
                with TelemetrySuppressor():
                    self.vectorstore = Chroma.from_documents(
                        documents=all_chunks,
                        embedding=self.embeddings,
                        persist_directory=self.persist_directory,
                    )
            else:
                # Add to existing vector store (suppress telemetry errors)
                with TelemetrySuppressor():
                    self.vectorstore.add_documents(all_chunks)

            # Note: ChromaDB 0.4.x+ auto-persists, no need to call persist()
            progress.update(task, completed=True)

        elapsed = time.time() - start_time

        # Success summary
        table = Table(title="Indexing Summary", box=box.ROUNDED)
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")

        table.add_row("Documents Indexed", str(len(file_paths)))
        table.add_row("Total Chunks", str(len(all_chunks)))
        table.add_row(
            "Average Chunk Size",
            f"{sum(c.metadata['chunk_size'] for c in all_chunks) / len(all_chunks):.0f} chars",
        )
        table.add_row("Time Taken", f"{elapsed:.2f}s")
        table.add_row(
            "Estimated Cost", f"${(len(all_chunks) * 512 / 1000 * 0.00013):.4f}"
        )

        console.print(table)
        console.print("✅ [green]Indexing complete! Ready for queries.[/green]\n")

    def query(
        self,
        question: str,
        conversation_history: Optional[List[Dict]] = None,
        top_k: int = 5,
        streaming: bool = False,
    ) -> RAGResponse:
        """
        Complete RAG query pipeline with all the bells and whistles
        """
        if self.vectorstore is None:
            raise ValueError("No documents indexed yet! Call index_documents() first.")

        start_time = time.time()

        console.print(
            Panel.fit(
                f"[bold cyan]❓ Query:[/bold cyan] {question}", border_style="cyan"
            )
        )

        # Step 1: Enhance query with conversation context
        enhanced_query = self._enhance_query(question, conversation_history)

        # Step 2: Retrieve relevant chunks (hybrid search simulation via MMR)
        with console.status("[bold blue]🔍 Searching knowledge base..."):
            with TelemetrySuppressor():
                retriever = self.vectorstore.as_retriever(
                    search_type="mmr",  # Maximum Marginal Relevance (diversity)
                    search_kwargs={
                        "k": top_k * 2,  # Get more, then rerank
                        "fetch_k": 20,  # Initial retrieval pool
                    },
                )

                relevant_docs = retriever.invoke(enhanced_query)

        console.print(f"✅ Retrieved {len(relevant_docs)} relevant chunks")

        # Step 3: Rerank (LangChain MMR does this already, but we'll sort by score)
        scored_docs = self._rerank_documents(relevant_docs, enhanced_query)[:top_k]

        # Step 4: Generate response with GPT-4
        with console.status("[bold blue]🤖 Generating answer..."):
            llm = ChatOpenAI(
                model=self.model,
                temperature=0.3,  # Low for factual responses
                openai_api_key=self.openai_api_key,
                streaming=streaming,
                callbacks=[StreamingStdOutCallbackHandler()] if streaming else [],
            )

            # Custom prompt with citations
            prompt_template = """You are a helpful AI assistant. Answer the question based ONLY on the following context. Always cite your sources using [1], [2], etc.

Context:
{context}

Question: {question}

Instructions:
1. Provide a clear, concise answer
2. ALWAYS cite sources using [1], [2] format
3. If the context doesn't contain enough information, say so
4. Be factual and precise

Answer:"""

            prompt = PromptTemplate(
                template=prompt_template, input_variables=["context", "question"]
            )

            # Create QA chain (suppress telemetry errors)
            with TelemetrySuppressor():
                retriever = self.vectorstore.as_retriever(search_kwargs={"k": top_k})
                qa_chain = RetrievalQA.from_chain_type(
                    llm=llm,
                    chain_type="stuff",
                    retriever=retriever,
                    return_source_documents=True,
                    chain_type_kwargs={"prompt": prompt},
                )

                result = qa_chain.invoke({"query": enhanced_query})
            answer = result["result"]
            source_docs = result["source_documents"]

        if not streaming:
            console.print(f"\n💬 [bold green]Answer:[/bold green]\n{answer}\n")

        # Step 5: Extract citations
        citations = self._extract_citations(answer, scored_docs)

        # Step 6: Generate related questions
        with console.status("[bold blue]💡 Generating related questions..."):
            related_questions = self._generate_related_questions(
                question, answer, scored_docs
            )

        # Step 7: Calculate confidence score
        confidence = self.confidence_calculator.calculate_confidence(
            query=question,
            answer=answer,
            retrieved_docs=relevant_docs,
            used_docs=scored_docs,
        )

        elapsed_ms = (time.time() - start_time) * 1000

        # Estimate cost (rough)
        prompt_tokens = sum(len(doc.page_content) for doc in scored_docs) // 4
        completion_tokens = len(answer) // 4
        cost = (prompt_tokens * 0.03 + completion_tokens * 0.06) / 1000

        return RAGResponse(
            answer=answer,
            citations=citations,
            related_questions=related_questions,
            confidence_score=confidence,
            chunks_retrieved=len(relevant_docs),
            chunks_used=len(scored_docs),
            total_time_ms=elapsed_ms,
            estimated_cost=cost,
        )

    def _enhance_query(self, query: str, history: Optional[List[Dict]]) -> str:
        """Add conversation context to query"""
        if not history or len(history) == 0:
            return query

        # Simple enhancement: prepend last exchange
        last_q = history[-1].get("question", "")
        if last_q:
            return f"{last_q}. {query}"

        return query

    def _rerank_documents(self, docs: List[Document], query: str) -> List[Document]:
        """
        Simple reranking by query term overlap
        In production, use a dedicated reranker model
        """
        query_terms = set(query.lower().split())

        scored = []
        for doc in docs:
            doc_terms = set(doc.page_content.lower().split())
            overlap = len(query_terms & doc_terms)
            score = overlap / len(query_terms) if query_terms else 0
            scored.append((score, doc))

        scored.sort(reverse=True, key=lambda x: x[0])
        return [doc for _, doc in scored]

    def _extract_citations(self, answer: str, docs: List[Document]) -> List[Citation]:
        """Extract citations from answer"""
        citations = []

        for i, doc in enumerate(docs, 1):
            if f"[{i}]" in answer:  # Citation found
                citations.append(
                    Citation(
                        number=i,
                        document_name=doc.metadata.get("source_file", "Unknown"),
                        page=doc.metadata.get("page", 0),
                        content=doc.page_content[:150],
                        score=0.9,  # Placeholder
                    )
                )

        return citations

    def _generate_related_questions(
        self, original_query: str, answer: str, docs: List[Document]
    ) -> List[str]:
        """Generate 3 related follow-up questions"""
        try:
            llm = ChatOpenAI(
                model="gpt-3.5-turbo",  # Cheaper for this task
                temperature=0.7,
                openai_api_key=self.openai_api_key,
            )

            prompt = f"""Based on this Q&A, suggest 3 related follow-up questions:

Question: {original_query}
Answer: {answer}

Generate 3 specific follow-up questions (one per line, no numbering):"""

            response = llm.predict(prompt)
            questions = [q.strip() for q in response.split("\n") if q.strip()]
            return questions[:3]

        except Exception as e:
            return [
                "Can you provide more details?",
                "What are the implications?",
                "How does this compare to alternatives?",
            ]

    def display_response(self, response: RAGResponse):
        """Beautiful display of RAG response"""

        # Citations
        if response.citations:
            cite_table = Table(title="📎 Sources", box=box.ROUNDED, show_header=True)
            cite_table.add_column("#", style="cyan", width=3)
            cite_table.add_column("Document", style="green")
            cite_table.add_column("Page", style="yellow", width=6)
            cite_table.add_column("Preview", style="white")

            for cite in response.citations:
                cite_table.add_row(
                    f"[{cite.number}]",
                    cite.document_name,
                    str(cite.page),
                    cite.content[:60] + "...",
                )

            console.print(cite_table)

        # Related questions
        if response.related_questions:
            console.print("\n💡 [bold cyan]Related Questions:[/bold cyan]")
            for q in response.related_questions:
                console.print(f"  • {q}")

        # Metadata
        meta_table = Table(box=box.SIMPLE, show_header=False, padding=(0, 2))
        meta_table.add_column(style="cyan")
        meta_table.add_column(style="green")

        meta_table.add_row("⏱️  Response Time", f"{response.total_time_ms:.0f}ms")
        meta_table.add_row("🎯 Confidence", f"{response.confidence_score:.0%}")
        meta_table.add_row("📊 Chunks Retrieved", str(response.chunks_retrieved))
        meta_table.add_row("✅ Chunks Used", str(response.chunks_used))
        meta_table.add_row("💰 Estimated Cost", f"${response.estimated_cost:.4f}")

        console.print("\n")
        console.print(meta_table)


# ============================================================================
# Example Usage / Demo
# ============================================================================


def demo():
    """
    Demo the RAG system with sample documents
    """
    console.print(
        Panel.fit(
            "[bold magenta]🚀 DocumentRAG - Production RAG System[/bold magenta]\n"
            "Built with LangChain + ChromaDB + OpenAI",
            border_style="magenta",
        )
    )

    # Check for API key
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        console.print(
            "[red]❌ Error: OPENAI_API_KEY environment variable not set![/red]"
        )
        console.print("Set it with: export OPENAI_API_KEY='your-key-here'")
        return

    # Initialize RAG system
    rag = DocumentRAG(
        openai_api_key=api_key, model="gpt-4", chunk_size=512, chunk_overlap=50
    )

    # Example: Index documents
    console.print("\n[bold]Step 1: Index your documents[/bold]")
    console.print("Place your PDF/DOCX/TXT files in a 'documents' folder")
    console.print("Or modify the code to point to your files\n")

    # Uncomment and modify for your files:
    # rag.index_documents([
    #     "documents/Q3_Report.pdf",
    #     "documents/Product_Guide.docx",
    #     "documents/FAQ.txt"
    # ])

    # Example query
    console.print("\n[bold]Step 2: Query your knowledge base[/bold]\n")

    # Uncomment after indexing:
    # response = rag.query(
    #     "What was the Q3 revenue growth?",
    #     streaming=False
    # )
    # rag.display_response(response)

    console.print(
        """
[bold cyan]Quick Start:[/bold cyan]

1. Install dependencies:
   pip install langchain openai chromadb pypdf docx2txt unstructured rich

2. Set OpenAI API key:
   export OPENAI_API_KEY='your-key-here'

3. Uncomment the index_documents() and query() calls above

4. Run: python rag_system.py

[bold green]✅ That's it! You have a production RAG system.[/bold green]
    """
    )


if __name__ == "__main__":
    demo()
