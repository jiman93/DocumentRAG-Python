"""
Quick Start Example - Get Running in 2 Minutes
"""

from rag_system import DocumentRAG
import os
from dotenv import load_dotenv

# Step 1: Load OpenAI API key from environment
# Option 1: Set it in your environment: export OPENAI_API_KEY='sk-...'
# Option 2: Create a .env file with: OPENAI_API_KEY=sk-...
# Get your key from: https://platform.openai.com/api-keys
load_dotenv()  # Load from .env file if it exists

openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key or openai_api_key == "sk-your-key-here":
    raise ValueError(
        "❌ OpenAI API key not found!\n"
        "Please set your API key in one of these ways:\n"
        "1. Environment variable: export OPENAI_API_KEY='sk-...'\n"
        "2. Create a .env file with: OPENAI_API_KEY=sk-...\n"
        "Get your key from: https://platform.openai.com/api-keys"
    )

# Step 2: Initialize RAG system
print("🚀 Initializing DocumentRAG...")
rag = DocumentRAG(
    openai_api_key=openai_api_key,
    model="gpt-5-mini",  # Or "gpt-3.5-turbo" for faster/cheaper
    chunk_size=1000,
    chunk_overlap=200,
)

# Step 3: Index your documents
print("\n📚 Indexing documents...")
# Put your PDF/DOCX/TXT files here:
rag.index_documents(
    [
        "Impact-Study-of-Artificial-Intelligence-Digital-and-Green-Economy-on-the-Malaysian-Workforce-Volume-2-Sector-Global-Business-Services-1747879429.pdf",  # Replace with your file paths
        # "another_doc.docx",
        # "notes.txt"
    ]
)

# Step 4: Ask questions!
print("\n💬 Querying knowledge base...")

response = rag.query(
    question="What are the main topics covered?",  # Your question here
    top_k=5,
    streaming=False,  # Set to True for word-by-word streaming
)

# Step 5: See results
rag.display_response(response)

# Additional queries
print("\n" + "=" * 50)
print("Try more questions:\n")

test_queries = [
    # Factual (works well)
    "What is the youth unemployment rate in Malaysia?",
    "How many GBS companies are in Malaysia?",
    # Statistical (impressive)
    "What percentage of GBS roles are highly impacted by AI?",
    "How many job roles were assessed in this study?",
    # Complex (shows capability)
    "What are the main challenges for GBS organizations adopting AI?",
    "What skills are needed for a Payroll Specialist to transition to other roles?",
    # Multi-part (demonstrates reasoning)
    "What is the projected GBS revenue growth from 2022 to 2025, and what's driving it?",
]

for q in test_queries:
    print(f"❓ {q}")
    response = rag.query(q, top_k=3)
    print(f"💬 {response.answer[:200]}...")
    print(f"   Confidence: {response.confidence_score:.0%}\n")

print("✅ Done! Modify this script to ask your own questions.")
