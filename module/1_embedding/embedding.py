# -*- coding: utf-8 -*-
"""
================================================================================
Embeddings & Similarity Search Demo
================================================================================

WHY EMBEDDINGS MATTER:
━━━━━━━━━━━━━━━━━━━━━
Traditional search: "password reset" only finds docs with those exact words
Semantic search:    "password reset" also finds "forgot credentials",
                    "can't log in after changing password", "auth issues"

This is the FOUNDATION of RAG - without good embeddings, retrieval fails!

1. How to generate embeddings from text using OpenAI
2. What embeddings actually look like (1536 numbers!)
3. Computing similarity scores (cosine similarity)
4. Finding most similar documents (semantic search)
5. Visualizing the relationships embeddings capture

THE BIG PICTURE:
━━━━━━━━━━━━━━━
    Text → [Embedding Model] → Vector (1536 floats) → [Compare] → Similarity Score

    "login fails"     →  [0.023, -0.041, ...]  ─┐
                                                 ├─→ 0.89 (similar!)
    "auth not working" → [0.019, -0.038, ...]  ─┘

LEARNING RESOURCES:
- OpenAI Embeddings Guide: https://platform.openai.com/docs/guides/embeddings
- Understanding Vector Embeddings: https://www.pinecone.io/learn/vector-embeddings/
- Cosine Similarity Explained: https://en.wikipedia.org/wiki/Cosine_similarity
"""
import json
import os

import numpy as np
from dotenv import load_dotenv
from openai import OpenAI
from sklearn.metrics.pairwise import cosine_similarity  # Measure similarity between vectors


# Load dotenv
load_dotenv()

# =============================================================================
# INITIALIZE OPENAI CLIENT
# =============================================================================
print("Initializing OpenAI client...")
print(os.getenv('OPENAI_API_KEY'))
client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
# =============================================================================
# EMBEDDING MODEL SELECTION
# =============================================================================
# OpenAI offers several embedding models (as of 2024):
#
# | Model                    | Dimensions | Cost          | Use Case            |
# |--------------------------|------------|---------------|---------------------|
# | text-embedding-3-small   | 1536       | $0.02/1M tok  | General purpose ✓   |
# | text-embedding-3-large   | 3072       | $0.13/1M tok  | Higher quality      |
#
# Reference: https://platform.openai.com/docs/guides/embeddings/embedding-models
# =============================================================================
embedding_model = os.getenv('OPENAI_EMBEDDING_MODEL', 'text-embedding-3-small')
embedding_dim = 1536
# =============================================================================
# INITIALIZE OPENAI CLIENT
# =============================================================================
client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
# =============================================================================
# LOAD DATA: Support Tickets
# =============================================================================
print("\nLoading support tickets...")
with open('../../data/synthetic_tickets.json', 'r') as f:
    tickets = json.load(f)
print(f"Loaded {len(tickets)} support tickets")
# ============================================================================
# PART 1: Generate Embeddings
# ============================================================================
# WHAT IS AN EMBEDDING?
# ━━━━━━━━━━━━━━━━━━━━
# A list of 1536 numbers that represents the "meaning" of text.
# Think of it as coordinates in a 1536-dimensional space where
# similar meanings are located close together.
# -----------------------------------------------------------------------------
# Prepare text for embedding
# -----------------------------------------------------------------------------
# TIP: Combine relevant fields for richer context
# The more context, the better the embedding captures the meaning
#
# BAD:  Just title Ex: "Login issue"
# GOOD: Title + description Ex: "Login issue. User reports authentication
#       failure after password reset. Error code 401..."
# -----------------------------------------------------------------------------
ticket_texts = [
    f"{ticket['title']}. {ticket['description']}"
    for ticket in tickets
]
# -----------------------------------------------------------------------------
# Generate embeddings via OpenAI API
# -----------------------------------------------------------------------------
# This is a BATCH call - we send all texts at once for efficiency
# The API returns one embedding per input text
# -----------------------------------------------------------------------------
print("\nGenerating embeddings for all tickets...")
response = client.embeddings.create(input=ticket_texts, model=embedding_model)
# Convert API response to NumPy array for mathematical operations
# Shape: (num_tickets, 1536) - each row is one ticket's embedding
embeddings = np.array([data.embedding for data in response.data])

# Create a search query - this is what a user might type
query = "Users can't login after changing password"
print(f"\nSearch Query: '{query}'")

# -----------------------------------------------------------------------------
# Generate embedding for the query
# -----------------------------------------------------------------------------
# CRITICAL: Use the SAME model as documents!
# Different models produce incompatible vector spaces
#
# text-embedding-3-small documents + text-embedding-3-large query = WRONG!
# -----------------------------------------------------------------------------
query_response = client.embeddings.create(input=[query], model=embedding_model)
print(query_response);
query_embedding = np.array([query_response.data[0].embedding])
print(f"Query embedding shape: {query_embedding.shape}")  # (1, 1536)

# -----------------------------------------------------------------------------
# Compute cosine similarity between query and ALL tickets
# -----------------------------------------------------------------------------
# This is the core of semantic search!
# We compare the query vector against every document vector
# Result: one similarity score per document
#
# EXACTLY what this line does:
#   similarities = cosine_similarity(query_embedding, embeddings)[0]
#
# Shapes in this demo:
#   query_embedding: (1, 1536)      -> 1 query vector
#   embeddings:      (N, 1536)      -> N ticket vectors
#   cosine_similarity(...) returns: (1, N)
#
# Why [0]?
#   There is only one query row, so [0] extracts that row,
#   giving a 1D array of length N (one score per ticket).
#
# Mini-example:
#   query_embedding = [[1, 1]]
#   embeddings      = [[1, 1], [1, 0], [-1, -1]]
#   cosine_similarity(...) -> [[1.0000, 0.7071, -1.0000]]   # shape (1, 3)
#   ...[0]                ->  [1.0000, 0.7071, -1.0000]     # shape (3,)
# -----------------------------------------------------------------------------
similarities = cosine_similarity(query_embedding, embeddings)[0]
print(f"\nComputed similarity scores for {len(similarities)} tickets")
print(f"Similarity range: [{similarities.min():.4f}, {similarities.max():.4f}]")

# ============================================================================
# PART 3: Retrieve Most Similar Tickets
# ============================================================================
#
# THIS IS SEMANTIC SEARCH!
# ━━━━━━━━━━━━━━━━━━━━━━━━
# Instead of matching keywords, we find documents with similar MEANING.
#
# The process:
# 1. Embed the query → vector
# 2. Compare to all document vectors → similarity scores
# 3. Sort by similarity → ranked results
# 4. Return top-K → most relevant documents
#
# Notice: "password reset" in query will match "forgot credentials"
#         even though they share no words!
#
# ============================================================================
print("\n" + "=" * 80)
print("PART 3: Finding Most Similar Tickets")
print("=" * 80)

# Get top-5 most similar tickets
top_k = 2

# np.argsort() returns indices that would sort the array (ascending)
# [::-1] reverses to get descending order (highest similarity first)
# [:top_k] takes only the top K results
top_indices = np.argsort(similarities)[::-1][:top_k]

print(f"\nTop {top_k} most similar tickets to query: '{query}'")
print("-" * 80)

for rank, idx in enumerate(top_indices, 1):
    ticket = tickets[idx]
    score = similarities[idx]

    print(f"\n#{rank} - Similarity: {score:.4f}")
    print(f"Ticket ID: {ticket['ticket_id']}")
    print(f"Title: {ticket['title']}")

print("""
KEY TAKEAWAYS:
━━━━━━━━━━━━━━
1. EMBEDDINGS convert text → 1536-dimensional vectors
   Each vector encodes the semantic "meaning" of the text

2. COSINE SIMILARITY measures how similar two vectors are
   Range: -1 to 1 (higher = more similar)

3. SEMANTIC SEARCH finds meaning, not just keywords
   "password reset" matches "forgot credentials"! 

4. THE SAME MODEL must be used for queries and documents.
   Different models = incompatible vector spaces

5. SIMILARITY SCORES are the truth
   Don't trust 2D visualizations - they lose information

CONNECTION TO RAG:
━━━━━━━━━━━━━━━━━
In RAG, we use embeddings to find relevant documents,
then pass those documents to an LLM for generation.

Query → [Embed] → [Search] → Top Documents → [LLM] → Answer

NEXT: Module 2 - Chunking & Vector Stores
""")