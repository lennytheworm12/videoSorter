"""
PHASE 5: Query the knowledge base using semantic search + RAG.

PURPOSE:
    Takes a natural language question, finds the most relevant insights from
    the coach's entire video library, then uses Gemma 4 to generate a structured
    answer grounded in those insights.

HOW IT WORKS:
    1. RETRIEVAL — embed the user's question into a vector using the same
       sentence-transformers model used in embed.py. Compute cosine similarity
       against every stored insight vector. Return the top-k closest matches.

    2. GENERATION — feed the retrieved insights as context into Gemma 4 with
       a prompt like: "Using only these coaching insights, answer the question."
       Gemma synthesizes a coherent answer rather than just listing raw tips.

EXAMPLE QUERIES:
    "How do I play Cassiopeia against poke mages?"
    "What does the coach say about wave management in mid lane?"
    "What items should I build on Vladimir?"
    "How do I roam effectively as a mid laner?"

FILTER OPTIONS:
    Results can be pre-filtered before vector search by:
    - role (mid, top, jungle, adc, support)
    - champion name
    - insight_type (principles, laning_tips, matchup_advice, macro_advice,
      champion_mechanics, teamfight_tips, vision_control, itemization, general_advice)
    This narrows the search space and improves relevance.

INSIGHT TYPES EXPLAINED:
    principles        — underlying mental models, matchup archetypes, wave logic,
                        win conditions. The WHY behind other tips.
    laning_tips       — specific actions during the laning phase
    champion_mechanics — abilities, combos, positioning for a specific champion
    matchup_advice    — how to play against specific champions or champion types
    macro_advice      — roaming, objective control, map decisions
    teamfight_tips    — positioning, target priority, engage/disengage
    vision_control    — ward placement, sweeper usage, vision timing
    itemization       — item choices, build paths, situational buys
    general_advice    — mindset, game sense, applies across champions/roles

DEPENDENCIES:
    embed.py must have run first — insights need vectors before they can be searched.
"""

# TODO: import numpy as np
# TODO: import ollama
# from embed import load_all_vectors
# from database import get_connection

TOP_K = 10  # number of insights to retrieve per query


def cosine_similarity_search(query_vector, matrix, top_k: int = TOP_K):
    """
    Compute cosine similarity between query_vector and every row in matrix.
    Returns indices of the top_k most similar rows.
    """
    # TODO:
    # 1. Normalize query_vector and matrix rows to unit length
    # 2. Dot product gives cosine similarity for unit vectors
    # 3. Return top_k indices sorted by score descending
    pass


def retrieve(
    question: str,
    role: str | None = None,
    champion: str | None = None,
    insight_type: str | None = None,
    top_k: int = TOP_K,
) -> list[dict]:
    """
    Embed the question, optionally filter the insight pool, then return
    the top_k most semantically similar insights with their metadata.
    """
    # TODO:
    # 1. Load SentenceTransformer and encode the question
    # 2. Load all vectors from embed.load_all_vectors()
    # 3. Apply optional filters (role, champion, insight_type) to narrow pool
    # 4. Run cosine_similarity_search
    # 5. Return top_k insights as list of dicts {text, role, champion, insight_type, score}
    pass


def answer(question: str, role: str | None = None, champion: str | None = None) -> str:
    """
    Full RAG pipeline: retrieve relevant insights → generate a structured answer.

    This is the main entry point for querying the knowledge base.
    """
    # TODO:
    # 1. Call retrieve() to get top_k relevant insights
    # 2. Format them as a numbered list for the LLM context
    # 3. Send to Gemma 4 with a prompt:
    #    "You are a LoL coach assistant. Using ONLY the following coaching insights
    #     from a real coach's video library, answer the question. Cite which insight
    #     supports each point. Do not add advice not present in the insights."
    # 4. Return the generated answer string
    pass


if __name__ == "__main__":
    # Example usage once implemented
    # print(answer("How do I play Cassiopeia against poke mages?", role="mid", champion="Cassiopeia"))
    pass
