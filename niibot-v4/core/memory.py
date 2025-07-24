from langchain_community.chat_message_histories.in_memory import ChatMessageHistory

# ==== Memory Management Section ====
"""
Session-based memory system to maintain conversation history
Each user session gets its own memory store
"""
session_histories = {}  # Dictionary to store chat histories by session ID


def get_session_history(session_id: str):
    """
    Retrieves or creates a new chat history for a given session.

    Args:
        session_id (str): Unique identifier for the user session

    Returns:
        ChatMessageHistory: Memory object containing conversation history

    Purpose: Enables contextual conversations by remembering previous exchanges
    """
    if session_id not in session_histories:
        session_histories[session_id] = ChatMessageHistory()
    return session_histories[session_id]


def rewrite_query_with_context(query: str, chat_history: list, llm) -> str:
    """
    Intelligently rewrites user queries to resolve pronouns and ambiguous references
    using conversation history for context.

    Args:
        query (str): Current user query that may contain pronouns (his, her, their)
        chat_history (list): Previous conversation exchanges for context
        llm: Language model to perform the rewriting

    Returns:
        str: Rewritten query with pronouns replaced by actual names

    Example:
        Context: "Tell me about Dr. Monica Sundd"
        Query: "her publications" → "Dr. Monica Sundd's publications"

    Purpose: Converts ambiguous queries into self-contained, searchable queries
    """
    # If no chat history, return original query
    if not chat_history or len(chat_history) == 0:
        return query

    # Check if query contains pronouns that need resolution
    pronouns = ["his", "her", "their", "its", "he", "she", "they"]
    query_lower = query.lower()

    # Quick optimization: if no pronouns and no implicit references, return as-is
    if not any(pronoun in query_lower.split() for pronoun in pronouns):
        # Also check for implicit references like "publications", "research"
        if not any(
            word in query_lower
            for word in ["about his", "about her", "publications", "research", "lab"]
        ):
            return query

    # If query already contains a specific name, don't rewrite
    if "dr." in query_lower or "prof." in query_lower:
        return query

    # Extract recent conversation context (last exchange only for efficiency)
    context_parts = []
    for i, chat in enumerate(chat_history[-1:]):  # Last 1 exchange
        # Handle both CLI and Streamlit conversation formats
        if isinstance(chat, dict):
            if "query" in chat:
                context_parts.append(f"User: {chat['query']}")
            if "response" in chat:
                # Truncate response to avoid token limits
                response_preview = (
                    chat["response"][:300] + "..."
                    if len(chat["response"]) > 300
                    else chat["response"]
                )
                context_parts.append(f"Assistant: {response_preview}")
        else:
            # Handle other conversation formats
            context_parts.append(str(chat))

    if not context_parts:
        return query  # No valid context, return original

    context = "\n".join(context_parts)

    # Create a specialized prompt for query rewriting
    rewrite_prompt = f"""Given this conversation history:
    {context}

    The user now asks: "{query}"

    IMPORTANT: Look at the MOST RECENT person mentioned in the conversation.
    - If the last person mentioned was female (Dr. Sarika Gupta, Dr. Monica Sundd, etc.), replace "his" with her name
    - If the last person mentioned was male (Dr. Debasisa Mohanty, Dr. Nimesh Gupta, etc.), replace "her" with his name

    Rewrite this query to be self-contained by:
    1. Replacing ALL pronouns (his/her/their) with the actual person's FULL NAME from the conversation
    2. The person's name should be exactly as it appears in the conversation (e.g., "Dr. Monica Sundd")
    3. Keep the query natural and concise

    Examples:
    - Context: "Tell me about Dr. Monica Sundd" → Query: "her publications" → "Dr. Monica Sundd's publications"
    - Context: "Tell me about Dr. Sarika Gupta" → Query: "his publications" → "Dr. Sarika Gupta's publications"
    - Context: "Who is the director" (Dr. Debasisa Mohanty) → Query: "his research" → "Dr. Debasisa Mohanty's research"

    For the current conversation, identify the most recent person mentioned and rewrite accordingly.

    Return ONLY the rewritten query, nothing else:"""

    try:
        rewritten = llm.invoke(rewrite_prompt)
        # Extract content from LLM response object
        if hasattr(rewritten, "content"):
            rewritten = rewritten.content

        rewritten = str(rewritten).strip()

        # Validate the rewritten query (reasonable length check)
        if len(rewritten) > 0 and len(rewritten) < 200:
            print(f"✏️ Rewritten query: '{query}' → '{rewritten}'")
            return rewritten
        else:
            return query
    except Exception as e:
        print(f"⚠️ Query rewriting failed: {e}")
        return query
