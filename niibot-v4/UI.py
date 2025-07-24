# UI.py
# Streamlit interface for modular NIIBot

import warnings
import os
import logging

# Suppress PyTorch warnings that clutter Streamlit console
warnings.filterwarnings("ignore", category=UserWarning, module="torch")
warnings.filterwarnings("ignore", category=FutureWarning, module="torch")
warnings.filterwarnings("ignore", message=".*torch.classes.*")
warnings.filterwarnings("ignore", category=RuntimeWarning, message=".*event loop.*")

# Set environment variables to suppress torch warnings
os.environ["PYTORCH_DISABLE_WARNINGS"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Reduce logging level for problematic modules
logging.getLogger("torch").setLevel(logging.ERROR)
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("streamlit").setLevel(logging.ERROR)

# streamlit and other modules
import streamlit as st
import time
import json
from datetime import datetime
from typing import List, Dict, Any
import uuid

# Updated imports for modular architecture
from chains.rag_chain import get_enhanced_chain_for_domain
from domain_router import classify_domain
from config.settings import llm

# Page configuration
st.set_page_config(
    page_title="NIIBot",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for minimalistic design (same as before)
st.markdown(
    """
<style>
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Clean header */
    .main-header {
        font-size: 2rem;
        font-weight: 600;
        color: #1f2937;
        margin-bottom: 0.5rem;
    }
    
    .sub-header {
        color: #6b7280;
        font-size: 0.95rem;
        margin-bottom: 2rem;
    }
    
    /* Chat message styling */
    .stChatMessage {
        padding: 0.75rem 1rem;
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        padding: 1rem;
    }
    
    /* Button styling */
    .stButton > button {
        background-color: #f3f4f6;
        color: #1f2937;
        border: 1px solid #e5e7eb;
        padding: 0.5rem 1rem;
        font-size: 0.875rem;
        font-weight: 500;
        border-radius: 0.375rem;
        width: 100%;
        margin-bottom: 0.5rem;
    }
    
    .stButton > button:hover {
        background-color: #e5e7eb;
        border-color: #d1d5db;
    }
    
    /* Chat input styling */
    .stChatInput {
        border-color: #e5e7eb;
    }
    
    /* Remove extra padding */
    .block-container {
        padding-top: 2rem;
        max-width: 1000px;
    }
    
    /* Domain tag */
    .domain-tag {
        background: #e5e7eb;
        color: #4b5563;
        padding: 0.25rem 0.75rem;
        border-radius: 0.25rem;
        font-size: 0.75rem;
        font-weight: 500;
        display: inline-block;
        margin-bottom: 0.5rem;
    }
    
    /* New chat section */
    .new-chat-section {
        border-bottom: 1px solid #e5e7eb;
        padding-bottom: 1rem;
        margin-bottom: 1rem;
    }
    
    /* Chat history item */
    .chat-item {
        padding: 0.75rem;
        margin-bottom: 0.5rem;
        border-radius: 0.375rem;
        cursor: pointer;
        transition: background-color 0.2s;
    }
    
    .chat-item:hover {
        background-color: #f3f4f6;
    }
    
    .chat-item.active {
        background-color: #e5e7eb;
    }
</style>
""",
    unsafe_allow_html=True,
)


class MinimalNIIBot:
    """Minimalistic Streamlit interface for modular NIIBot"""

    def __init__(self):
        self.initialize_session_state()

    def initialize_session_state(self):
        """Initialize session state variables"""
        if "chats" not in st.session_state:
            # Initialize with one empty chat
            initial_chat_id = str(uuid.uuid4())
            st.session_state.chats = {
                initial_chat_id: {
                    "id": initial_chat_id,
                    "title": "New Chat",
                    "messages": [],
                    "created_at": datetime.now(),
                }
            }
            st.session_state.current_chat_id = initial_chat_id

        if "current_chat_id" not in st.session_state:
            st.session_state.current_chat_id = list(st.session_state.chats.keys())[0]

    def create_header(self):
        """Create minimalistic header"""
        st.markdown('<h1 class="main-header">NIIBot</h1>', unsafe_allow_html=True)
        st.markdown(
            '<p class="sub-header">National Institute of Immunology Assistant</p>',
            unsafe_allow_html=True,
        )

    def create_sidebar(self):
        """Create minimalistic sidebar with chat management"""
        with st.sidebar:
            # New chat button
            st.markdown('<div class="new-chat-section">', unsafe_allow_html=True)
            if st.button("➕ New Chat", use_container_width=True):
                self.create_new_chat()
            st.markdown("</div>", unsafe_allow_html=True)

            # Chat history
            st.markdown("### Chats")

            # Sort chats by creation time (most recent first)
            sorted_chats = sorted(
                st.session_state.chats.values(),
                key=lambda x: x["created_at"],
                reverse=True,
            )

            for chat in sorted_chats:
                chat_title = chat["title"]
                if len(chat_title) > 30:
                    chat_title = chat_title[:27] + "..."

                # Create a unique key for each button
                if st.button(
                    f"💬 {chat_title}",
                    key=f"chat_{chat['id']}",
                    use_container_width=True,
                    disabled=(chat["id"] == st.session_state.current_chat_id),
                ):
                    st.session_state.current_chat_id = chat["id"]
                    st.rerun()

            # Divider
            st.markdown("---")

            # Example queries section
            with st.expander("Example Queries"):
                examples = [
                    "Who is Dr. Debasisa Mohanty?",
                    "Tell me about Dr. Vineeta Bal's research",
                    "What are Dr. Monica Sundd's publications?",
                    "About NII director",
                    "Tell me about Dr. Nimesh Gupta's lab",
                ]

                for example in examples:
                    if st.button(
                        example, key=f"ex_{example}", use_container_width=True
                    ):
                        self.process_query(example)

            # Clear current chat
            st.markdown("---")
            if st.button("🗑️ Clear Current Chat", use_container_width=True):
                self.clear_current_chat()

    def create_new_chat(self):
        """Create a new chat session"""
        new_chat_id = str(uuid.uuid4())
        st.session_state.chats[new_chat_id] = {
            "id": new_chat_id,
            "title": "New Chat",
            "messages": [],
            "created_at": datetime.now(),
        }
        st.session_state.current_chat_id = new_chat_id
        st.rerun()

    def clear_current_chat(self):
        """Clear messages in current chat"""
        current_chat = st.session_state.chats[st.session_state.current_chat_id]
        current_chat["messages"] = []
        current_chat["title"] = "New Chat"
        st.rerun()

    def update_chat_title(self, query: str):
        """Update chat title based on first query"""
        current_chat = st.session_state.chats[st.session_state.current_chat_id]
        if current_chat["title"] == "New Chat" and len(current_chat["messages"]) == 0:
            # Generate title from first query (max 50 chars)
            title = query[:50] + "..." if len(query) > 50 else query
            current_chat["title"] = title

    def display_chat(self):
        """Display current chat messages"""
        current_chat = st.session_state.chats[st.session_state.current_chat_id]

        if not current_chat["messages"]:
            # Empty state
            st.markdown(
                """
                <div style="text-align: center; color: #9ca3af; padding: 3rem 0;">
                    <p style="font-size: 1.1rem;">Ask me about NII faculty, research, or publications</p>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            # Display messages
            for message in current_chat["messages"]:
                with st.chat_message(message["role"]):
                    if message["role"] == "assistant" and "domain" in message:
                        st.markdown(
                            f'<span class="domain-tag">{message["domain"].replace("_", " ").title()}</span>',
                            unsafe_allow_html=True,
                        )
                    st.write(message["content"])

    def process_query(self, query: str):
        """Process user query using modular architecture"""
        if not query.strip():
            return

        # Update chat title if needed
        self.update_chat_title(query)

        # Get current chat
        current_chat = st.session_state.chats[st.session_state.current_chat_id]

        # Add user message
        current_chat["messages"].append({"role": "user", "content": query})

        # Display user message immediately
        with st.chat_message("user"):
            st.write(query)

        # Process with loading state
        with st.chat_message("assistant"):
            with st.spinner(""):
                try:
                    # Classify domain using existing domain router
                    domain = classify_domain(query)

                    # Display domain tag
                    st.markdown(
                        f'<span class="domain-tag">{domain.replace("_", " ").title()}</span>',
                        unsafe_allow_html=True,
                    )

                    # Get chain using new modular architecture
                    chain, _ = get_enhanced_chain_for_domain(
                        domain, use_memory_wrapper=False
                    )

                    if chain is None:
                        response = (
                            "I couldn't access the knowledge base. Please try again."
                        )
                    else:
                        # Prepare chat history for context
                        chat_history = []
                        for msg in current_chat["messages"][
                            :-1
                        ]:  # Exclude current query
                            if msg["role"] == "user":
                                chat_history.append(
                                    {"query": msg["content"], "response": ""}
                                )
                            elif msg["role"] == "assistant" and chat_history:
                                chat_history[-1]["response"] = msg["content"]

                        # Get response using modular chain
                        response = chain.invoke(
                            {"question": query, "chat_history": chat_history}
                        )

                    # Display response
                    st.write(response)

                    # Add assistant message to history
                    current_chat["messages"].append(
                        {"role": "assistant", "content": response, "domain": domain}
                    )

                except Exception as e:
                    error_msg = f"An error occurred: {str(e)[:100]}..."
                    st.error(error_msg)

                    # Add error to history
                    current_chat["messages"].append(
                        {"role": "assistant", "content": error_msg, "domain": "error"}
                    )

    def run(self):
        """Main application runner"""
        # Sidebar
        self.create_sidebar()

        # Main content
        self.create_header()

        # Chat display
        self.display_chat()

        # Chat input
        if prompt := st.chat_input("Type your question..."):
            self.process_query(prompt)


def main():
    """Main function"""
    app = MinimalNIIBot()
    app.run()


if __name__ == "__main__":
    main()
