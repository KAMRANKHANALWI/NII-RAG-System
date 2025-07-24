from chains.rag_chain import get_enhanced_chain_for_domain
from domain_router import classify_domain
from utils.document_utils import display_sources
from core.caching import (
    _cached_query_preprocessing,
    _cached_name_lookup,
)


def run_chat():
    """
    Main command-line interface for the NIIBot system.
    NOW INCLUDES CACHE PERFORMANCE MONITORING.

    Features:
    - Interactive chat loop
    - Domain routing for queries
    - Conversation memory
    - Source attribution
    - Enhanced error handling
    - Cache performance monitoring

    Usage: Run this function to start an interactive chat session with NIIBot.
    """
    print("🎯 " + "=" * 70)
    print("🤖 NIIBot v4.0 - Comprehensive Multi-Source RAG System")
    print("🎯 " + "=" * 70)
    print("\n💡 Enhanced Features:")
    print("   • Smart faculty name extraction and matching")
    print("   • Metadata-filtered retrieval for accurate results")
    print("   • Multi-source comprehensive information gathering")
    print("   • Enhanced director query handling with rich context")
    print("   • Cross-domain search for complete faculty profiles")
    print("   • 🚀 LRU Caching for improved performance")
    print("\n📝 Type 'exit', 'quit', or 'bye' to end the session")
    print("📊 Type 'cache_stats' to see caching performance")
    print("🎯 " + "=" * 70 + "\n")

    session_id = "enhanced-nii-session"
    chat_history = []  # Track conversation for context

    while True:
        try:
            user_query = input("👤 Ask about NII faculty or institute: ").strip()

            if not user_query:
                continue

            # Handle special commands
            if user_query.lower() == "cache_stats":
                print("\n📊 Cache Performance Statistics:")
                print(
                    f"   🔍 Query preprocessing cache: {_cached_query_preprocessing.cache_info()}"
                )
                print(f"   👥 Name lookup cache: {_cached_name_lookup.cache_info()}")

                # Check if we have an active faculty extractor with cache
                try:
                    # Test with a simple domain classification
                    domain = "faculty_info"  # Default for testing

                    # Check if extractor is already initialized
                    if hasattr(_cached_name_lookup, "_extractor"):
                        extractor = _cached_name_lookup._extractor
                        print(
                            f"   📝 Name extraction cache: {extractor._cached_extract_names.cache_info()}"
                        )
                    else:
                        print(
                            "   📝 Name extraction cache: Not yet initialized (no queries processed)"
                        )

                except Exception as e:
                    print(
                        f"   📝 Name extraction cache: Error checking ({str(e)[:50]}...)"
                    )

                print("   💡 Higher hit ratios = better performance")
                print()
                continue

            # Handle exit commands
            if user_query.lower() in {"exit", "quit", "bye", "goodbye"}:
                print("\n📊 Final Cache Statistics:")
                print(
                    f"   🔍 Query preprocessing: {_cached_query_preprocessing.cache_info()}"
                )
                print(f"   👥 Name lookup: {_cached_name_lookup.cache_info()}")
                print("\n👋 Thank you for using NIIBot! Goodbye!")
                break

            # ===== DOMAIN CLASSIFICATION =====
            # Route query to appropriate domain/collection
            try:
                domain = classify_domain(user_query)
                print(f"🔀 Routing to: {domain}")
            except Exception as e:
                print(f"⚠️ Domain classification failed: {e}")
                domain = "nii_info"  # Default fallback
                print(f"🔄 Using default domain: {domain}")

            # ===== GET DOMAIN-SPECIFIC CHAIN =====
            result = get_enhanced_chain_for_domain(domain)
            if result[0] is None:
                print("❌ Error: Could not load the knowledge base.")
                print("\n💡 This is likely because:")
                print("   1. Vectorstore files are missing")
                print("   2. Wrong directory structure")
                print("   3. Embedding model mismatch")
                print("   4. Chroma dependencies not installed")
                print("\n🔧 To fix this:")
                print("   1. Ensure vectorstore directory exists: ./vectorstores/")
                print(
                    "   2. Create collections: nii_info, faculty_info, research, publications, labs"
                )
                print("   3. Install: pip install chromadb")
                print("   4. Check your embedding model compatibility")
                print("\nPlease try again after setting up the vectorstore.\n")
                continue

            chain, _ = result

            # ===== PREPARE INPUT AND GET RESPONSE =====
            chain_input = {
                "question": user_query,
                "chat_history": chat_history,  # Pass conversation history
            }

            # Get response from the chain
            response = chain.invoke(
                chain_input,
                config={"configurable": {"session_id": session_id}},
            )

            # ===== DISPLAY RESPONSE =====
            print(f"\n🤖 NIIBot: {response}")

            # Update chat history for next iteration
            chat_history.append({"query": user_query, "response": response})

            # Show source attribution
            if "retrieved_docs" in chain_input:
                display_sources(chain_input["retrieved_docs"])

            print("\n" + "-" * 70 + "\n")

        except KeyboardInterrupt:
            print("\n\n👋 Session interrupted. Goodbye!")
            break
        except Exception as e:
            print(f"\n❌ An error occurred: {e}")
            print("Please try rephrasing your question or try again.\n")


# ==== Entry Point ====
if __name__ == "__main__":
    """
    Entry point for the NIIBot application.
    When run directly, starts the interactive chat interface.
    """
    run_chat()
