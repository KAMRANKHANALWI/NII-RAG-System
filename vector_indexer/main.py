"""Main entry point for the vector indexer system (UPDATED)."""

import argparse
import logging
import os
import sys
from typing import List
from config.settings import COLLECTION_CONFIG, LOG_LEVEL, LOG_FORMAT, BASE_PATH
from core.document_loader import DocumentLoader
from core.vectorstore import VectorStoreManager
from auditing.duplicate_auditor import DuplicateAuditor


def setup_logging(verbose: bool = False):
    """
    Configure logging for the application.

    Args:
        verbose: Enable debug logging if True
    """
    level = logging.DEBUG if verbose else getattr(logging, LOG_LEVEL)
    logging.basicConfig(
        level=level,
        format=LOG_FORMAT,
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler("vector_indexer.log", mode="a"),
        ],
    )


def validate_collection(collection_name: str) -> bool:
    """
    Validate that collection exists in configuration.

    Args:
        collection_name: Name of collection to validate

    Returns:
        True if valid, False otherwise
    """
    if collection_name not in COLLECTION_CONFIG:
        logging.error(f"❌ Unknown collection: {collection_name}")
        logging.info(f"Available collections: {', '.join(COLLECTION_CONFIG.keys())}")
        return False
    return True


def check_paths() -> bool:
    """
    Check that required paths exist.

    Returns:
        True if all paths exist, False otherwise
    """
    if not os.path.exists(BASE_PATH):
        logging.error(f"❌ Base path does not exist: {BASE_PATH}")
        return False

    missing_folders = []
    for collection_name, folder_config in COLLECTION_CONFIG.items():
        folders = [folder_config] if isinstance(folder_config, str) else folder_config
        for folder in folders:
            folder_path = os.path.join(BASE_PATH, folder)
            if not os.path.exists(folder_path):
                missing_folders.append(folder_path)

    if missing_folders:
        logging.warning(f"⚠️ Missing folders: {', '.join(missing_folders)}")
        logging.info("Continuing with available folders...")

    return True


def index_collection(collection_name: str, folder_names: List[str]) -> int:
    """
    Index a single collection with enhanced error handling.

    Args:
        collection_name: Name of the collection
        folder_names: List of folder names to process

    Returns:
        Total number of documents indexed
    """
    logging.info(f"🚀 Starting indexing for collection: {collection_name}")

    try:
        loader = DocumentLoader()
        manager = VectorStoreManager(collection_name)

        all_documents = []

        for folder_name in folder_names:
            folder_path = os.path.join(BASE_PATH, folder_name)

            if not os.path.exists(folder_path):
                logging.warning(f"⚠️ Folder does not exist: {folder_path}")
                continue

            logging.info(f"📁 Processing folder: {folder_path}")

            docs = loader.load_documents_from_folder(folder_path, collection_name)
            logging.info(f"➕ Loaded {len(docs)} documents from {folder_path}")
            all_documents.extend(docs)

        if not all_documents:
            logging.warning(f"⚠️ No documents found for collection: {collection_name}")
            return 0

        indexed_count = manager.add_new_documents(all_documents)
        logging.info(
            f"✅ Successfully indexed {indexed_count} documents for {collection_name}"
        )

        return indexed_count

    except Exception as e:
        logging.error(f"❌ Failed to index collection {collection_name}: {e}")
        return 0


def index_all_collections() -> int:
    """
    Index all configured collections with progress tracking.

    Returns:
        Total number of documents indexed across all collections
    """
    logging.info("🚀 Starting indexing process for all collections")
    total_indexed = 0
    successful_collections = 0
    failed_collections = []

    for collection_name, folder_config in COLLECTION_CONFIG.items():
        try:
            folders = (
                [folder_config] if isinstance(folder_config, str) else folder_config
            )
            indexed_count = index_collection(collection_name, folders)
            total_indexed += indexed_count

            if indexed_count > 0:
                successful_collections += 1
            else:
                failed_collections.append(collection_name)

        except Exception as e:
            logging.error(f"❌ Failed to process collection {collection_name}: {e}")
            failed_collections.append(collection_name)

    # Summary
    logging.info(f"📊 Indexing Summary:")
    logging.info(f"   ✅ Successful collections: {successful_collections}")
    logging.info(f"   ❌ Failed collections: {len(failed_collections)}")
    logging.info(f"   📄 Total documents indexed: {total_indexed}")

    if failed_collections:
        logging.warning(f"   ⚠️ Failed collections: {', '.join(failed_collections)}")

    return total_indexed


def main():
    """Main entry point with enhanced CLI and error handling."""
    parser = argparse.ArgumentParser(
        description="Modular vector indexer for JSON documents with comprehensive chunking.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                                    # Index all collections
  %(prog)s --collection faculty_info         # Index specific collection
  %(prog)s --audit                          # Index all and audit duplicates
  %(prog)s --audit-only                     # Only run duplicate audit
  %(prog)s --collection labs --audit --verbose  # Index labs with audit and debug logs
        """,
    )

    parser.add_argument(
        "--audit", action="store_true", help="Run duplicate audit after indexing"
    )
    parser.add_argument(
        "--audit-only",
        action="store_true",
        help="Run only duplicate audit without indexing",
    )
    parser.add_argument(
        "--collection",
        type=str,
        choices=list(COLLECTION_CONFIG.keys()),
        help="Index only specific collection",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Enable verbose debug logging"
    )
    parser.add_argument(
        "--list-collections",
        action="store_true",
        help="List all available collections and exit",
    )

    args = parser.parse_args()

    # Setup logging
    setup_logging(args.verbose)

    # List collections if requested
    if args.list_collections:
        print("Available collections:")
        for name, folder in COLLECTION_CONFIG.items():
            print(f"  • {name} -> {folder}")
        return

    # Check paths
    if not check_paths():
        sys.exit(1)

    try:
        if args.audit_only:
            # Run audit only
            logging.info("🔍 Running duplicate audit only")
            DuplicateAuditor.audit_all_collections(list(COLLECTION_CONFIG.keys()))

        elif args.collection:
            # Index specific collection
            if not validate_collection(args.collection):
                sys.exit(1)

            folder_config = COLLECTION_CONFIG[args.collection]
            folders = (
                [folder_config] if isinstance(folder_config, str) else folder_config
            )

            indexed_count = index_collection(args.collection, folders)

            if indexed_count == 0:
                logging.warning("No documents were indexed")

            if args.audit:
                logging.info("🔍 Running duplicate audit for collection")
                DuplicateAuditor.audit_collection(args.collection)

        else:
            # Index all collections
            total_indexed = index_all_collections()

            if total_indexed == 0:
                logging.warning("No documents were indexed across all collections")

            if args.audit:
                logging.info("🔍 Running duplicate audit for all collections")
                DuplicateAuditor.audit_all_collections(list(COLLECTION_CONFIG.keys()))

        logging.info("🎉 Process completed successfully!")

    except KeyboardInterrupt:
        logging.info("🛑 Process interrupted by user")
        sys.exit(130)
    except Exception as e:
        logging.error(f"💥 Unexpected error: {e}")
        if args.verbose:
            logging.exception("Full traceback:")
        sys.exit(1)


if __name__ == "__main__":
    main()
