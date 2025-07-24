"""Duplicate detection and auditing functionality."""

import logging
from collections import defaultdict
from typing import Dict, List
from core.vectorstore import VectorStoreManager


class DuplicateAuditor:
    """Handles duplicate detection and auditing in vector collections."""

    @staticmethod
    def audit_collection(collection_name: str) -> Dict[str, int]:
        """
        Audit a single collection for duplicates.

        Args:
            collection_name: Name of collection to audit

        Returns:
            Dictionary of duplicate hashes and their counts
        """
        logging.info(f"🔍 Auditing collection: {collection_name}")

        manager = VectorStoreManager(collection_name)
        hash_count = defaultdict(int)

        try:
            results = manager.vectorstore.get(include=["metadatas"])
            for metadata in results["metadatas"]:
                if isinstance(metadata, dict) and "doc_hash" in metadata:
                    hash_count[metadata["doc_hash"]] += 1
        except Exception as e:
            logging.error(f"Failed to read metadata from {collection_name}: {e}")
            return {}

        duplicates = {h: c for h, c in hash_count.items() if c > 1}

        if duplicates:
            logging.warning(
                f"⚠️ Found {len(duplicates)} duplicate hashes in {collection_name}!"
            )
            for h, c in duplicates.items():
                logging.warning(f"  - {h} → {c} occurrences")
        else:
            logging.info(f"✅ No duplicates found in {collection_name}.")

        return duplicates

    @staticmethod
    def audit_all_collections(collection_names: List[str]) -> Dict[str, Dict[str, int]]:
        """
        Audit all collections for duplicates.

        Args:
            collection_names: List of collection names to audit

        Returns:
            Dictionary mapping collection names to their duplicate reports
        """
        all_duplicates = {}

        for collection_name in collection_names:
            duplicates = DuplicateAuditor.audit_collection(collection_name)
            if duplicates:
                all_duplicates[collection_name] = duplicates

        return all_duplicates
