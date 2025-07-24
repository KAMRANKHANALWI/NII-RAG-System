"""Configuration settings for the vector indexer system."""

import os

# Base paths
BASE_PATH = "../collections"
VECTOR_DB_DIR = "../vectorstores"

# Collection configuration mapping
COLLECTION_CONFIG = {
    "faculty_info": "FacultyInfo",
    "research": "Research",
    "publications": "Publications",
    "labs": "Lab",
    "nii_info": "BasicInfo",
    "staff": "Staff",
    "programs_courses": "Programs",
    "recruitments": "Announcement",
    "magazine": "Magazine",
}

# Embedding model configuration
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

# Logging configuration
LOG_LEVEL = "INFO"
LOG_FORMAT = "%(levelname)s: %(message)s"
