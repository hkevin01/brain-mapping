"""
Centralized error handling utilities for the brain mapping toolkit.
"""
import logging


class ErrorHandler:
    """Handles errors and exceptions across modules."""
    def __init__(self):
        logging.info("ErrorHandler initialized")

    def handle(self, error: Exception, context: str = ""):
        logging.error("Error in %s: %s", context, str(error))
        # Optionally, raise or log to external system
        return {"context": context, "error": str(error)}
