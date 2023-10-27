import logging
import sys

def setup_logger():
    # Set up logging configurations
    logger = logging.getLogger(__name__)

    if not logger.handlers:  # Check if handlers are already added
        logger.setLevel(logging.INFO)

        stdout_handler = logging.StreamHandler(sys.stdout)
        stdout_handler.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        stdout_handler.setFormatter(formatter)

        logger.addHandler(stdout_handler)

    return logger

