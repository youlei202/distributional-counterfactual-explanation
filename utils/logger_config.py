import sys  # Import sys to get access to stdout

# Set up logging configurations
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Create a StreamHandler for stdout
stdout_handler = logging.StreamHandler(sys.stdout)
stdout_handler.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
stdout_handler.setFormatter(formatter)

# Add the handler to logger
logger.addHandler(stdout_handler)
