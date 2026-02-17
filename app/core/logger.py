import logging
import sys

def setup_logger():
    # Create a custom logger
    logger = logging.getLogger("app_logger")
    
    # Set the threshold to INFO (Captures INFO, WARNING, ERROR, CRITICAL)
    logger.setLevel(logging.INFO)

    # Define the format for logs
    log_format = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # 1. Console Handler: Prints logs to your terminal
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(log_format)
    logger.addHandler(console_handler)

    # 2. File Handler: Saves logs to a file named 'app.log'
    file_handler = logging.FileHandler('app.log')
    file_handler.setFormatter(log_format)
    logger.addHandler(file_handler)

    return logger

# Initialize the logger instance
logger = setup_logger()