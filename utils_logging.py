import os
import sys
import logging
import shutil


def setup_logging(log_dir):
    """Setup logging format and output to file"""
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    log_file = os.path.join(log_dir, "imputation.log")

    # Create logger object
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # Clear existing handlers
    if logger.handlers:
        logger.handlers.clear()

    # Add console output handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_format = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(console_format)
    logger.addHandler(console_handler)

    # Add file output handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    file_format = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(file_format)
    logger.addHandler(file_handler)

    return logger


def copy_code_folder(source_dir, target_dir):
    """Copy code folder to code subfolder under target folder"""
    logging.info("Starting to backup code folder...")

    # Create target code folder
    code_backup_dir = os.path.join(target_dir, "code")
    if not os.path.exists(code_backup_dir):
        os.makedirs(code_backup_dir)

    # Find project root directory
    if source_dir is None:
        # If no source directory provided, find directory of calling script
        calling_script = sys.argv[0]
        source_dir = os.path.dirname(os.path.abspath(calling_script))

    # List of files and folders to copy
    items_to_copy = []

    # Traverse project root directory
    for item in os.listdir(source_dir):
        source_item = os.path.join(source_dir, item)

        # Exclude target folder itself and its parent directory
        if os.path.abspath(target_dir).startswith(os.path.abspath(source_item)):
            continue

        # Exclude unnecessary files (like virtual environments, cache files, etc.)
        if item.startswith('.') or item.startswith('__pycache__') or \
                item.startswith('venv') or item.startswith('env') or \
                item.startswith('node_modules'):
            continue

        items_to_copy.append((source_item, os.path.join(code_backup_dir, item)))

    # Copy files and folders
    for source, dest in items_to_copy:
        if os.path.isdir(source):
            logging.info(f"Copying directory: {source} -> {dest}")
            shutil.copytree(source, dest, ignore=shutil.ignore_patterns('*.pyc', '__pycache__', '.git*'))
        else:
            logging.info(f"Copying file: {source} -> {dest}")
            shutil.copy2(source, dest)

    logging.info(f"Code backup completed, saved in: {code_backup_dir}")
    return code_backup_dir