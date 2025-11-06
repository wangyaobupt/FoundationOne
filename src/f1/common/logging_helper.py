import time
import os
import logging

def log_execution_time(func):
    def wrapper(self, *args, **kwargs):
        start_time = time.time()
        result = func(self, *args, **kwargs)
        end_time = time.time()
        execution_time = end_time - start_time
        self.logger.debug(f"Execution of {self.__class__.__name__}.{func.__name__} took {execution_time:.3f} seconds")
        return result
    return wrapper

def configure_module_logging(module_names_list:list[str]=None, level=logging.DEBUG, log_dir='./log'):
    """
    Configure logging with dynamic log file paths.

    Args:
        module_names_list (list, optional): module names to configure logging for, if empty, apply to all modules
        level (int): Logging level (default: logging.DEBUG)
        log_dir: dir path to store logfile
    """
    # Ensure log directory exists
    os.makedirs(log_dir, exist_ok=True)

    # Formatter for log messages
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # If specific module names are provided, create file handlers for each
    if module_names_list:
        for module_name in module_names_list:
            # Create logger for specific module
            module_logger = logging.getLogger(module_name)
            module_logger.setLevel(level)

            # Console handler for all loggers
            console_handler = logging.StreamHandler()
            console_handler.setFormatter(formatter)
            module_logger.addHandler(console_handler)

            # Create log filename based on module name
            # Replace dots with underscores to create valid filename
            safe_module_name = module_name.split(".")[-1]
            log_filename = f'{safe_module_name}.log'
            log_path = os.path.join(log_dir, log_filename)

            # Create file handler for the module
            file_handler = logging.FileHandler(log_path, mode='w')
            file_handler.setFormatter(formatter)
            module_logger.addHandler(file_handler)
    else:
        # Configure root logger
        root_logger = logging.getLogger()
        root_logger.setLevel(level)

        # Close and remove any existing handlers to prevent duplicate logging and resource warnings
        for handler in root_logger.handlers:
            handler.close()
        root_logger.handlers.clear()

        # Console handler for all loggers
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        root_logger.addHandler(console_handler)

        # Default log file for root logger
        default_log_path = os.path.join(log_dir, 'output.log')
        default_file_handler = logging.FileHandler(default_log_path, mode='w+')
        default_file_handler.setFormatter(formatter)
        root_logger.addHandler(default_file_handler)


