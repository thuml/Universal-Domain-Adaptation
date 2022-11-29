import logging
import os


def logger_init(logger, output_dir, save_as_file=True):
    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO
    )

    logger.setLevel(logging.INFO)

    if not os.path.isdir(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    if save_as_file: 
        logging_output_file = os.path.join(output_dir, "output.log")
        file_formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(name)s - %(message)s")
        file_handler = logging.FileHandler(logging_output_file)
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)



def print_dict(logger, string, dict):
    logger.info(string)
    for k,v in dict.items():
        logger.info(f'{k} > {v}')
    logger.info('\n\n')