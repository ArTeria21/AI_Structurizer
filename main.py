import logging
from src.manager import Manager

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,  # Установите уровень на DEBUG для подробных логов
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler("app.log"),
            logging.StreamHandler()
        ]
    )

if __name__ == '__main__':
    setup_logging()
    logger = logging.getLogger(__name__)
    logger.info("Starting the text structuring service")

    input_folder = 'unstructured'
    output_folder = 'structured'
    try:
        manager = Manager(input_folder=input_folder, output_folder=output_folder)
        manager.process_files()
        logger.info("Processing completed successfully")
    except Exception as e:
        logger.exception("An error occurred during processing")
