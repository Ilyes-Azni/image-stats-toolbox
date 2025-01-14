from src.utils.image_loader import ImageLoader
import logging
from pprint import pprint

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

if __name__ == "__main__":
    logger.info("Initializing ImageLoader...")
    loader = ImageLoader(
        root_path=r"C:\Users\PC\Desktop\données\deepfake_database",
        extensions={".jpg", ".jpeg", ".png"},
        recursive=True
    )

    # Print class information
    logger.info("Available classes:")
    pprint(loader.get_class_names())

    logger.info("\nClass distribution:")
    pprint(loader.get_class_distribution())

    logger.info("\nDetailed dataset statistics:")
    pprint(loader.get_dataset_stats())

    logger.info("\nDataset validation results:")
    pprint(loader.validate_dataset())
