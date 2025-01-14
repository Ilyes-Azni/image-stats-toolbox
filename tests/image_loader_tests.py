from src.utils.image_loader import ImageLoader
import logging
from pprint import pprint

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

if __name__ == "__main__":
    # Initialize loader with logging
    logger.info("Initializing ImageLoader...")
    loader = ImageLoader(
        root_path=r"C:\Users\PC\Desktop\données\deepfake_database",
        extensions={".jpg", ".jpeg", ".png"},
        recursive=True
    )

    # Get and display stats
    logger.info("Collecting dataset statistics...")
    stats = loader.get_dataset_stats()
    
    logger.info("Dataset Statistics:")
    pprint(stats)

    # Additional validation
    logger.info("Running dataset validation...")
    validation_results = loader.validate_dataset()
    
    logger.info("Validation Results:")
    pprint(validation_results)
