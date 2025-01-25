from src.utils.image_loader import ImageLoader
from src.utils.visualisation import DatasetVisualizer
from src.outliers.mahalanobis import mahalanobis
import logging
from pprint import pprint

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

if __name__ == "__main__":
    # Initial dataset loading and visualization
    logger.info("Initializing ImageLoader...")
    loader = ImageLoader(
        root_path=r"C:\Users\PC\Desktop\données\deepfake_database",
        extensions={".jpg", ".jpeg", ".png"},
        recursive=True
    )

    # Print class information
    logger.info("Available classes:")
    pprint(loader.get_class_names())

    logger.info("\nInitial class distribution:")
    initial_distribution = loader.get_class_distribution()
    pprint(initial_distribution)

    # Visualize initial dataset
    logger.info("\nVisualizing initial dataset...")
    DatasetVisualizer.viz(loader, images_per_class=5)

    # Outlier detection and visualization
    logger.info("\nDetecting outliers...")
    detector = mahalanobis(loader, class_name="df")
    outliers = detector.detect(threshold=2.0)
    outlier_paths = detector.get_outlier_paths()
    
    logger.info(f"Number of outliers detected: {len(outlier_paths)}")
    
    # Visualize detected outliers
    logger.info("\nVisualizing detected outliers...")
    detector.visualize_outliers(num_samples=5)

    # Remove outliers and verify changes
    logger.info("\nRemoving outliers from dataset...")
    loader.remove_outliers(outlier_paths, move_to=r"C:\Users\PC\Desktop\données\deepfake_database\outliers_backup")

    logger.info("\nUpdated class distribution after outlier removal:")
    final_distribution = loader.get_class_distribution()
    pprint(final_distribution)

    # Show final dataset state
    logger.info("\nVisualizing final dataset...")
    DatasetVisualizer.viz(loader, images_per_class=5)

    # Print summary of changes
    logger.info("\nSummary of changes:")
    for class_name in loader.get_class_names():
        initial = initial_distribution.get(class_name, 0)
        final = final_distribution.get(class_name, 0)
        diff = initial - final
        logger.info(f"{class_name}: {initial} -> {final} (removed {diff} images)")
