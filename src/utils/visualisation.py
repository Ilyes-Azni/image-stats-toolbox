import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import numpy as np
from PIL import Image
import random
import time
from typing import List, Union

class DatasetVisualizer:
    @staticmethod
    def viz(loader, images_per_class: int = 5):
        """
        Creates an organized carousel visualization from an ImageLoader instance with random sampling
        
        Args:
            loader: ImageLoader instance containing the dataset
            images_per_class: Number of images to display per class
        """
        random.seed(time.time())
        
        class_names = loader.get_class_names()
        total_images = len(class_names) * images_per_class
        
        fig = plt.figure(figsize=(15, 3 * len(class_names)))
        gs = GridSpec(len(class_names), images_per_class, figure=fig)
        
        for row, class_name in enumerate(class_names):
            class_images = loader.get_images_by_class(class_name)
            sampled_images = random.sample(class_images, min(images_per_class, len(class_images)))
            
            for col, img_path in enumerate(sampled_images):
                ax = fig.add_subplot(gs[row, col])
                img = Image.open(img_path)
                ax.imshow(img)
                # Add label under each image
                ax.set_title(class_name, pad=10)
                ax.axis('off')
        
        plt.suptitle("Dataset Overview", fontsize=16, y=1.02)
        plt.tight_layout()
        plt.show()
