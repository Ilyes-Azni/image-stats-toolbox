from pathlib import Path
from typing import List, Set, Dict, Optional
import os

class ImageLoader:
    def __init__(
            self,
            root_path:str,
            extensions:Set[str]={".jpg", ".jpeg", ".png"},
            recursive:bool=True
    ):
        self.root_path = Path(root_path)
        self.extensions = extensions
        self.recursive = recursive
        self.dataset_index={}
        self._scan_directory()

    def _scan_directory(self):
        """ Scans directory and build a dataset index """
        for ext in self.extensions:
            pattern = f"**/*{ext}" if self.recursive else f"*{ext}"
            self.dataset_index[ext] = [

                str(p) for p in self.root_path.glob(pattern)
            ]

    def get_batch(self, batch_size: int = 32, extension: Optional[str] = None) -> List[str]:
        """Returns a batch of image paths"""
        paths = []
        if extension:
            paths.extend(self.dataset_index.get(extension, []))
        else:
            for ext_paths in self.dataset_index.values():
                paths.extend(ext_paths)
        return paths[:batch_size]

    def validate_dataset(self) -> Dict[str, int]:
        """Validates dataset and returns statistics"""
        stats = {
            'total_files': 0,
            'valid_files': 0,
            'invalid_files': 0
        }
        
        for ext_paths in self.dataset_index.values():
            for path in ext_paths:
                stats['total_files'] += 1
                if self.is_valid_image(path):
                    stats['valid_files'] += 1
                else:
                    stats['invalid_files'] += 1
        
        return stats

    def get_dataset_stats(self) -> Dict[str, int]:
        """Returns dataset statistics"""
        return {
            'total_images': sum(len(paths) for paths in self.dataset_index.values()),
            'extensions': {ext: len(paths) for ext, paths in self.dataset_index.items()}
        }

    @staticmethod
    def is_valid_image(file_path: str) -> bool:
        """Checks if file exists and has non-zero size"""
        path = Path(file_path)
        return path.exists() and path.stat().st_size > 0

    def get_subset(self, start_idx: int = 0, end_idx: Optional[int] = None) -> List[str]:
        """Returns a subset of dataset paths"""
        all_paths = []
        for ext_paths in self.dataset_index.values():
            all_paths.extend(ext_paths)
        
        end_idx = end_idx or len(all_paths)
        return all_paths[start_idx:end_idx]
    

# Example usage:
"""  
loader = ImageLoader(
    root_path="/path/to/dataset",
    extensions={'.jpg', '.png'},
    recursive=True
)

# Get dataset statistics
stats = loader.get_dataset_stats()

# Get a batch of images
batch_paths = loader.get_batch(batch_size=32)

# Validate dataset
validation_results = loader.validate_dataset()

# Get specific subset
subset = loader.get_subset(start_idx=100, end_idx=200)
"""