from pathlib import Path
from typing import List, Set, Dict, Optional
from collections import defaultdict

class ImageLoader:
    def __init__(
        self,
        root_path: str,
        extensions: Set[str] = {'.jpg', '.jpeg', '.png'},
        recursive: bool = True
    ):
        self.root_path = Path(root_path)
        self.extensions = extensions
        self.recursive = recursive
        self.dataset_index = {}
        self.class_mapping = {}
        self.class_statistics = {}
        self._scan_directory()
        self.map_class_folders()

    def _scan_directory(self) -> None:
        """Scans directory and builds dataset index"""
        for ext in self.extensions:
            pattern = f"**/*{ext}" if self.recursive else f"*{ext}"
            self.dataset_index[ext] = [
                str(p) for p in self.root_path.glob(pattern)
            ]

    def map_class_folders(self) -> None:
        """Maps class folders and organizes images by class"""
        self.class_mapping = defaultdict(list)
        
        for ext_paths in self.dataset_index.values():
            for path in ext_paths:
                path_obj = Path(path)
                # Get the immediate parent folder as class name
                class_name = path_obj.parent.name
                self.class_mapping[class_name].append(str(path))
        
        # Convert defaultdict to regular dict
        self.class_mapping = dict(self.class_mapping)
        self._update_class_statistics()

    def _update_class_statistics(self) -> None:
        """Updates statistics for each class"""
        self.class_statistics = {
            class_name: {
                'count': len(paths),
                'extensions': self._count_extensions(paths)
            }
            for class_name, paths in self.class_mapping.items()
        }

    def _count_extensions(self, paths: List[str]) -> Dict[str, int]:
        """Counts file extensions in a list of paths"""
        ext_count = defaultdict(int)
        for path in paths:
            ext = Path(path).suffix.lower()
            ext_count[ext] += 1
        return dict(ext_count)

    def get_class_distribution(self) -> Dict[str, int]:
        """Returns the distribution of images across classes"""
        return {
            class_name: stats['count']
            for class_name, stats in self.class_statistics.items()
        }

    def get_images_by_class(self, class_name: str) -> List[str]:
        """Returns all image paths for a specific class"""
        return self.class_mapping.get(class_name, [])

    def get_class_names(self) -> List[str]:
        """Returns list of all class names"""
        return list(self.class_mapping.keys())

    def get_dataset_stats(self) -> Dict:
        """Returns comprehensive dataset statistics"""
        return {
            'total_images': sum(len(paths) for paths in self.dataset_index.values()),
            'extensions': {ext: len(paths) for ext, paths in self.dataset_index.items()},
            'class_distribution': self.get_class_distribution(),
            'class_statistics': self.class_statistics
        }

    def get_batch(self, batch_size: int = 32, class_name: Optional[str] = None) -> List[str]:
        """Returns a batch of image paths, optionally from a specific class"""
        if class_name:
            paths = self.get_images_by_class(class_name)
        else:
            paths = []
            for ext_paths in self.dataset_index.values():
                paths.extend(ext_paths)
        return paths[:batch_size]

    def validate_dataset(self) -> Dict[str, int]:
        """Validates dataset and returns statistics"""
        stats = {
            'total_files': 0,
            'valid_files': 0,
            'invalid_files': 0,
            'class_validation': {}
        }
        
        for class_name, paths in self.class_mapping.items():
            class_stats = {'valid': 0, 'invalid': 0}
            for path in paths:
                if self.is_valid_image(path):
                    class_stats['valid'] += 1
                else:
                    class_stats['invalid'] += 1
            stats['class_validation'][class_name] = class_stats
            stats['valid_files'] += class_stats['valid']
            stats['invalid_files'] += class_stats['invalid']
            stats['total_files'] += len(paths)
        
        return stats

    @staticmethod
    def is_valid_image(file_path: str) -> bool:
        """Checks if file exists and has non-zero size"""
        path = Path(file_path)
        return path.exists() and path.stat().st_size > 0
