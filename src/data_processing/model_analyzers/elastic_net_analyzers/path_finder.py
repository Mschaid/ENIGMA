import re
from typing import Dict
from pathlib import Path


class PathFinder:
    def __init__(self, main_path: Path):
        self.main_path = main_path


    @property
    def metrics_paths(self):
        paths = [p for p in self.main_path.glob(
            '**/*') if p.stem.startswith('metric')]
        return paths

    @property
    def feature_importance_paths(self):
        paths = [p for p in self.main_path.glob(
            '**/*') if p.stem.startswith('feature')]
        return paths
