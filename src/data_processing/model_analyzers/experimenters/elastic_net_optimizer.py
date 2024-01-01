from pathlib import Path


from src.data_processing.processors.guppy_processors.config_loader import ConfigLoader


class ElasticNetOptimizer:
    def __init__(self, experiment_path: Path):
        self._experiment_path: Path = experiment_path
        self._configs: ConfigLoader = None
        self.experimental_params = None

    @property
    def experiment_path(self):
        if not isinstance(self._experiment_path, Path):
            self._experiment_path = Path(self._experiment_path)
        return self._experiment_path

    @property
    def configs(self):
        if not self._configs:
            self._configs = self._fetch_configs()
        return self._configs

    def _fetch_configs(self):
        if not self._configs:
            for f in self.experiment_path.iterdir():
                if f.is_file() and f.suffix == '.yaml':
                    self._configs = ConfigLoader(f)
        return self._configs

    @property
    def initial_params(self):
        return self.configs.config_data['best_params']

    def set_experimental_params(self, new_params=None):
        if new_params is None:
            self.experimental_params = None
            return self.initial_params
        else:
            self.experimental_params = self.initial_params.copy()
            for k, v in new_params.items():
                self.experimental_params.update({k: v})
            return self.experimental_params
