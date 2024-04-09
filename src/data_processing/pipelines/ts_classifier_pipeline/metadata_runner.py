import logging
import pretty_errors
from pathlib import Path
from typing import List
from src.data_processing.pipelines import AAMetaDataFetcher, meta_data_factory, directory_finder

# set up logger for metadata runner
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setLevel(logging.INFO)
formatter = logging.Formatter(
    "%(asctime)s - %(name)s - %(levelname)s - %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)


def main(PATH):
    main_path = Path(PATH)
    logger.info(f"Looking for directories in {main_path}")

    files = directory_finder(main_path, "output")
    logger.info(f"Found {len(files)} directories")

    fetchers = [meta_data_factory(file, AAMetaDataFetcher) for file in files]
    logger.info(f"Created {len(fetchers)} fetchers")

    list(map(lambda fetcher: fetcher.save_metadata_to_yaml(), fetchers))
    logger.info("Saved metadata to yaml files")


if __name__ == "__main__":
    PATH = "/Volumes/fsmresfiles/Basic_Sciences/Phys/Lerner_Lab_tnl2633/Gaby/Data Analysis/ActiveAvoidance/Core_guppy_postcross"
    main(PATH)
