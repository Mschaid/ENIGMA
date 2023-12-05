

from src.data_processing.processors.pavlovian_processors.DataLoader import DataLoader
from src.data_processing.processors.pavlovian_processors.PavProcessor import PavProcessor
from src.data_processing.processors.pavlovian_processors.PavVisualizer import PavVisualizer

def run_analysis():
    CONFIG_PATH = '/Volumes/fsmresfiles/Basic_Sciences/Phys/Lerner_Lab_tnl2633/Shudi/LHA_dopamine/conf/config.yaml'
    data_loader = DataLoader(CONFIG_PATH)
    # pav_processor = PavProcessor(meta_data)

    data_loader.fetch_data_files(
        keywords=data_loader.meta_data['search_keys']['keywords'], file_extensions=data_loader.meta_data['search_keys']['file_types'])
    for file in data_loader.data_files:
        print(file)

if __name__ == '__main__':
    run_analysis()
