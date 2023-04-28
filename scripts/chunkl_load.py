from src.processors.Preprocessor import Preprocessor, JSONProcessor

path = f'/projects/p31961/gaby_data/aggregated_data/processors/5_day_training_gaby.pkl'
proc = Preprocessor().load_processor_json(path)


