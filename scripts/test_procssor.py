from src.processors.Preprocessor import Preprocessor

PATH = '/projects/p31961/gaby_data/aggregated_data/processors/5_day_training_gaby.pkl'
proc = Preprocessor().load_processor(PATH)

if __name__=='__main__':
    print(proc.X_train)