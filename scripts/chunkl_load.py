from src.processors.Preprocessor import Preprocessor
import pandas as pd



# test.y_train = pd.Series([1, 2, 3])

path_to_save = '/projects/p31961/dopamine_modeling/data/gaby_test/test_preprocessor.json'
test = Preprocessor().load_processor_json(path_to_save)
    
if __name__ == '__main__':
    print(test.X_train)
    


