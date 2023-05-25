from src.processors.Preprocessor import Preprocessor

PATH = '/home/mds8301/gaby_test/processors/unit_test_processor.pkl'
proc = Preprocessor().load_processor(PATH)
proc.one_hot_encode(labels = ['event', 'sensor'])
proc.split_train_by_query('trial', 20, processed_data=True)
proc.path_to_save_processor = '/home/mds8301/gaby_test/processors/processed_unit_test_processor.pkl'
proc.save_processor()
if __name__=='__main__':
    print(proc.X_train)
    print(proc.X_test)