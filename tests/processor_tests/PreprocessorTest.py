import os
import pandas as pd
import shutil
import unittest


from src.processors.Preprocessor import Preprocessor

FEATURES = ['day', 'time', 'trial',
            'event_cue', 'event_shock',
            'sensor_D1', 'sensor_D2', 'sensor_DA']
ENCODED_FEATURES = ['day', 'trial','event_cue',
                    'event_shock', 'sensor_D1', 'sensor_D2',
                    'sensor_DA']
TARGET = 'signal'
TEST_DATA_DIR = '/home/mds8301/gaby_test'
test_data = os.path.join(TEST_DATA_DIR, 'testing_data.parquet.gzp')
test_processed_data = os.path.join(TEST_DATA_DIR, 'processed_data.parquet.gzp')


class TestPreprocessor(unittest.TestCase):
    
    """ # Summary
        --- 
        A test suite for the Preprocessor class.

        ## Attributes:
        -----------
        - processor : Preprocessor
            An instance of the Preprocessor class used for testing.
        - new_processor : Preprocessor
            An instance of the Preprocessor class used for testing the load_processor method.

        ## Methods:
        --------
        setUp() -> None:
            Initializes the processor instance and loads the data.
        test_load_data():
            Tests if the load_data() method loads data into the processor object.
        test_one_hot_encode(labels=ENCODED_FEATURES):
            Tests if the one_hot_encode() method properly one-hot encodes categorical features.
        test_split_train_test():
            Tests if the split_train_test() method properly splits data into train and test sets.
        test_split_train_by_query(column_name:str, cutoff:int):
            Tests if the split_train_by_query() method properly splits data by a cutoff date.
        test_save_processed_data():
            Tests if the save_processor() method properly saves the processed data.
        test_load_processed_data():
            Tests if the load_processor() method properly loads a saved processor object.
        tearDown():
            Removes the directory where saved processors are stored after testing is complete.
    """
    

    def setUp(self)->None:
        """
        Initializes the processor instance and loads the data.
        """
        self.processor = Preprocessor(path_to_data = test_data, path_to_processed_data= test_processed_data,
                         features=FEATURES,
                         target=TARGET, 
                         processor_name='unit_test_processor')
        self.processor.load_data()
        
    def test_load_data(self):
        """
        Tests if the load_data() method loads data into the processor object.
        tests that the data and processed_data attributes are of type pd.DataFrame.
        """
        self.assertIsInstance(self.processor.data, pd.DataFrame)
        self.assertIsInstance(self.processor.processed_data, pd.DataFrame)
        
    def test_one_hot_encode(self, labels=ENCODED_FEATURES):
        """ 
        - Tests if the one_hot_encode() method properly one-hot encodes categorical features.
        - Tests that the one_hot_encode() method adds the encoded features to the processed_data attribute.
        - Tests that the dummy_data attribute is of type pd.DataFrame.

        Args:
            labels (list: optional): feature names to use. Defaults to ENCODED_FEATURES.
        """
        self.processor.one_hot_encode(*ENCODED_FEATURES)
        for label in labels:
            with self.subTest(label=label):
                self.assertTrue(label in self.processor.processed_data.columns) 
                
        self.assertIsInstance(self.processor.dummy_data, pd.DataFrame)
        
    def test_split_train_test(self):
        """
        - Tests if the split_train_test() method properly splits data into train and test sets.
        - Tests that the X_train, y_train, X_test, and y_test attributes are of type pd.DataFrame and pd.Series.
        - Tests that the X_train and X_test attributes have the correct shape.
        """
        self.processor.split_train_test()

        expected_x_train_shape = (
            (int(self.processor.processed_data[FEATURES].shape[0]*0.8)),
            int(self.processor.processed_data[FEATURES].shape[1])
                               )
        self.assertEqual(self.processor.X_train.shape, expected_x_train_shape)
        self.assertIsInstance(self.processor.X_train, pd.DataFrame)
        self.assertIsInstance(self.processor.y_train, pd.Series)
        self.assertIsInstance(self.processor.X_test, pd.DataFrame)
        self.assertIsInstance(self.processor.y_test, pd.Series)
        
    def test_split_train_by_query(self):
        """ 
        - Tests if the split_train_by_query() method properly splits data by a cutoff valu.
        - Tests that the X_train, y_train, X_test, and y_test attributes are of type pd.DataFrame and pd.Series.
        """
        
        
        self.processor.split_train_by_query(column_name='day', cutoff=7)
        self.assertIsInstance(self.processor.X_train, pd.DataFrame)
        self.assertIsInstance(self.processor.y_train, pd.Series)
        self.assertIsInstance(self.processor.X_test, pd.DataFrame)
        self.assertIsInstance(self.processor.y_test, pd.Series)

        
    def test_save_processed_data(self):
        """ 
        Tests if the save_processor() method properly saves the processed processor object.
        Tests that the path_to_processed_data attribute is set to the correct directory.
        Tests that the directory where the processor is saved exists.
        Tests that the file where the processor is saved exists. 
        """
        expected_processor_dir = os.path.join(TEST_DATA_DIR, 'processors')
        expected_file_path = os.path.join(expected_processor_dir, 'unit_test_processor.pkl')

        self.processor.save_processor()
        self.assertEqual(os.path.dirname(self.processor.path_to_processed_data), TEST_DATA_DIR)
        self.assertTrue(os.path.exists(expected_processor_dir))
        self.assertTrue(os.path.exists(expected_file_path))
        
        
    def test_load_processed_data(self):
        """ 
        Tests if the load_processor() method properly loads a saved processor object.
        Tests that the new_processor attribute is of type Preprocessor.
        
        """
        
        self.processor.save_processor()
        path_to_processor = os.path.join(TEST_DATA_DIR, 'processors', 'unit_test_processor.pkl')
        self.new_processor = Preprocessor.load_processor(path_to_processor)
        self.assertIsInstance(self.new_processor, Preprocessor)
        

    # @classmethod 
    # def tearDown(cls):
    #     """ 
    #     if the directory where saved processors are stored exists, remove it.
    #     """
    #     shutil.rmtree(os.path.join(TEST_DATA_DIR, 'processors'))
if __name__=="__main__":
    unittest.main(verbosity=2)