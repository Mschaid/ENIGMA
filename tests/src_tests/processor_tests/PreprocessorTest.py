import logging
import os
import pandas as pd
import shutil
import unittest

from src.processors.Preprocessor import Preprocessor
from src.utilities.logger_helpers import LoggingTestRunner

# Globals for tests
FEATURES = ['day', 'time', 'trial',
            'event_cue', 'event_shock',
            'sensor_D1', 'sensor_D2', 'sensor_DA']
ENCODED_FEATURES = ['event', 'sensor']
TARGET = 'signal'
TEST_DATA_DIR = '/home/mds8301/gaby_test'
test_data = os.path.join(TEST_DATA_DIR, 'aggregated_data.parquet.gzp')
test_processed_data = os.path.join(TEST_DATA_DIR, 'processed_data.parquet.gzp')

class TestPreprocessor(unittest.TestCase):
    """
    Unit tests for the Preprocessor class.

        Attributes
        ----------
        processor : Preprocessor
            An instance of the Preprocessor class used for testing.
        new_processor : Preprocessor
            An instance of the Preprocessor class used for testing the load_processor method.

        Methods
        -------
        setUp() -> None
            Initializes the processor instance and loads the data.
        test_load_data() -> None
            Test if the load_data() method loads data into the processor object.
        test_one_hot_encode(labels=ENCODED_FEATURES) -> None
            Test if the one_hot_encode() method properly one-hot encodes categorical features.
        test_split_train_test() -> None
            Test if the split_train_test() method properly splits data into train and test sets.
        test_split_train_by_query(column_name:str, cutoff:int) -> None
            Test if the split_train_by_query() method properly splits data by a cutoff date.
        test_save_processed_data() -> None
            Test if the save_processor() method properly saves the processed processor object.
        tearDown() -> None
            Removes the directory where saved processors are stored after testing is complete.
        """
        

    def setUp(self)->None:
        """
        Set up the test fixture.

        Initializes the processor instance and loads the data.
        """
        self.processor = Preprocessor(path_to_data = test_data, path_to_processed_data= test_processed_data,
                         features=FEATURES,
                         target=TARGET, 
                         processor_name='unit_test_processor')
        self.processor.load_data()
    

    def test_load_data(self) -> None:
        """
       Test the one_hot_encode() method.

        Test if the one_hot_encode() method properly one-hot encodes categorical features.
        Test that the dummy_data attribute is of type pd.DataFrame.
        
        Parameters
        ----------
        labels : list of str, optional
            The feature names to use. Defaults to ENCODED_FEATURES.
        """
        self.assertIsInstance(self.processor.data, pd.DataFrame)
        self.assertIsInstance(self.processor.processed_data, pd.DataFrame)
        
    def test_one_hot_encode(self)->None:
        """ 
        Test if the one_hot_encode() method properly one-hot encodes categorical features.

        Parameters
        ----------
        labels : list of str, optional
            The feature names to use. Defaults to ENCODED_FEATURES.

        Raises
        ------
        AssertionError
            If the one_hot_encode() method fails to add the encoded features to the processed_data attribute,
            or if the dummy_data attribute is not of type pd.DataFrame.
        """
        self.processor.one_hot_encode(labels = ENCODED_FEATURES)
        expected_cols = ['mouse_id', 'day', 'time', 'trial', 'signal', 'event_avoid',
       'event_cue', 'event_escape', 'event_shock', 'sensor_D1', 'sensor_D2',
       'sensor_DA']
        for col in expected_cols:
            with self.subTest(col=self.processor.processed_data.columns):
                self.assertTrue(col in expected_cols) 
                
        self.assertIsInstance(self.processor.processed_data, pd.DataFrame)
        
    def test_split_train_test(self)->None:
        """
        Tests if the split_train_test() method properly splits data into train and test sets.

        Parameters:
        ----------
        self: instance of the class

        Returns:
        ----------
        None

        Raises:
        ----------
        AssertionError: if any of the tests fail

        Notes:
        ----------
        - This function tests if the X_train, y_train, X_test, and y_test attributes are of type pd.DataFrame and pd.Series.
        - This function also tests if the X_train and X_test attributes have the correct shape.
        """
        
        self.processor.split_train_test(processed_data=True)


        expected_x_train_shape = (
            (int(self.processor.processed_data[FEATURES].shape[0]*0.8)),
            int(self.processor.processed_data[FEATURES].shape[1])
                               )
        self.assertEqual(self.processor.X_train.shape, expected_x_train_shape)
        self.assertIsInstance(self.processor.X_train, pd.DataFrame)
        self.assertIsInstance(self.processor.y_train, pd.Series)
        self.assertIsInstance(self.processor.X_test, pd.DataFrame)
        self.assertIsInstance(self.processor.y_test, pd.Series)
        

    def test_split_train_by_query(self) -> None:
        """
        Summary
        ----------
        test if the split_train_by_query() method properly splits data by a cutoff .
        Parameters
        ----------
        self: instance of the class
        
        Returns
        ---------- 
        None
        
        Raises
        ----------
        AssertionError: if any of the tests fail
        
 
        """


        self.processor.split_train_by_query(column_name='day', cutoff=7, processed_data=True)
        self.assertIsInstance(self.processor.X_train, pd.DataFrame)
        self.assertIsInstance(self.processor.y_train, pd.Series)
        self.assertIsInstance(self.processor.X_test, pd.DataFrame)
        self.assertIsInstance(self.processor.y_test, pd.Series)

    # def test_save_dataframes_to_parquet(self):
        
    def test_save_processed_data(self): 
        
        
        """
        Summary
        ----------
        test if the save_processor() method properly saves the processed processor object.
        
        Parameters    
        ----------  
        self: instance of the class
        
        Returns
        ---------- 
        None
        
        Raises
        ----------
        AssertionError: if any of the tests fail
        """
        expected_processor_dir = os.path.join(TEST_DATA_DIR, 'processors')
        expected_file_path = os.path.join(expected_processor_dir, 'unit_test_processor.pkl')

        self.processor.save_processor()
        self.assertEqual(os.path.dirname(self.processor.path_to_processed_data), TEST_DATA_DIR)
        self.assertTrue(os.path.exists(expected_processor_dir))
        self.assertTrue(os.path.exists(expected_file_path))
        

    def test_load_processed_data(self):
        """ 
        Parameters
        ----------
        Returns
        None
        ----------
        Raises
        AssertError: If any of the tests fail
        ----------
        Notes:
        ----------
        """
        def remove_test_dir(path):
            """ 
            Removes the directory where saved processors are stored after testing is complete.
            """
            if os.path.exists(path):
                shutil.rmtree(path)
            else:
                pass
        self.processor.save_processor()
        path_to_processor = os.path.join(TEST_DATA_DIR, 'processors', 'unit_test_processor.pkl')
        self.new_processor = Preprocessor.load_processor(path_to_processor)
        self.assertIsInstance(self.new_processor, Preprocessor)
        remove_test_dir(os.path.join(TEST_DATA_DIR, 'processors'))



if __name__=="__main__":
    # logging configuration
    TEST_LOG_DIR = '/projects/p31961/dopamine_modeling/tests/test_logs'
    TEST_LOG_FILE = 'test_preprocessor.log'
    FULL_TEST_LOG_DIR = os.path.join(TEST_LOG_DIR, TEST_LOG_FILE)

    # create logger
    logging.basicConfig(filename=FULL_TEST_LOG_DIR, 
                        level=logging.INFO,
                        format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                        datefmt='%d-%b-%y %H:%M:%S')

    with open(FULL_TEST_LOG_DIR, 'w') as f:
        runner = LoggingTestRunner(verbosity=2)
        unittest.main(testRunner=runner)

    # run tests and log output

