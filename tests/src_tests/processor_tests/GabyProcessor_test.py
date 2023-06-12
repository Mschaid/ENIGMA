import logging
import os
import pandas as pd
import unittest

from src.processors.GabyProcessor import GabyProcessor
from src.utilities.logger_helpers import LoggingTestRunner


TEST_PATH =r'/home/mds8301/gaby_test/Day7/142-237_Day7_Avoid_D1_z_score_D1.h5'

class TestGabyProcessor(unittest.TestCase):
    """ #Summary
    Testing for Gaby Processor

    """
    
    @classmethod
    def setUp(self) -> None:
        """# Summary
        
        initializes gaby processor as class method to be used in all functions
        
        """
        self.processor = GabyProcessor(path=TEST_PATH)
        self.processor.get_meta_data()
        self.processor.get_data()
        
    def test_get_meta_data(self):
        
        self.assertEqual(self.processor.path, TEST_PATH) #tests path attribute  =  TEST_PATH
        self.assertIsInstance(self.processor.meta_data, dict) # meta data is dictionary 
        self.assertIsInstance(self.processor.meta_data['mouse_id'], str)  #moue id is string
        self.assertEqual(self.processor.meta_data['mouse_id'], '142_237') # mouse id is correct
        self.assertIsInstance(self.processor.meta_data['day'], int) # day is int
        self.assertEqual(self.processor.meta_data['day'], 7) # day is correct
        self.assertIsInstance(self.processor.meta_data['event'], str) # event is string
        self.assertEqual(self.processor.meta_data['event'], 'avoid') #event is correct
        self.assertIsInstance(self.processor.meta_data['sensor'], str) # sensor is string
        self.assertEqual(self.processor.meta_data['sensor'], 'D1') # sensor is corect

    def test_get_data(self):
        self.assertIsInstance(self.processor.data, pd.DataFrame)
        self.assertEqual(len(self.processor.data.columns), 7)
    

    
if __name__ == '__main__':
    TEST_LOG_DIR = '/projects/p31961/dopamine_modeling/tests/test_logs'
    TEST_LOG_FILE = 'GabyProcessorTest.log'
    FULL_TEST_LOG_DIR = os.path.join(TEST_LOG_DIR, TEST_LOG_FILE)

    # create logger
    logging.basicConfig(filename=FULL_TEST_LOG_DIR, 
                        level=logging.INFO,
                        format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                        datefmt='%d-%b-%y %H:%M:%S')

    with open(FULL_TEST_LOG_DIR, 'w') as f:
        runner = LoggingTestRunner(verbosity=2)
        unittest.main(testRunner=runner)
