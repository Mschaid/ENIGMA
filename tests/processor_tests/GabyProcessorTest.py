import unittest
from src.processors.GabyProcessor import GabyProcessor

TEST_PATH =r'/Users/michaelschaid/GitHub/dopamine_modeling/data/test_data_files/gaby_testing/Day7/142-237_Day7_Avoid_D1_z_score_D1.h5'

class TestGabyProcessor(unittest.TestCase):
    
    
    def setUp(self) -> None:
        self.processor = GabyProcessor(path=TEST_PATH)
        
    def test_get_meta_data(self):
        self.processor.get_meta_data()
        self.assertIsInstance(self.processor.meta_data, dict)
        self.assertIsInstance(self.processor.meta_data['mouse_id'], str)  
        self.assertEqual(self.processor.meta_data['mouse_id'], '142_237')
        self.assertIsInstance(self.processor.meta_data['day'], int)
        self.assertEqual(self.processor.meta_data['day'], 7)
        self.assertIsInstance(self.processor.meta_data['event'], str)
        self.assertEqual(self.processor.meta_data['event'], 'avoid')
        self.assertIsInstance(self.processor.meta_data['sensor'], str)
        self.assertEqual(self.processor.meta_data['sensor'], 'D1')

   

    def test_get_data(self):
        pass
    

    
if __name__ == '__main__':
    unittest.main(verbosity=2)
