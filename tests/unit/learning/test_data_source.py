# append base directory `./../../`
import sys
sys.path.append('./../../')

import numpy as np
import unittest
from src.learning.implementations.ArrayDataSource import ArrayDataSource


class TestArrayDataSource(unittest.TestCase):
    
    def setUp(self):
        tuples = []
        last = 0.5
        for i in range(1000):
            tuples.append((last, last+0.5, last+1.0))
            last += 1.5

        self.data = np.array(tuples)
        self.ds = ArrayDataSource(self.data)
        
    def test_len(self):
        self.assertEqual(len(self.ds), 1000)
        
    def test_shape(self):
        self.assertEqual(self.ds.shape(), (1000,3))
        
    def test_get_next_batch(self):
        batch1 = self.ds.get_next_batch(2)
        batch2 = self.ds.get_next_batch(2)
        batch3 = self.ds.get_next_batch(2)

        np.testing.assert_equal(batch1, np.array([(0.5, 1.0, 1.5), (2.0, 2.5, 3.0)]))
        np.testing.assert_equal(batch2, np.array([(3.5, 4.0, 4.5), (5.0, 5.5, 6.0)]))
        np.testing.assert_equal(batch3, np.array([(6.5, 7.0, 7.5), (8.0, 8.5, 9.0)]))
        
    def test_get_all_data(self):
        data = self.ds.get_all_data()
        self.assertTrue(np.array_equal(data, self.data))
        
    def test_reset(self):
        self.ds.get_next_batch(2)
        self.ds.reset()
        batch = self.ds.get_next_batch(2)
        np.testing.assert_equal(batch, np.array([(0.5, 1.0, 1.5), (2.0, 2.5, 3.0)]))
        
    def test_split(self):
        splits = (0.5, 0.25, 0.25)
        s_1, s_2, s_3 = self.ds.split(splits)
        all = self.ds.get_all_data()
        expected_s1 = all[:500]
        expected_s2 = all[500:750]
        expected_s3 = all[750:]
        np.testing.assert_equal(s_1.get_all_data(), expected_s1)
        np.testing.assert_equal(s_2.get_all_data(), expected_s2)
        np.testing.assert_equal(s_3.get_all_data(), expected_s3)

    def test_split_raises_value_error_when_splits_sum_to_more_than_one(self):
        splits = (0.5, 0.5, 0.5)
        with self.assertRaises(ValueError):
            self.ds.split(splits)
    
    def test_split_raises_value_error_when_splits_sum_to_less_than_one(self):
        splits = (0.5, 0.25)
        with self.assertRaises(ValueError):
            self.ds.split(splits)
    
    def test_split_raises_value_error_when_splits_contains_negative_value(self):
        splits = (0.5, 0.5, -0.5)
        with self.assertRaises(ValueError):
            self.ds.split(splits)

    def test_split_raises_value_error_when_splits_contains_inner_zeros(self):
        splits = (0.25, 0, 0.75)
        with self.assertRaises(ValueError):
            self.ds.split(splits)
        
        
if __name__ == '__main__':
    unittest.main()
