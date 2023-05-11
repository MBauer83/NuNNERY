# append base directory `./../../`
import sys
sys.path.append('./../../')

import numpy as np
import unittest
from src.learning.implementations.MultiClassClassification import MultiClassClassification


class TestMultiClassClassification(unittest.TestCase):
    
    def setUp(self):
        self.mc = MultiClassClassification(('label1', 'label2', 'label3', 'label4'))
        
    def test_get_class_names(self):
        expected_output = ['label1', 'label2', 'label3', 'label4']
        np.testing.assert_equal(self.mc.get_class_names(), expected_output)
        
    def test_get_class_for_one_hot_encoding(self):
        input_encoding = np.array([0, 1, 0, 0])
        expected_output = 'label2'
        self.assertEqual(self.mc.get_class_for_one_hot_encoding(input_encoding), expected_output)
        
    def test_get_classes_with_values_gte(self):
        output_activations = np.array([0.1, 0.5, 0.6, 0.8])
        threshold = 0.6
        expected_output = [('label3', 0.6), ('label4', 0.8)]
        np.testing.assert_equal(self.mc.get_classes_with_values_gte(output_activations, threshold), expected_output)
        
    def test_get_classes_with_values_lte(self):
        output_activations = np.array([0.1, 0.5, 0.6, 0.8])
        threshold = 0.5
        expected_output = [('label1', 0.1), ('label2', 0.5)]
        np.testing.assert_equal(self.mc.get_classes_with_values_lte(output_activations, threshold), expected_output)
        
    def test_get_classes_with_values_gt(self):
        output_activations = np.array([0.1, 0.5, 0.6, 0.8])
        threshold = 0.5
        expected_output = [('label3', 0.6), ('label4', 0.8)]
        np.testing.assert_equal(self.mc.get_classes_with_values_gt(output_activations, threshold), expected_output)
        
    def test_get_classes_with_values_lt(self):
        output_activations = np.array([0.1, 0.5, 0.6, 0.8])
        threshold = 0.6
        expected_output = [('label1', 0.1), ('label2', 0.5)]
        np.testing.assert_equal(self.mc.get_classes_with_values_lt(output_activations, threshold), expected_output)
        
    def test_get_one_hot_encoding(self):
        label = 'label1'
        expected_output = np.array([1, 0, 0, 0])
        np.testing.assert_equal(self.mc.get_one_hot_encoding(label), expected_output)
        
if __name__ == '__main__':
    unittest.main()
