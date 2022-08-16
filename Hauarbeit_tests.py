#%%test if exception is raised when no csv is given in Hausarbeit.py
import unittest
import Hausarbeit as H
import pandas as pd
import numpy as np

#%%
class MyTestCase(unittest.TestCase):


# Returns true if 1 + '1' raises a TypeError
    def test_1(self):
        with self.assertRaises(Exception):
            1 + '1'
if __name__ == '__main__':
        unittest.main()






#%% test if find_nearest is working
class TestFindNearest(unittest.TestCase):
    def test_find_nearest(self):
        array = np.array([1,2,3,4,5,6,7,8,9,10])
        self.assertEqual(H.find_nearest(array, 5), 5)

if __name__ == '__main__':
    unittest.main()











        

# %%
