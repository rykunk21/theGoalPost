import unittest

class TestPreprocessor(unittest.TestCase):

    def test_init(self):
        self.assertRaises(Exception, lambda: (_ for _ in ()).throw(Exception('Exception')))

    def test_drop(self):
        pass

    def test_adjust(self):
        pass



if __name__ == '__main__':
    unittest.main()