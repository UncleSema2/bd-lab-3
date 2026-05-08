import unittest

from src.train import Model


class TestModel(unittest.TestCase):

    def setUp(self) -> None:
        self.model = Model()

    def test_01_log_reg(self):
        self.assertTrue(self.model.log_reg(predict=True))
