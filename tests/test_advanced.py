# -*- coding: utf-8 -*-

from .context import covid_forecast

import unittest


class AdvancedTestSuite(unittest.TestCase):
    """Advanced test cases."""

    def test_thoughts(self):
        self.assertIsNone(covid_forecast.hmm())


if __name__ == '__main__':
    unittest.main()
