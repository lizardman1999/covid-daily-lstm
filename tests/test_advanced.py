# -*- coding: utf-8 -*-

from .context import covid_forecast

import unittest


class AdvancedTestSuite(unittest.TestCase):
    """Advanced test cases."""

    def test_thoughts(self):
        self.assertIsNone(covid_forecast.run_daily_stats)


if __name__ == '__main__':
    unittest.main()
