# -*- coding: utf-8 -*-


from setuptools import setup, find_packages


with open('README.rst') as f:
    readme = f.read()

with open('LICENSE') as f:
    license = f.read()

setup(
    name='covid_forecast',
    version='0.1.0',
    description='Simple implementaiton of LSTM forecasting of covid19 daily cases',
    long_description=readme,
    author='Glen Ryman',
    author_email='glen.ryman@gmail.com',
    url='https://github.com/lizardman1999/covid-daily-lstm',
    license=license,
    packages=find_packages(exclude=('tests', 'docs'))
)

