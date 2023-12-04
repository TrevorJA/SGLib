from setuptools import setup, find_packages

setup(
    name='sglib',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'matplotlib',
        'statsmodels',
        'pandas'
    ],
    # Other metadata like author, description, etc.
)
