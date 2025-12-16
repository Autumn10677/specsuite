from setuptools import setup

setup(
    name='specsuite',
    version='0.1.0',    
    description='A toolbox for processing data from slitless spectrographs',
    url='https://github.com/Autumn10677/specsuite',
    author='Autumn Stephens',
    author_email='aust8150@colorado.edu',
    packages=['specsuite'],
    install_requires=[
        'astropy>=7.0',
        'ipywidgets>=8.0',
        'ipympl',
        'joblib>=1.0',
        'lxml',
        'matplotlib>=3.0',
        'numpy>=2.0',
        'pandas>=2.0',
        'requests>=2.0',
        'scipy>=1.0',
        'tqdm>=4.0',
    ],
)