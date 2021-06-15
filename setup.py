from setuptools import find_packages, setup
from glob import glob

classes = """
    Development Status :: 3 - Alpha
    License :: OSI Approved :: BSD License
    Topic :: Software Development :: Libraries
    Topic :: Scientific/Engineering
    Topic :: Scientific/Engineering :: Bio-Informatics
    Programming Language :: Python :: 3
    Programming Language :: Python :: 3 :: Only
    Programming Language :: Python :: 3.4
    Programming Language :: Python :: 3.5
    Programming Language :: Python :: 3.6
    Programming Language :: Python :: 3.7
    Operating System :: Unix
    Operating System :: POSIX
    Operating System :: MacOS :: MacOS X
"""
classifiers = [s.strip() for s in classes.split('\n') if s]

description = ('Toolbox for simulation of microbiome data.')

setup(name='birdman_jr',
      version='0.1.0',
      license='BSD-3-Clause',
      description=description,
      author_email="cameronmartino@gmail.com",
      maintainer_email="cameronmartino@gmail.com",
      packages=find_packages(),
      install_requires=[
          'numpy',
          'biom',
          'pandas',
      ],
      entry_points={},
      package_data={},
      #scripts=glob('birdman_jr/scripts/*'),
      classifiers=classifiers)