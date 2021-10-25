from setuptools import setup

setup(
  # Silence Warnings
  name='Machine Learning Algorithms',
  url='https://github.com/ShamHolder/ML-Algorithms/tree/main/project',
  author='Shane Holden',
  author_email='holdenmark3@gmail.com',
  packages=['msh_machinelearning'],
  install_requires=['numpy'],
  version='0.1',
  license='MIT',
  description='Python package with classification algorithms for machine learning',
  long_description=open('README.md').read()
)
  
  
