from setuptools import setup, find_packages

setup(name='dynamic-graff',
      version='0.1',
      description='Dynamic GRAFF',
      author='Bertrand Charpentier',
      author_email='charpent@in.tum.de',
      packages=['src'],
      install_requires=[],
      # install_requires=['numpy', 'scipy', 'scikit-learn', 'matplotlib', 'pandas', 'torch', 'tqdm',
      #                   'sacred', 'deprecation', 'pymongo', 'pytorch-lightning>=0.9.0rc2', 'seml'],
      zip_safe=False)
