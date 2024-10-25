#!/usr/bin/env python
from setuptools import setup

setup(name='reasoners',
      version='0.0.0',
      packages=['exllamav2', 'reasoners'],
      entry_points={
          'console_scripts':
          ['reasoners-visualizer=reasoners.visualization:main'],
      },
      install_requires=[
          'tqdm', 'numpy', 'scipy', 'torch', 'datasets', 'huggingface_hub',
          'transformers', 'sentencepiece', 'optimum', 'ninja', 'bitsandbytes',
          'scikit-build', 'pddl', 'matplotlib', 'protobuf', 'accelerate', 'sentencepiece', 'psutil'
      ],
      include_package_data=True,
      python_requires='>=3.10')
