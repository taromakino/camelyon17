from setuptools import setup, find_packages


setup(
    name='camelyon17',
    packages=find_packages(),
    install_requires=[
        'matplotlib',
        'pandas',
        'pytorch-lightning',
        'torch',
        'torchvision',
        'tqdm'
    ]
)