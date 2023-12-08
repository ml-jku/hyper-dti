import setuptools

with open(r'README.md', mode=r'r') as readme_handle:
    long_description = readme_handle.read()

setuptools.setup(
    name=r'hyper-dti',
    version=r'1.0.0',
    author=r'Emma Svensson',
    author_email=r'svensson@ml.jku.at',
    url=r'https://github.com/ml-jku/hyper-dti',
    description=r'Robust task-conditioned modeling of drug-target interactions.',
    long_description=long_description,
    long_description_content_type=r'text/markdown',
    packages=setuptools.find_packages(),
    dependency_links = ['https://github.com/ml-jku/hopfield-layers'],
    python_requires=r'>=3.8.0',
    install_requires=[
        r'torch>=1.9.0',
        r'scikit-learn>=0.24.0',
        r'pytdc>=0.3.8',
        r'hopfield-layers>=1.0.2',
        r'rdkit>=2023.03.1',
        r'wandb>=0.15.4'
    ],
    zip_safe=True
)
