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
    python_requires=r'>=3.8.0',
    install_requires=[
        r'torch>=1.5.0',
        r'numpy>=1.20.0'
    ],
    zip_safe=True
)