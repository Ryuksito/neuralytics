import pathlib
from setuptools import find_packages, setup

HERE = pathlib.Path(__file__).parent

# Información de la biblioteca
VERSION = '0.1'
PACKAGE_NAME = 'neuralytics'
AUTHOR = ''
AUTHOR_EMAIL = 'alanhd1302@gmail.com'
URL = 'https://github.com/Ryuksito/neuralytics'
LICENSE = 'MIT'

# Dependencias requeridas
REQUIRES = [
    'numpy',
    'tqdm'
]

DESCRIPTION = 'Una biblioteca para machine learning y deep learning' 
LONG_DESCRIPTION = (HERE / "README.md").read_text(encoding='utf-8')
LONG_DESC_TYPE = "text/markdown"

INSTALL_REQUIRES = [
    'numpy',
    'tqdm'
]

setup(
    name=PACKAGE_NAME,
    version=VERSION,
    description=DESCRIPTION,
    url=URL,
    author=AUTHOR,
    author_email=AUTHOR_EMAIL,
    license=LICENSE,
    packages=find_packages(),
    install_requires=REQUIRES,
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
    ],
    long_description=LONG_DESCRIPTION,
    long_description_content_type=LONG_DESC_TYPE,
    include_package_data=True
)

if __name__ == '__main__':
    print('This is the wrong setup.py file to run')
