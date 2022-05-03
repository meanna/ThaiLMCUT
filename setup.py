from setuptools import setup

setup(
    name='lmcut',
    packages=['lmcut', 'lmcut.weight', 'train'],
    include_package_data=True,
    package_data={
        "lmcut.weight": ["*"],
    },
    version='0.1',
    install_requires=['torch>=1.0', 'numpy>=1.17.3'],
    license='MIT',
    description='A Thai word tokenization library using Deep Neural Network',
    author='Mean',
    url='https://github.com/meanna/Thai_Tokenizer/tree/dev2',
    download_url='https://github.com/meanna/Thai_Tokenizer.git'
)