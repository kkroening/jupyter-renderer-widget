from setuptools import setup


version = '0.1.0'


setup(
    name='jupyter-renderer-widget',
    packages=['jupyter_renderer_widget'],
    version=version,
    description='Renderer widget for JupyterLab',
    author='Karl Kroening',
    author_email='karlk@kralnet.us',
    url='https://github.com/kkroening/jupyterlab-renderer-widget',
    long_description='Renderer widget for JupyterLab',
    install_requires=[
        'ffmpeg-python',
        'ipywidgets',
        'matplotlib',
        'numpy',
        'Pillow',
        'tqdm',
    ],
    extras_require={
        'dev': [
            'pytest',
            'pytest-runner',
            'sphinx',
        ],
    },
    classifiers=[
        'Intended Audience :: Developers',
        'License :: OSI Approved :: Apache Software License',
        'Natural Language :: English',
        'Operating System :: OS Independent',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
    ],
)
