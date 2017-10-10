from setuptools import setup
import imp


version = imp.load_source('skm.version', 'skm/version.py')

setup(
    name='skm',
    version=version.version,
    description='Spherical k-means clustering',
    author='Justin Salamon',
    author_email='justin.salamon@gmail.com',
    url='https://github.com/justinsalamon/skm',
    download_url='http://github.com/justinsalamon/skm/releases',
    packages=['skm'],
    long_description='Spherical k-means',
    keywords='clustering spherical k-means',
    license='BSD-3-Clause',
    classifiers=[
            "License :: OSI Approved :: BSD License",
            "Programming Language :: Python",
            "Development Status :: 3 - Alpha",
            "Intended Audience :: Developers",
            "Intended Audience :: Science/Research",
            "Programming Language :: Python :: 2",
            "Programming Language :: Python :: 2.7",
            "Programming Language :: Python :: 3",
            "Programming Language :: Python :: 3.4",
            "Programming Language :: Python :: 3.5",
            "Programming Language :: Python :: 3.6",
        ],
    install_requires=[
        'numpy',
        'sklearn',
        'simplejson'
    ],
    extras_require={
        'display': ['matplotlib'],
        'docs': [
                'sphinx==1.2.3',  # autodoc was broken in 1.3.1
                'sphinxcontrib-napoleon',
                'sphinx_rtd_theme',
                'numpydoc',
            ],
        'tests': ['backports.tempfile']
    }
)
