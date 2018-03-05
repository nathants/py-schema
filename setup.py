import setuptools


setuptools.setup(
    version="0.0.1",
    license='mit',
    name="py-schema",
    author='nathan todd-stone',
    author_email='me@nathants.com',
    python_requires='>=3.6',
    url='http://github.com/nathants/py-schema',
    install_requires=['py-util'],
    dependency_links=['https://github.com/nathants/py-util/tarball/4d1fe20ecfc0b6982933a8c9b622b1b86da2be5e#egg=py-util-0.0.1'],
    packages=['schema'],
    description='data centric schema validation',
)
