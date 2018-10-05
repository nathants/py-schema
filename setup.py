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
    dependency_links=['https://github.com/nathants/py-util/tarball/e1fd73d9aa90d121e0d5c9c2da4cf04283bc2b94#egg=py-util-0.0.1'],
    packages=['schema'],
    description='data centric schema validation',
)
