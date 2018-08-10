import setuptools

with open('requirements.txt', 'r') as f:
    requirements = f.readlines()

with open('readme.org', 'r') as f:
    readme = f.read()

setuptools.setup(
    name="decaf-espresso",
    version="0.1.1",
    url="https://github.com/jboes/decaf-espresso",

    author="Jacob Boes",
    author_email="jacobboes@gmail.com",

    description="Light-weight ASE calculator wrapper for Quantum Espresso.",
    long_description=readme,
    license='GPL-3.0',

    packages=['espresso'],
    package_dir={'espresso': 'espresso'},
    install_requires=requirements,
    python_requires='>=2.7, !=3.0.*, !=3.1.*, !=3.2.*, !=3.3.*, !=3.4.*, <4',

    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'Topic :: Scientific/Engineering :: Chemistry',
        'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
        'Programming Language :: Python',
        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
    ],
)
