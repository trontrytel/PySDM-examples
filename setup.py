from setuptools import setup, find_packages

setup(
    name='PySDM-examples',
    use_scm_version=True,
    setup_requires=['setuptools_scm'],
    install_requires=['git+git://github.com/atmos-cloud-sim-uj/PySDM@b4780cb#egg=PySDM',
                      'git+git://github.com/atmos-cloud-sim-uj/PyMPDATA@f33d602#egg=PyMPDATA',
                      'pystrict>=1.1',
                      'matplotlib>=3.2.2',
                      'ipywidgets>=7.5.1',
                      'ghapi'],  # TODO
    author='https://github.com/orgs/atmos-cloud-sim-uj/people',
    license="GPL-3.0",
    packages=find_packages(include=['PySDM_examples', 'PySDM_examples.*'])
)
