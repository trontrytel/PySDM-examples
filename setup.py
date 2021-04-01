from setuptools import setup, find_packages

setup(
    name='PySDM-examples',
    use_scm_version=True,
    setup_requires=['setuptools_scm'],
    install_requires=['PySDM @ git+https://github.com/atmos-cloud-sim-uj/PySDM@674f110#egg=PySDM',
                      'PyMPDATA @ git+https://github.com/atmos-cloud-sim-uj/PyMPDATA@f33d602#egg=PyMPDATA',
                      'pystrict>=1.1',
                      'matplotlib>=3.2.2',
                      'ipywidgets>=7.5.1',
                      'ghapi'],  # TODO #457
    author='https://github.com/orgs/atmos-cloud-sim-uj/people',
    license="GPL-3.0",
    packages=find_packages(include=['PySDM_examples', 'PySDM_examples.*'])
)
