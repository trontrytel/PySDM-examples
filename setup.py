from setuptools import setup, find_packages

setup(
    name='PySDM-examples',
    use_scm_version=True,
    setup_requires=['setuptools_scm'],
    install_requires=['PySDM',
                      'PyMPDATA',
                      'pystrict',
                      'matplotlib',
                      'ipywidgets',
                      'ghapi'],  # note: test-time-requirement?
    author='https://github.com/orgs/atmos-cloud-sim-uj/people',
    license="GPL-3.0",
    packages=find_packages(include=['PySDM_examples', 'PySDM_examples.*'])
)
