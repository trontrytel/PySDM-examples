def pip_install_on_colab(*args):
    import subprocess, notebook, tornado, sys
    subprocess.check_call([sys.executable, '-m', 'pip', '--quiet', '--use-deprecated=legacy-resolver', 'install',
                           'chempy==0.7.10',
                           'pyodesys==0.13.0',
                           f'notebook=={notebook.__version__}',
                           f'tornado=={tornado.version}',
                           *args]
                          )
    subprocess.check_call(['ldconfig'])
