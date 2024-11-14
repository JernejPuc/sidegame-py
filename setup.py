import os
from setuptools import setup


dirname = os.path.dirname(__file__)

with open(os.path.join(dirname, 'README.md'), 'r') as f:
    readme = f.read()

with open(os.path.join(dirname, 'requirements.txt'), 'r') as f:
    requirements = f.read().splitlines()

with open(os.path.join(dirname, 'requirements-ai.txt'), 'r') as f:
    requirements_ai = f.read().splitlines()

with open(os.path.join(dirname, 'requirements-rl.txt'), 'r') as f:
    requirements_rl = f.read().splitlines()

setup(
    name='sidegame',
    version='0.2.1',
    description='SiDeGame - Simplified Defusal Game',
    long_description=readme,
    long_description_content_type='text/markdown',
    url='https://github.com/JernejPuc/sidegame-py',
    author='Jernej Puc',
    author_email='jernej.puc@fs.uni-lj.si',
    license='MPL 2.0',
    classifiers=[
        'License :: OSI Approved :: Mozilla Public License 2.0 (MPL 2.0)',
        'Development Status :: 7 - Inactive',
        'Environment :: Console',
        'Intended Audience :: Education',
        'Intended Audience :: End Users/Desktop',
        'Intended Audience :: Science/Research',
        'Topic :: Games/Entertainment :: First Person Shooters',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Operating System :: Microsoft :: Windows',
        'Operating System :: POSIX :: Linux',
        'Programming Language :: Python :: 3 :: Only',
        'Programming Language :: Python :: 3.11'
    ],
    platforms=['Windows', 'Linux'],
    packages=['sidegame', 'sdgai'],
    package_data={'sidegame': ['assets/*.png', 'assets/*.wav', 'assets/*.npy', 'assets/*.json']},
    data_files=[('user_data', ['*.json']), ('models', ['*.pth'])],
    python_requires='~=3.11',
    install_requires=requirements,
    extras_require={'AI': requirements_ai, 'RL': requirements_rl},
    zip_safe=False
)
