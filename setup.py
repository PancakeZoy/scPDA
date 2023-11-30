from setuptools import setup

setup(
	name='scPDA',
	version='0.1',
	description='Single Cell Protein Counts Denoising',
	author='Ouyang Zhu, Jun Li',
	author_email='ozhu@nd.edu',
	url='https://github.com/PancakeZoy/scPDA',
	install_requires=[
		'torch >= 2.0.0',
		'tqdm',
		'anndata',
		'pandas',
		'numpy',
	]
)