from setuptools import setup, find_packages

setup(name='neuralcoref',
      version='0.1',
      description="State-of-the-art coreference resolution using neural nets",
      url='https://github.com/huggingface/neuralcoref',
      download_url='https://github.com/huggingface/neuralcoref/archive/0.1.tar.gz',
      author='Thomas Wolf',
      author_email='thomwolf@gmail.com',
      classifiers=[
        'Development Status :: 3 - Alpha',
        'Environment :: Console',
        'Intended Audience :: Developers',
        'Programming Language :: Python',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3.3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5'
      ],
      packages=find_packages(),
      include_package_data=True,
      package_data={'neuralcoref': ['neuralcoref/weights/*.npy']},

      keywords='NLP chatbots coreference resolution',
      license='MIT',
      zip_safe=False,
      platforms='any')
