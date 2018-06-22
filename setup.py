#!/usr/bin/env python
from __future__ import print_function
import os
import subprocess
import sys
import numpy
import contextlib
from distutils.command.build_ext import build_ext
from distutils.sysconfig import get_python_inc
from distutils import ccompiler, msvccompiler
from setuptools import Extension, setup, find_packages

PACKAGE_DATA = {'': ['*.pyx', '*.pxd']}


PACKAGES = find_packages()


MOD_NAMES = ['neuralcoref.neuralcoref']


@contextlib.contextmanager
def chdir(new_dir):
    old_dir = os.getcwd()
    try:
        os.chdir(new_dir)
        sys.path.insert(0, new_dir)
        yield
    finally:
        del sys.path[0]
        os.chdir(old_dir)


def generate_cython(root, source):
    print('Cythonizing sources')
    p = subprocess.call([sys.executable,
                         os.path.join(root, 'bin', 'cythonize.py'),
                         source], env=os.environ)
    if p != 0:
        raise RuntimeError('Running cythonize failed')


def is_source_release(path):
    return os.path.exists(os.path.join(path, 'PKG-INFO'))


def setup_package():
    root = os.path.abspath(os.path.dirname(__file__))
    with chdir(root):
        if not is_source_release(root):
            generate_cython(root, 'neuralcoref')

        include_dirs = [
            get_python_inc(plat_specific=True),
            os.path.join(root, 'neuralcoref', 'cli', 'include')]

        if (ccompiler.new_compiler().compiler_type == 'msvc'
            and msvccompiler.get_build_version() == 9):
            include_dirs.append(os.path.join(root, 'neuralcoref', 'cli', 'include', 'msvc9'))

        ext_modules = []
        for mod_name in MOD_NAMES:
            mod_path = mod_name.replace('.', '/') + '.cpp'
            extra_link_args = []
            # ???
            # Imported from patch from @mikepb
            # See Issue #267. Running blind here...
            if sys.platform == 'darwin':
                dylib_path = ['..' for _ in range(mod_name.count('.'))]
                dylib_path = '/'.join(dylib_path)
                dylib_path = '@loader_path/%s/neuralcoref/platform/darwin/lib' % dylib_path
                extra_link_args.append('-Wl,-rpath,%s' % dylib_path)
            ext_modules.append(
                Extension(mod_name, [mod_path],
                    language='c++', include_dirs=include_dirs,
                    extra_link_args=extra_link_args))

        setup(name='neuralcoref',
            version='3.0',
            description="State-of-the-art coreference resolution using neural nets",
            url='https://github.com/huggingface/neuralcoref',
            download_url='https://github.com/huggingface/neuralcoref/archive/3.0.tar.gz',
            author='Thomas Wolf',
            author_email='thomwolf@gmail.com',
            ext_modules=ext_modules,
            include_dirs=[numpy.get_include()],
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
            install_requires=[
            'spacy',
            'falcon'],
            packages=PACKAGES,
            package_data=PACKAGE_DATA,
            keywords='NLP chatbots coreference resolution',
            license='MIT',
            zip_safe=False,
            platforms='any')


if __name__ == '__main__':
    setup_package()
