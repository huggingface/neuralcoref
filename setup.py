#!/usr/bin/env python
"""
Simple check list from AllenNLP repo: https://github.com/allenai/allennlp/blob/master/setup.py

To create the package for pypi.

1. Change the version in __init__.py and setup.py.

2. Commit these changes with the message: "Release: VERSION"

3. Add a tag in git to mark the release: "git tag VERSION -m'Adds tag VERSION for pypi' "
   Push the tag to git: git push --tags origin master

4. Build both the sources and the wheel. Do not change anything in setup.py between
   creating the wheel and the source distribution (obviously).

   For the wheel, run: "python setup.py bdist_wheel" in the top level allennlp directory.
   (this will build a wheel for the python version you use to build it - make sure you use python 3.x).

   For the sources, run: "python setup.py sdist"
   You should now have a /dist directory with both .whl and .tar.gz source versions of allennlp.

5. Check that everything looks correct by uploading the package to the pypi test server:

   twine upload dist/* -r pypitest
   (pypi suggest using twine as other methods upload files via plaintext.)

   Check that you can install it in a virtualenv by running:
   pip install -i https://testpypi.python.org/pypi neuralcoref

6. Upload the final version to actual pypi:
   twine upload dist/* -r pypi

7. Copy the release notes from RELEASE.md to the tag in github once everything is looking hunky-dory.

"""
import os
import subprocess
import sys
import contextlib
from distutils.command.build_ext import build_ext
from distutils.sysconfig import get_python_inc
import distutils.util
from distutils import ccompiler, msvccompiler
from setuptools import Extension, setup, find_packages

def is_new_osx():
    """Check whether we're on OSX >= 10.10"""
    name = distutils.util.get_platform()
    if sys.platform != "darwin":
        return False
    elif name.startswith("macosx-10"):
        minor_version = int(name.split("-")[1].split(".")[1])
        if minor_version >= 7:
            return True
        else:
            return False
    else:
        return False


PACKAGE_DATA = {'': ['*.pyx', '*.pxd'],
                '': ['*.h'],}


PACKAGES = find_packages()


MOD_NAMES = ['neuralcoref.neuralcoref']



COMPILE_OPTIONS = {
    "msvc": ["/Ox", "/EHsc"],
    "mingw32": ["-O2", "-Wno-strict-prototypes", "-Wno-unused-function"],
    "other": ["-O2", "-Wno-strict-prototypes", "-Wno-unused-function"],
}


LINK_OPTIONS = {"msvc": [], "mingw32": [], "other": []}


if is_new_osx():
    # On Mac, use libc++ because Apple deprecated use of
    # libstdc
    COMPILE_OPTIONS["other"].append("-stdlib=libc++")
    LINK_OPTIONS["other"].append("-lc++")
    # g++ (used by unix compiler on mac) links to libstdc++ as a default lib.
    # See: https://stackoverflow.com/questions/1653047/avoid-linking-to-libstdc
    LINK_OPTIONS["other"].append("-nodefaultlibs")


USE_OPENMP_DEFAULT = "0" if sys.platform != "darwin" else None
if os.environ.get("USE_OPENMP", USE_OPENMP_DEFAULT) == "1":
    if sys.platform == "darwin":
        COMPILE_OPTIONS["other"].append("-fopenmp")
        LINK_OPTIONS["other"].append("-fopenmp")
        PACKAGE_DATA["spacy.platform.darwin.lib"] = ["*.dylib"]
        PACKAGES.append("spacy.platform.darwin.lib")

    elif sys.platform == "win32":
        COMPILE_OPTIONS["msvc"].append("/openmp")

    else:
        COMPILE_OPTIONS["other"].append("-fopenmp")
        LINK_OPTIONS["other"].append("-fopenmp")


# By subclassing build_extensions we have the actual compiler that will be used which is really known only after finalize_options
# http://stackoverflow.com/questions/724664/python-distutils-how-to-get-a-compiler-that-is-going-to-be-used
class build_ext_options:
    def build_options(self):
        for e in self.extensions:
            e.extra_compile_args += COMPILE_OPTIONS.get(
                self.compiler.compiler_type, COMPILE_OPTIONS["other"]
            )
        for e in self.extensions:
            e.extra_link_args += LINK_OPTIONS.get(
                self.compiler.compiler_type, LINK_OPTIONS["other"]
            )


class build_ext_subclass(build_ext, build_ext_options):
    def build_extensions(self):
        build_ext_options.build_options(self)
        build_ext.build_extensions(self)


# def is_installed(requirement):
#     try:
#         pkg_resources.require(requirement)
#     except pkg_resources.ResolutionError:
#         return False
#     else:
#         return True

# if not is_installed('numpy>=1.11.0') or not is_installed('spacy>=2.1.0'):
#     print(textwrap.dedent("""
#             Error: requirements needs to be installed first.
#             You can install them via:
#             $ pip install -r requirements.txt
#             """), file=sys.stderr)
#     exit(1)

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
            os.path.join(root, 'include')]

        if (ccompiler.new_compiler().compiler_type == 'msvc'
            and msvccompiler.get_build_version() == 9):
            include_dirs.append(os.path.join(root, 'include', 'msvc9'))

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
            version='4.0',
            description="Coreference Resolution in spaCy with Neural Networks",
            url='https://github.com/huggingface/neuralcoref',
            author='Thomas Wolf',
            author_email='thomwolf@gmail.com',
            ext_modules=ext_modules,
            classifiers=[
                'Development Status :: 3 - Alpha',
                'Environment :: Console',
                'Intended Audience :: Developers',
                "Intended Audience :: Science/Research",
                "License :: OSI Approved :: MIT License",
                "Operating System :: POSIX :: Linux",
                "Operating System :: MacOS :: MacOS X",
                "Operating System :: Microsoft :: Windows",
                "Programming Language :: Cython",
                "Programming Language :: Python :: 3.6",
                "Programming Language :: Python :: 3.7",
                "Programming Language :: Python :: 3.8",
                "Topic :: Scientific/Engineering",
            ],
            install_requires=[
                "numpy>=1.15.0",
                "boto3",
                "requests>=2.13.0,<3.0.0",
                "spacy>=2.1.0,<3.0.0"],
            setup_requires=['wheel', 'spacy>=2.1.0,<3.0.0'],
            python_requires=">=3.6",
            packages=PACKAGES,
            package_data=PACKAGE_DATA,
            keywords='NLP chatbots coreference resolution',
            license='MIT',
            zip_safe=False,
            platforms='any',
            cmdclass={"build_ext": build_ext_subclass})

if __name__ == '__main__':
    setup_package()
