# coding: utf8
from __future__ import unicode_literals

import shutil
from pathlib import Path
import plac

from spacy.cli._messages import Messages
from spacy.compat import path2str, json_dumps
from spacy.util import prints
from spacy import util
from spacy import about


@plac.annotations(
    input_dir=("directory with model data", "positional", None, str),
    output_dir=("output parent directory", "positional", None, str),
    meta_path=("path to meta.json", "option", "m", str),
    create_meta=("create meta.json, even if one exists in directory – if "
                 "existing meta is found, entries are shown as defaults in "
                 "the command line prompt", "flag", "c", bool),
    force=("force overwriting of existing model directory in output directory",
           "flag", "f", bool))
def package(input_dir, output_dir, meta_path=None, create_meta=False,
            force=False):
    """
    Generate Python package for model data, including meta and required
    installation files. A new directory will be created in the specified
    output directory, and model data will be copied over.
    """
    input_path = util.ensure_path(input_dir)
    output_path = util.ensure_path(output_dir)
    meta_path = util.ensure_path(meta_path)
    if not input_path or not input_path.exists():
        prints(input_path, title=Messages.M008, exits=1)
    if not output_path or not output_path.exists():
        prints(output_path, title=Messages.M040, exits=1)
    if meta_path and not meta_path.exists():
        prints(meta_path, title=Messages.M020, exits=1)

    meta_path = meta_path or input_path / 'meta.json'
    if meta_path.is_file():
        meta = util.read_json(meta_path)
        if not create_meta:  # only print this if user doesn't want to overwrite
            prints(meta_path, title=Messages.M041)
        else:
            meta = generate_meta(input_dir, meta)
    meta = validate_meta(meta, ['lang', 'name', 'version'])
    model_name = meta['lang'] + '_' + meta['name']
    model_name_v = model_name + '-' + meta['version']
    main_path = output_path / model_name_v
    package_path = main_path / model_name
    bin_path = main_path / 'bin'
    include_path = main_path / 'include'
    orig_nc_path = Path(__file__).parent.parent
    nc_path = package_path / 'neuralcoref'

    create_dirs(package_path, force)
    create_dirs(bin_path, force)
    create_dirs(nc_path, force)

    shutil.copytree(path2str(input_path),
                    path2str(package_path / model_name_v))

    orig_include_path = path2str(Path(__file__).parent / 'include')
    shutil.copytree(path2str(orig_include_path),
                    path2str(include_path))

    nc1_path = path2str(orig_nc_path / 'neuralcoref.pyx')
    nc2_path = path2str(orig_nc_path / 'neuralcoref.pxd')
    shutil.copyfile(path2str(nc1_path),
                    path2str(nc_path / 'neuralcoref.pyx'))
    shutil.copyfile(path2str(nc2_path),
                    path2str(nc_path / 'neuralcoref.pxd'))
    create_file(nc_path / '__init__.py', TEMPLATE_INIT_NC)
    create_file(nc_path / '__init__.pxd', TEMPLATE_INIT_PXD)

    orig_bin_path = path2str(Path(__file__).parent.parent.parent / 'bin' / 'cythonize.py')
    shutil.copyfile(path2str(orig_bin_path),
                    path2str(bin_path / 'cythonize.py'))

    create_file(main_path / 'meta.json', json_dumps(meta))
    create_file(main_path / 'setup.py', TEMPLATE_SETUP)
    create_file(main_path / 'MANIFEST.in', TEMPLATE_MANIFEST)
    create_file(package_path / '__init__.py', TEMPLATE_INIT.format(model_name))
    create_file(package_path / '__init__.pxd', TEMPLATE_INIT_PXD)
    prints(main_path, Messages.M043,
           title=Messages.M042.format(name=model_name_v))


def create_dirs(package_path, force):
    if package_path.exists():
        if force:
            shutil.rmtree(path2str(package_path))
        else:
            prints(package_path, Messages.M045, title=Messages.M044, exits=1)
    Path.mkdir(package_path, parents=True)


def create_file(file_path, contents):
    file_path.touch()
    file_path.open('w', encoding='utf-8').write(contents)


def generate_meta(model_path, existing_meta):
    meta = existing_meta or {}
    settings = [('lang', 'Model language', meta.get('lang', 'en')),
                ('name', 'Model name', meta.get('name', 'model')),
                ('version', 'Model version', meta.get('version', '0.0.0')),
                ('spacy_version', 'Required spaCy version',
                 '>=%s,<3.0.0' % about.__version__),
                ('description', 'Model description',
                  meta.get('description', False)),
                ('author', 'Author', meta.get('author', False)),
                ('email', 'Author email', meta.get('email', False)),
                ('url', 'Author website', meta.get('url', False)),
                ('license', 'License', meta.get('license', 'CC BY-SA 3.0'))]
    nlp = util.load_model_from_path(Path(model_path))
    meta['pipeline'] = nlp.pipe_names
    meta['vectors'] = {'width': nlp.vocab.vectors_length,
                       'vectors': len(nlp.vocab.vectors),
                       'keys': nlp.vocab.vectors.n_keys}
    prints(Messages.M047, title=Messages.M046)
    for setting, desc, default in settings:
        response = util.get_raw_input(desc, default)
        meta[setting] = default if response == '' and default else response
    if about.__title__ != 'spacy':
        meta['parent_package'] = about.__title__
    return meta


def validate_meta(meta, keys):
    for key in keys:
        if key not in meta or meta[key] == '':
            prints(Messages.M049, title=Messages.M048.format(key=key), exits=1)
    return meta


TEMPLATE_SETUP = """
#!/usr/bin/env python
from __future__ import print_function
import io
import os
from os import path, walk
import json
import sys
import contextlib
import subprocess
from shutil import copy
from distutils.sysconfig import get_python_inc
from distutils import ccompiler, msvccompiler
from setuptools import Extension, setup, find_packages

PACKAGES = find_packages()

PACKAGE_DATA = {'': ['*.pyx', '*.pxd']}

def load_meta(fp):
    with io.open(fp, encoding='utf8') as f:
        return json.load(f)


def list_files(data_dir):
    output = []
    for root, _, filenames in walk(data_dir):
        for filename in filenames:
            if not filename.startswith('.'):
                output.append(path.join(root, filename))
    output = [path.relpath(p, path.dirname(data_dir)) for p in output]
    output.append('meta.json')
    return output


def list_requirements(meta):
    parent_package = meta.get('parent_package', 'spacy')
    requirements = [parent_package + ">=" + meta['spacy_version']]
    if 'setup_requires' in meta:
        requirements += meta['setup_requires']
    return requirements

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
    print('Cythonizing sources in', source)
    p = subprocess.call([sys.executable,
                         os.path.join(root, 'bin', 'cythonize.py'),
                         source], env=os.environ)
    if p != 0:
        raise RuntimeError('Running cythonize failed')


def is_source_release(model_path):
    return os.path.exists(os.path.join(model_path, 'neuralcoref/neuralcoref.cpp'))


def setup_package():
    root = path.abspath(path.dirname(__file__))

    with chdir(root):
        meta_path = path.join(root, 'meta.json')
        meta = load_meta(meta_path)
        model_name = str(meta['lang'] + '_' + meta['name'])
        model_dir = path.join(model_name, model_name + '-' + meta['version'])

        include_dirs = [
            get_python_inc(plat_specific=True),
            os.path.join(root, 'include')]

        if (ccompiler.new_compiler().compiler_type == 'msvc'
            and msvccompiler.get_build_version() == 9):
            include_dirs.append(os.path.join(root, 'include', 'msvc9'))

        ext_modules = []
        mod_name = model_name + '.neuralcoref.neuralcoref'
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

        if not is_source_release(model_name):
            generate_cython(root, model_name)

        copy(meta_path, path.join(model_name))
        copy(meta_path, model_dir)
        package_data = PACKAGE_DATA
        package_data[model_name] = list_files(model_dir)
        setup(
            name=model_name,
            description=meta['description'],
            author=meta['author'],
            author_email=meta['email'],
            url=meta['url'],
            version=meta['version'],
            license=meta['license'],
            ext_modules=ext_modules,
            packages=PACKAGES,
            package_data=package_data,
            install_requires=list_requirements(meta),
            zip_safe=False,
        )


if __name__ == '__main__':
    setup_package()
""".strip()


TEMPLATE_MANIFEST = """
include meta.json
recursive-include include *.h
recursive-include bin *.py
""".strip()


TEMPLATE_INIT = """
# coding: utf8
from __future__ import unicode_literals

from pathlib import Path
from spacy.util import load_model_from_init_py, get_model_meta
from {}.neuralcoref import NeuralCoref

__version__ = get_model_meta(Path(__file__).parent)['version']


def load(**overrides):
    disable = overrides.get('disable', [])
    overrides['disable'] = disable + ['neuralcoref']
    nlp = load_model_from_init_py(__file__, **overrides)
    coref = NeuralCoref(nlp.vocab)
    coref.from_disk(nlp.path / 'neuralcoref')
    nlp.add_pipe(coref, name='neuralcoref')
    return nlp
""".strip()

TEMPLATE_INIT_NC = """
from .neuralcoref import NeuralCoref
""".strip()

TEMPLATE_INIT_PXD = """
""".strip()

