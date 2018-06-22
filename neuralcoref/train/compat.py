# coding: utf8
"""Py2/3 compatibility"""
import sys

is_python2 = int(sys.version[0]) == 2
is_windows = sys.platform.startswith('win')
is_linux = sys.platform.startswith('linux')
is_osx = sys.platform == 'darwin'

try:
    unicode_ = unicode  # Python 2
    bytes_ = str
    string_types = (str, unicode)
    chr_ = unichr

    def unicode_to_bytes(s, encoding='utf8', errors='strict'):
        return s.encode(encoding=encoding, errors=errors)

    def bytes_to_unicode(b, encoding='utf8', errors='strict'):
        return unicode_(b, encoding=encoding, errors=errors)

except NameError:
    unicode_ = str      # Python 3
    bytes_ = bytes
    string_types = (bytes, str)
    chr_ = chr

    def unicode_to_bytes(s, encoding='utf8', errors='strict'):
        return s.encode(encoding=encoding, errors=errors)

    def bytes_to_unicode(b, encoding='utf8', errors='strict'):
        return b.decode(encoding=encoding, errors=errors)
