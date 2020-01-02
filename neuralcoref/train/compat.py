import sys

is_windows = sys.platform.startswith("win")
is_linux = sys.platform.startswith("linux")
is_osx = sys.platform == "darwin"


# Python 3 is default, Python 2 is not supported anymore
unicode_ = str
bytes_ = bytes
string_types = (bytes, str)
chr_ = chr


def unicode_to_bytes(s, encoding="utf8", errors="strict"):
    return s.encode(encoding=encoding, errors=errors)


def bytes_to_unicode(b, encoding="utf8", errors="strict"):
    return b.decode(encoding=encoding, errors=errors)
