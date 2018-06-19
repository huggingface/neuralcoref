# coding: utf8
from __future__ import print_function
# NB! This breaks in plac on Python 2!!
# from __future__ import unicode_literals

if __name__ == '__main__':
    import plac
    import sys
    from neuralcoref.cli import package
    from spacy.util import prints

    commands = {
        'package': package,
    }
    if len(sys.argv) == 1:
        prints(', '.join(commands), title="Available commands", exits=1)
    command = sys.argv.pop(1)
    sys.argv[0] = 'neuralcoref %s' % command
    if command in commands:
        plac.call(commands[command], sys.argv[1:])
    else:
        prints(
            "Available: %s" % ', '.join(commands),
            title="Unknown command: %s" % command,
            exits=1)
