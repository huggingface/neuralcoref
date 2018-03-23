#!/usr/bin/env python

#---- standard library imports ----#
import sys

# version check
if(not (sys.hexversion >= 0x2050000)):
    sys.exit("\n\nplease use python version >= 2.5\n\n")

import string
import re
import math
import os
import time
import getopt
import zlib
import gzip
import codecs
import optparse
import collections
import ConfigParser
from optparse import OptionParser











def error(error_string, terminate_program=True, current_frame=False):
    """Print error messages to stderr, optionally sys.exit"""

    if(current_frame == False):
        pretty_error_string = """

--------------------------------------------------------------------------------
                                       ERROR
--------------------------------------------------------------------------------
%s
--------------------------------------------------------------------------------

""" % (error_string)
    else:
        pretty_error_string = """

--------------------------------------------------------------------------------
                                       ERROR
--------------------------------------------------------------------------------
FILE: %s
LINE: %s
--------------------------------------------------------------------------------
%s
--------------------------------------------------------------------------------

""" % (current_frame.f_code.co_filename, current_frame.f_lineno, error_string)

    sys.stderr.write(pretty_error_string)

    if(terminate_program == True):
        sys.exit(1)










def parse_cfg_args(arg_list):
    """Parse command-line style config settings to a dictionary.

    If you want to override configuration file values on the command
    line or set ones that were not set, this should make it simpler.
    Given a list in format [section.key=value, ...] return a
    dictionary in form { (section, key): value, ...}.

    So we might have:

    .. code-block:: python

      ['corpus.load=english-mz',
       'corpus.data_in=/home/user/corpora/ontonotes/data/']

    we would then return the dictionary:

    .. code-block:: python

      { ('corpus', 'load') : 'english-mz',
        ('corpus', 'data_in') : '/home/user/corpora/ontonotes/data/' }

    See also :func:`load_config` and :func:`load_options`

    """

    if not arg_list:
        return {}

    config_append = {}

    for arg in arg_list:
        if len(arg.split("=")) != 2 or len(arg.split("=")[0].split('.')) != 2:
            raise Exception("Invalid argument; not in form section.key=value : " + arg)

        key, value = arg.split("=")
        config_append[tuple(key.split("."))] = value

    return config_append



















__registered_config_options = collections.defaultdict( dict )

def required_config_sections():
    return [section for section in __registered_config_options if
            [True for value in __registered_config_options[section]
             if __registered_config_options[section][value][3]]] # section_required












def load_options(parser=None, argv=[], positional_args=True):
    """ parses sys.argv, possibly exiting if there are mistakes

    If you set parser to a ConfigParser object, then you have control
    over the usage string and you can prepopulate it with options you
    intend to use.  But don't set a ``--config`` / ``-c`` option;
    load_options uses that to find a configuration file to load

    If a parser was passed in, we return ``(config, parser, [args])``.
    Otherwise we return ``(config, [args])``.  Args is only included
    if ``positional_args`` is True and there are positional arguments

    See :func:`load_config` for details on the ``--config`` option.

    """

    def is_config_appender(arg):
        return "." in arg and "=" in arg and arg.find(".") < arg.find("=")

    parser_passed_in=parser
    if not parser:
        parser = OptionParser()

    parser.add_option("-c", "--config", help="the path to a config file to read options from")

    if argv:
        options, args = parser.parse_args(argv)
    else:
        options, args = parser.parse_args()

    config = load_config(options.config, [a for a in args if is_config_appender(a)])

    other_args = [a for a in args if not is_config_appender(a)]

    return_list = [config]
    if parser_passed_in:
        return_list.append(options)
    if other_args:
        if positional_args:
            return_list.append(other_args)
        else:
            raise Exception("Arguments %s not understood" % other_args)
    else:
        if positional_args:
            raise Exception("This program expects one or more positional arguments that are missing")

    if len(return_list) == 1:
        return return_list[0]
    else:
        return tuple(return_list)















class FancyConfigParserError(Exception):
    """ raised by :class:`FancyConfigParser` when used improperly """

    def __init__(self, vals):
        Exception.__init__(self, 'Config usage must be in the form "config[\'section\', \'item\']". '
                           'Given something more like "config[%s]".' % (", ".join("%r"%v for v in vals)))











class FancyConfigParser(ConfigParser.SafeConfigParser):
    """ make a config parser with support for config[section, value]

    raises :class:`FancyConfigParserError` on improper usage.

    """

    def __getitem__(self, vals):
        try:
            section, item = vals
        except (ValueError, TypeError):
            raise FancyConfigParserError(vals)
        return self.get(section, item)


    def __setitem__(self, vals, value):
        try:
            section, item = vals
        except (ValueError, TypeError):
            raise FancyConfigParserError(vals)
        return self.set(section, item, value)

    def __delitem__(self, vals):
        try:
            section, item = vals
        except (ValueError, TypeError):
            raise FancyConfigParserError(vals)

        self.remove_option(section, item)










def load_config(cfg_name=None, config_append=[]):
    """ Load a configuration file to memory.

    The given configuration file name can be a full path, in which
    case we simply read that configuration file.  Otherwise, if you
    give 'myconfig' or something similar, we look in the current
    directory and the home directory.  We also look to see if files
    with this name and extension '.conf' exist.  So for 'myconfig' we
    would look in the following places:

     * ./myconfig
     * ./myconfig.conf
     * [home]/.myconfig
     * [home]/.myconfig.conf

    Once we find the configuration, we load it.  We also extend
    ConfigParser to support ``[]`` notation.  So you could look up key
    ``k`` in section ``s`` with ``config[s,k]``.  See
    :func:`FancyConfigParser` .

    If config_append is set we use :func:`parse_cfg_args` and add any
    values it creates to the config object.  These values override any
    previous ones.

    """

    config = FancyConfigParser()

    if cfg_name:
        config_locs = [cfg_name + '.conf',
                       os.path.expanduser('~/.' + cfg_name + '.conf'),
                       cfg_name,
                       os.path.expanduser('~/.' + cfg_name)]
        l = config.read(config_locs)
        if not l:
            raise Exception("Couldn't find config file.  Looked in:" +
                            "".join(["\n - " + c for c in config_locs]) +
                            "\nto no avail.")


    for (section, key_name), value in parse_cfg_args(config_append).iteritems():
        if not config.has_section(section):
            config.add_section(section)
        config.set(section, key_name, value)

    problems = []
    for section in config.sections():
        if not is_config_section_registered(section):
            on.common.log.status("Ignoring unknown configuration section", section)
            continue
        for option in config.options(section):
            if not is_config_registered(section, option):
                problems.append("Unknown configuration variable %s.%s" % (section, option))
                continue

            value = config.get(section, option)
            allowed = allowed_config_values(section, option)
            multiple = allow_multiple_config_values(section, option)

            values = value.split() if multiple else [value]
            for value in values:
                if allowed and not value in allowed:
                    problems.append("Illegal value '%s' for configuration variable %s.%s.  Permitted values are: %s" %
                                    (value, section, option, ", ".join(["'%s'" % x for x in allowed])))

        for option in required_config_options(section):
            if not config.has_option(section, option):
                problems.append("Required configuration variable %s.%s is absent" % (section, option))

    for section in required_config_sections():
        if not config.has_section(section):
            problems.append("Required configuration section %s is absent" % section)

    if problems:
        print_config_docs()

        on.common.log.status("Configuration Problems:")
        for problem in problems:
            on.common.log.status("  " + problem)

        sys.exit(-1)

    return config
























def pretty_print_parse_string(a_parse_string, offset=''):

    if not a_parse_string.strip():
        return ""

    # Maximum depth we're prepared for in parses
    maxdepth=100
    maxindent=300

    # Table of indentation at parse depth
    depth_to_indent = [0 for i in xrange(maxdepth)]

    # Initialize indent_string[i] to be a string of i spaces
    indent_string = ['' for i in xrange(maxindent)]
    for i in xrange(maxindent-1):
        indent_string[i+1] = indent_string[i] + ' '

    # RE object for split that matches on a ')' followed by not a ')', but only consumes the ')'
    close_paren = re.compile(r'\)(?=\s*[^\s\)])')

    # RE object to pick up first on this line(should be only) POS tag and the word of each lexical leaf of the parse
    lexical_leaf = re.compile(r'\((?P<tag>[^\s\)\(]+)\s+(?P<word>[^\s\)\(]+)\)')

    # RE object to parse OntoNotes Normal Form parse lines:
    a_parse = a_parse_string

    pp_parse = ""

    def parseindent(depth):
        return indent_string[depth_to_indent[depth]]+offset  #Indent to appropriate point


    current_depth = 0
    for frag in  close_paren.split(a_parse):  #Split input into lines ending with a lexical item
        if frag[-1]!= '\n':
            frag=frag+')'
        else: frag=frag[0:-1]

        #Indent to appropriate point
        pp_parse += parseindent(current_depth)

        pfrag = ""
        for pfrag in (frag).split('(')[1:]:         # Split line into segments each beginning with an '('
            pfrag='('+pfrag                         # Restore deleted initial '('
            pp_parse += pfrag                      # Print each
            current_depth=current_depth+1           # Up the current depth count

            # Remember how far to indent following items at this depth
            depth_to_indent[current_depth]=depth_to_indent[current_depth-1]+len(pfrag)

        current_depth=current_depth-pfrag.count(')')    # Correct depth given closing parens
        if current_depth<=0:
            pp_parse += ''            # Separate toplevel parses with blank lines

        pp_parse += '\n'              # Print CRLF


    return re.sub("\)$", "", pp_parse)









class parsed_sentence:

    def __init__(self, r_c_matrix):
        self.words = []
        self.parse_bits = []

        for r_i, r in enumerate(r_c_matrix):

            word = r[0]
            self.words.append(word)

            # process encoded chain
            part_of_speech=r[1]
            encoded_parse=r[2]

            self.parse_bits.append(encoded_parse.replace("*", "(%s %s)" % (part_of_speech, word)))



    def __repr__(self):
        return pretty_print_parse_string("%s" % ("".join(self.parse_bits).replace("(", " (").strip()))











def main():
    # total number of expected actual arguments, not counting the command itself
    required_number_of_args = 1

    o_parser = optparse.OptionParser()
    o_parser.set_defaults(DEBUG=False)
    o_parser.set_defaults(LANGUAGE=None)

    o_parser.add_option("-d", "--debug", action="store_true", dest="DEBUG", help="Set debug mode on")
    o_parser.add_option("-l", "--language", action="store", dest="LANGUAGE", help="Set language")
    o_parser.add_option("-o", "--option", help="perform the specified task.  this can be 'pre-process' or 'decode'")

    if(required_number_of_args > 0):
        c_config, o_options, o_args = load_options(parser=o_parser)
        if(len(o_args) != required_number_of_args):
            error("please specify %s arguments" % (required_number_of_args))
    else:
        c_config, o_options = load_options(parser=o_parser, positional_args=False)



    legal_options = []
    if(legal_options != []
       and
       o_options.option not in legal_options):
        error("please specify one of %s options" % (" ".join(legal_options)))


    if o_options.LANGUAGE is None:
        error("please specify language using -l option")
        

    r_c_matrix = []
    if(required_number_of_args > 0):
        file = open(o_args[0])
        file_line = file.readline()
        while( file_line != "" ):
            #---- start processing here ----#
            file_line = file_line.strip()

            if(file_line.startswith("#begin") or file_line.startswith("#end")):
                pass

            elif(file_line == ""):
                a_parsed_sentence = parsed_sentence(r_c_matrix)
                print a_parsed_sentence
                r_c_matrix = []

            else:
                columns = file_line.split()
                (word, part_of_speech, encoded_parse) = columns[3], columns[4], columns[5]
                
                if o_options.LANGUAGE == "arabic":
                    r_c_matrix.append([re.sub("#.*", "", word), part_of_speech, encoded_parse])
                else:
                    r_c_matrix.append([word, part_of_speech, encoded_parse])

            file_line = file.readline()

        #---- close the file ----#
        if(file != sys.stdin):
            file.close()

if __name__ == '__main__':
    main()

