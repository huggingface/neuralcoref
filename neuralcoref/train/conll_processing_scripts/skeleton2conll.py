#!/usr/bin/env python

"""
Get most current usage with:

  python skeleton2conll.py --help

"""

from __future__ import with_statement
import codecs
import sys
import os
import re
import string
from collections import defaultdict


WORD_COLUMN=3
LEMMA_COLUMN=6




MIN_VERBOSITY = 0
MED_VERBOSITY = 5
MAX_VERBOSITY = 10
SUPER_VERBOSITY = 15

DEBUG = False
VERBOSITY = MAX_VERBOSITY








def debug(debug_object, debug_flag=DEBUG, verbosity=MAX_VERBOSITY, nl=False):
  if((debug_flag == True) and (verbosity <= VERBOSITY)):
    if nl:
      trailing_char = "\n"
    else:
      trailing_char = ""

    sys.stderr.write(str(debug_object) + trailing_char)












def warning(warning_string, verbosity=0):
  """ print warning string depending on the value of VERBOSITY """

  if(verbosity <= VERBOSITY):
    sys.stderr.write(u"""

--------------------------------------------------------------------------------
                                      WARNING
--------------------------------------------------------------------------------
%s
--------------------------------------------------------------------------------

""" % (warning_string))








class abstract_open_type_table:

  def __init__(self, a_id, data_pointer=None):
    self.id = a_id
    self.type_hash[self.id] += 1

  @classmethod
  def write_to_db(cls, cursor):
    for a_type in cls.type_hash.keys():
      insert_ignoring_dups(cls, cursor, a_type)

  @classmethod
  def __repr__(cls):
    return " ".join(cls.type_hash.keys())

  @classmethod
  def get_table(cls):
    try:
        return cls.sql_insert_statement.strip().split("\n")[0].split()[2]
    except Exception:
        return "unknown"








class lemma_type(abstract_open_type_table):
  type_hash = defaultdict(int)

  sql_table_name = "lemma_type"
  sql_create_statement = \
"""
create table lemma_type
(
  id varchar(255) not null collate utf8_bin primary key
)
default character set utf8;
"""


  sql_insert_statement = \
"""insert into lemma_type
(
  id
)
values (%s)
"""







class lemma:
  """ arabic trees have extra lemma information """

  def __init__(self, input_string, b_transliteration, comment, index, offset, unvocalized_string,
               vocalized_string, vocalized_input, pos, gloss, lemma, coarse_sense, leaf_id):

    self.input_string = input_string
    self.b_transliteration = b_transliteration
    self.comment = comment
    self.index = index
    self.offset = offset
    self.unvocalized_string = unvocalized_string
    self.vocalized_string = vocalized_string
    self.vocalized_input = vocalized_input
    self.pos = pos
    self.gloss = gloss
    self.lemma = lemma
    self.coarse_sense = coarse_sense
    self.leaf_id = leaf_id
    
    self.id = "%s@%s" % (self.lemma, self.leaf_id)

  sql_table_name = "lemma"

  def __repr__(self):
    return "\n".join(["lemma instance:",
                      "  input_string: " + self.input_string,
                      "  vocalized_input: " + self.vocalized_input,
                      "  unvocalized_string: " + self.unvocalized_string,
                      "  vocalized_string: " + self.vocalized_string,
                      "  gloss: " + self.gloss,
                      "  index: %s" % self.index,
                      "  offset: %s" % self.offset])

  def __str__(self):
    tr = ["INPUT STRING:%s" % self.input_string,
          "    IS_TRANS:%s" % self.b_transliteration,
          "     COMMENT:%s" % self.comment,
          "       INDEX:%s" % self.index,
          "     OFFSETS:%s" % self.offset,
          " UNVOCALIZED:%s" % self.unvocalized_string,
          "   VOCALIZED:%s" % self.vocalized_string,
          "  VOC_STRING:%s" % self.vocalized_input,
          "         POS:%s" % self.pos,
          "       GLOSS:%s" % self.gloss]

    if self.lemma != "lemma_not_set":
      if self.coarse_sense:
        lemma_str = "%s_%s" % (self.lemma, self.coarse_sense)
      else:
        lemma_str = self.lemma

      tr.append("       LEMMA: [%s]" % lemma_str)

    return "\n".join(tr)


  @staticmethod
  def from_db(a_leaf_id, a_cursor):
    a_cursor.execute("SELECT * FROM lemma WHERE leaf_id = '%s'" % a_leaf_id)
    rows = a_cursor.fetchall()

    if not rows:
      return None

    if len(rows) != 1:
      assert all(row["lemma"] == rows[0]["lemma"] for row in rows), \
          "\n".join(", ".join(": ".join(a) for a in row.iteritems()) for row in rows)

    r = rows[0]

    return lemma(r["input_string"],
                 r["b_transliteration"],
                 r["comment"],
                 r["lemma_index"],
                 r["lemma_offset"],
                 r["unvocalized_string"],
                 r["vocalized_string"],
                 r["vocalized_input"],
                 r["pos"],
                 r["gloss"],
                 r["lemma"],
                 r["coarse_sense"],
                 r["leaf_id"])

  # sql create statement for the syntactic_link table
  sql_create_statement = \
"""
create table lemma
(
  id varchar(255) not null,
  input_string varchar(255),
  b_transliteration varchar(255),
  comment varchar(255),
  lemma_index varchar(255),
  lemma_offset varchar(255),
  unvocalized_string varchar(255),
  vocalized_string varchar(255),
  vocalized_input varchar(255),
  pos varchar(255),
  gloss varchar(255),
  lemma varchar(255),
  coarse_sense varchar(16),
  leaf_id varchar(255),
  foreign key (leaf_id) references tree.id
)
default character set utf8;
"""


  # sql insert statement for the syntactic_link table
  sql_insert_statement = \
"""
insert into lemma
(
  id,
  input_string,
  b_transliteration,
  comment,
  lemma_index,
  lemma_offset,
  unvocalized_string,
  vocalized_string,
  vocalized_input,
  pos,
  gloss,
  lemma,
  coarse_sense,
  leaf_id
) values(%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
"""


  def write_to_db(self, cursor):
    data = [(self.id, self.input_string, self.b_transliteration, self.comment, self.index,
             self.offset, self.unvocalized_string, self.vocalized_string, self.vocalized_input,
             self.pos, self.gloss, self.lemma, self.coarse_sense, self.leaf_id)]
    
    cursor.executemany("%s" % (self.__class__.sql_insert_statement), data)









def iterate_trees(string_seq):
    """

    given string_seq which is a sequence of strings, read from
    string_seq and produce strings one at a time that represent trees.

    """

    return [x for x in _iterate_trees_helper(string_seq) if x.strip()]










def _iterate_trees_helper(string_seq):

    parens = 0
    cur_parse = []

    for s in string_seq:
        if (s.startswith(";") or s.startswith("<") or s.startswith("*")) and s.endswith("\n"):
            continue # ignore comments and sgml

        for c in s:
            if c == "(" and parens == 0 and cur_parse:
                yield "".join(cur_parse)
                cur_parse = []

            cur_parse.append(c)

            if c == "(":
                parens += 1
            elif c == ")":
                parens -= 1

                if parens == 0:
                    yield "".join(cur_parse).strip()
                    cur_parse = []

    if parens != 0:
        raise Exception("Parens should have been zero at end, were %s" % parens)
    if "".join(cur_parse).strip():
        raise Exception("curparse should have been empty at end, was %s" % cur_parse)










class InvalidSexprException(Exception):
    def __init__(self, sexpr, parent=None):
        self.sexpr = sexpr
        self.parent = parent

    def __str__(self):

        ns = ""
        ns += self.sexpr
        if self.parent:
            ns += "\n\n"
            ns += str(self.parent)
        return ns










def parse_sexpr(s):
    """ turn an s-expression into a tree of lists:

    (a (b c) d) -> [a, [b, c], d]

    uses spaces and parens only -- no way to have a token with a space in it

    """
    s = s.replace("\n", " ").replace("\t"," ").strip()

    if not s.startswith("(") and not s.endswith(")"):
        return s
    elif s.startswith("(") and s.endswith(")"):
        tr = []
        cur = []
        parens = 0
        for c in s[1:-1].strip() + " ":
            if c == "(":
                parens += 1
            elif c == ")":
                parens -= 1
            elif c == " " and cur:
                if parens == 0:
                    try:
                        x = parse_sexpr("".join(cur))
                    except InvalidSexprException, e:
                        raise InvalidSexprException("Parent: %s" % s, e)

                    if x:
                        tr.append(x)
                    cur = []

            cur.append(c)

        if (cur and cur != [" "]) or parens != 0:
            raise InvalidSexprException("Invalid s-expression: " + s + " note: %s" % "".join(cur) + " parens: %s" % parens)

        return tr
    else:
        raise InvalidSexprException("Invalid s-expression: \n" + s)










def unparse_sexpr(l):
    if type(l) == type([]):
        return "(" + " ".join(unparse_sexpr(a) for a in l) + ")"
    return str(l)










def pretty_print_tree_string(a_tree_string, offset=''):

    if not a_tree_string.strip():
        return ""

    # Maximum depth we're prepared for in trees
    maxdepth=100
    maxindent=300

    # Table of indentation at tree depth
    depth_to_indent = [0 for i in xrange(maxdepth)]

    # Initialize indent_string[i] to be a string of i spaces
    indent_string = ['' for i in xrange(maxindent)]
    for i in xrange(maxindent-1):
        indent_string[i+1] = indent_string[i] + ' '

    # RE object for split that matches on a ')' followed by not a ')', but only consumes the ')'
    close_paren = re.compile(r'\)(?=\s*[^\s\)])')

    # RE object to pick up first on this line(should be only) POS tag and the word of each lexical leaf of the tree
    lexical_leaf = re.compile(r'\((?P<tag>[^\s\)\(]+)\s+(?P<word>[^\s\)\(]+)\)')

    # RE object to parse OntoNotes Normal Form tree lines:
    a_tree = a_tree_string

    pp_tree = ""

    def treeindent(depth):
        return indent_string[depth_to_indent[depth]]+offset  #Indent to appropriate point


    current_depth = 0
    for frag in  close_paren.split(a_tree):  #Split input into lines ending with a lexical item
        if frag[-1]!= '\n':
            frag=frag+')'
        else: frag=frag[0:-1]

        #Indent to appropriate point
        pp_tree += treeindent(current_depth)

        pfrag = ""
        for pfrag in (frag).split('(')[1:]:         # Split line into segments each beginning with an '('
            pfrag='('+pfrag                         # Restore deleted initial '('
            pp_tree += pfrag                      # Print each
            current_depth=current_depth+1           # Up the current depth count

            # Remember how far to indent following items at this depth
            depth_to_indent[current_depth]=depth_to_indent[current_depth-1]+len(pfrag)

        current_depth=current_depth-pfrag.count(')')    # Correct depth given closing parens
        if current_depth<=0:
            pp_tree += ''            # Separate toplevel trees with blank lines

        pp_tree += '\n'              # Print CRLF


    return re.sub("\)$", "", pp_tree)






DONT_DELETE_TREES = True










def car(sp):
    return sp[0]










def cdr(sp):
    return sp[1:]










def split_node(sp):
    return car(sp), cdr(sp)










def is_leaf(sp):
    return len(sp) == 2 and type(sp[1]) != type([])


transformations = {}










def pp(sexpr, out_text=False):
    """ pretty print the S-expr, or just spit text out if out_text is true

    out_text also skips traces

    """
    if not out_text:
        return pretty_print_tree_string(unparse_sexpr(sexpr))
    else:
        words = [word for tag, word in all_leaves(sexpr)
                 if tag != "-NONE-"] # skip traces

        return "\n".join(words)






def transforms(transformation):
    assert transformation.startswith("+") or transformation.startswith("-")
    def regfunc(func):
        transformations[transformation] = func
        return func
    return regfunc










def require(b):
    if not b:
        raise Exception("Failed Requirement")










@transforms("-edited")
def remove_edits(sp):
    """Remove subtrees tagged 'EDITED' (disfluencies) """

    return remove_tagger(sp, "EDITED")










@transforms("-trace")
def remove_edits(sp):
    """Remove traces part of speech tagged '-NONE-' """

    return remove_tagger(sp, "-NONE-")










@transforms("-phrase-tags")
def all_leaves(sp):
    """Make a tree of just the leaves

    .. code-block: scheme

        (TOP (S (NP-SBJ (NNP Zambia))
            (VP (VBD had)
                (ADVP-TMP (RB previously))
                (VP (VBD lost)
                    (NP (PRP$ its)
                        (RB away)
                        (VBD game))
                    (NP-ADV (NP (CD 0))
                            (PP (SYM -)
                                (NP (CD 1))))))
            (. .)))

    becomes

    .. code-block: scheme

        ( (NNP Zambia)
          (VBD Had)
          (RB Previously)
          (VBD lost)
          (PRP$ its)
          (RB away)
          (VBG game)
          (CD 0)
          (SYM -)
          (CD 0) )

    """

    tag, rest = split_node(sp)
    if is_leaf(sp):
        return [[tag, rest[0]]]

    tr = []
    for x in rest:
        tr.extend(all_leaves(x))
    return tr










def remove_tagger(sp, tag_to_remove):
    """ remove tag_to_remove from sp, culling empty branches """
    def callback(tag, rest):
        return tag == tag_to_remove
    return remover(sp, callback)










def remover(sp, callback):
    tag, rest = split_node(sp)
    if callback(tag, rest):
        return []
    if is_leaf(sp):
        return sp

    new_rest = [y for y in [remover(x, callback) for x in rest] if y]

    if not new_rest:
        return []
    return [tag] + new_rest










def pad_items_in_list(a_list, a_character=None):
    """
    this function will return the same list with the right amount of
    padding equal to two spaces on each side of the widest string. it
    will perform right justification.

    if the optional character is specified, then it will do a
    centering around the character in the process of padding.
    left/right justification does not work with this option.
    """

    if(a_character != None):
        for an_item in a_list:
            if(an_item.find(a_character) == -1):
                a_character = None
                break

    if(a_character != None):
        lmax=0
        rmax=0
        for an_item in a_list:
            an_item = an_item.strip()

            lf = an_item.find("*")
            if(lmax < lf):
                lmax = lf

            rf = len(an_item) - an_item.find("*")
            if(rmax < rf):
                rmax = rf



        i=0
        for i in range(0, len(a_list)):
            a_list[i] = a_list[i].strip()

            x = a_list[i].find(a_character)

            len_i=len(a_list[i])

            a_list[i] = " "*(lmax-x+2) + a_list[i]
            a_list[i] = a_list[i] + " "*(rmax-len_i+x+2)

    else:
        max=0
        for an_item in a_list:
            an_item = an_item.strip()
            x = len(an_item)
            if(max < x):
                max = x

        i=0
        for i in range(0, len(a_list)):
            a_list[i] = a_list[i].strip()

            if(a_list[i].endswith("*") or
               a_list[i].endswith("-") or
               a_list[i][-1] in string.digits ):
                a_list[i] = "%s " % (a_list[i])

            a_list[i] = a_list[i].rjust(max+2)

    return a_list










def rows2columns(matrix):
    columns = []

    for row in matrix:
        c=0
        for cell in row:
            if(c == len(columns)):
                columns.append([])

            columns[c].append(cell)
            c = c + 1

    return columns










def pretty_print_table(rows, separator=None, out_file=None):

    # cells is the matrix
    r_c_matrix = []
    for row in rows:
        r_c_matrix.append(row.split())


    c_r_matrix = rows2columns(r_c_matrix)


    for i in range(0, len(c_r_matrix)):

        if(i==5 or i>10):
            padding_character=separator
        else:
            padding_character=None

        c_r_matrix[i] = pad_items_in_list(c_r_matrix[i], padding_character)

    r_c_matrix = rows2columns(c_r_matrix)

    if(out_file == None):
        for row in r_c_matrix:
            print " ".join(row).strip()
        print

    elif(out_file == "-"):
        rows=[]
        for row in r_c_matrix:
            rows.append(" ".join(row).strip())
        return "%s\n" % ("\n".join(rows))

    else:
        raise NotImplementedError("this functionality has not yet been implemented")










def start(input_fname, conll_fname, output_fname, encoding, changes):
    """ apply changes in order to the trees in input_fname, write to output_fname """


    out_text = False
    if "--text" in changes:
        out_text = True
        changes.remove("--text")

    out = []
    with codecs.open(input_fname, "r", encoding) as inf:

        for a_tree in iterate_trees(inf):
            sexpr = parse_sexpr(a_tree)
            for change in changes:
                if not sexpr:
                    continue

                try:
                    change_func = transformations[change]
                except KeyError:
                    raise Exception("Invalid argument '%s' for change.  Allowed changes are: %s" % (change, transformations.keys()))

                try:
                    old_sexpr = sexpr[:]
                    sexpr = change_func(sexpr)
                except Exception:
                    sys.stderr.write("ERR in %s\n\nTree:\n%s\n\nInput sexpr:\n%s\n" % (change, a_tree, pp(sexpr)))
                    raise


                if not sexpr and DONT_DELETE_TREES:
                    nullp = ["XX", "nullp"]
                    if old_sexpr and old_sexpr[0] == "TOP":
                        sexpr = ["TOP", nullp]
                    else:
                        sexpr = nullp

            if sexpr:
                out.append(pp(sexpr, out_text))



    w_list = []
    for o in out:
        w_list.append(o.split("\n"))


    num_words = 0
    for a_word_list in w_list:
      for a_word in a_word_list:
        num_words = num_words + 1

    debug("number of words: %d\n" % (num_words), DEBUG, MAX_VERBOSITY)
    debug("input_fname: %s" % (input_fname), DEBUG, MAX_VERBOSITY)


    is_arabic = False
    a_list_of_lemmas = []

    if re.search('data%s+arabic%s+annotations' % (os.sep, os.sep), input_fname):
      is_arabic = True

      


    if is_arabic is True:
      lemma_fname = re.sub("\.parse$", ".lemma", input_fname)
      debug("lemma_fname: %s" % (lemma_fname), DEBUG, MAX_VERBOSITY)

      if os.path.exists(lemma_fname):
        lemma_file = codecs.open(lemma_fname, "r", "utf-8")

        actual_word_list = []
        buckwalter_word_list = []
        lemma_list = []

        input_string_regex = re.compile(r"^\s*INPUT STRING:(.*)", re.U|re.MULTILINE)
        buckwalter_regex = re.compile(r"^\s*IS_TRANS:(.*)", re.U|re.MULTILINE)
        comment_regex = re.compile(r"^\s*COMMENT:(.*)", re.U|re.MULTILINE)
        index_regex = re.compile(r"^\s*INDEX:(.*)", re.U|re.MULTILINE)
        offsets_regex = re.compile(r"^\s*OFFSETS:(.*)", re.U|re.MULTILINE)
        unvocalized_string_regex = re.compile(r"^\s*UNVOCALIZED:(.*)", re.U|re.MULTILINE)
        vocalized_string_regex = re.compile(r"^\s*VOCALIZED:(.*)", re.U|re.MULTILINE)
        vocalized_input_string_regex = re.compile(r"^\s*VOC_STRING:(.*)", re.U|re.MULTILINE)
        pos_string_regex = re.compile(r"^\s*POS:(.*)", re.U|re.MULTILINE)
        gloss_string_regex = re.compile(r"^\s*GLOSS:(.*)", re.U|re.MULTILINE)
        lemma_regex = re.compile(r"LEMMA:\s+\[([^\]]*)\]", re.U|re.MULTILINE)

        lemma_file_lines = lemma_file.readlines()

        list_of_lemma_blocks = []

        i=0
        lemma_block = ""
        list_of_lemma_blocks = []
        while(i<len(lemma_file_lines)):
          input_string_regex_match = input_string_regex.findall(lemma_file_lines[i])

          if(input_string_regex_match != []):
            while(i<len(lemma_file_lines) and lemma_file_lines[i].strip() != ""):
              lemma_block = "%s%s" % (lemma_block, lemma_file_lines[i])
              i=i+1

          if(lemma_block.strip() != ""):
            list_of_lemma_blocks.append(lemma_block)

          lemma_block = ""
          i=i+1


        list_of_input_strings = []
        list_of_b_transliterations = []
        list_of_comments = []
        list_of_indices = []
        list_of_offsets = []
        list_of_unvocalized_strings = []
        list_of_vocalized_strings = []
        list_of_vocalized_inputs = []
        list_of_pos = []
        list_of_glosses = []
        list_of_lemmas = []

        for lemma_block in list_of_lemma_blocks:
          for a_list, a_regex, a_name in [[list_of_input_strings, input_string_regex, "input"],
                                          [list_of_b_transliterations, buckwalter_regex, "transliterations"],
                                          [list_of_comments, comment_regex, "comment"],
                                          [list_of_indices, index_regex, "indecies"],
                                          [list_of_offsets, offsets_regex, "offsets"],
                                          [list_of_unvocalized_strings, unvocalized_string_regex, "unvocalized_strings"],
                                          [list_of_vocalized_strings, vocalized_string_regex, "vocalized_strings"],
                                          [list_of_vocalized_inputs, vocalized_input_string_regex, "vocalized_inputs"],
                                          [list_of_pos, pos_string_regex, "pos_strings"],
                                          [list_of_glosses, gloss_string_regex, "gloss_strings"],
                                          [list_of_lemmas, lemma_regex, "lemmas"]]:
            try:
              a_list.append(a_regex.findall(lemma_block)[0])
            except IndexError:
              if a_name == "lemmas":
                list_of_lemmas.append("lemma_not_set")
              else:
                raise Exception("Didn't find any %s in %s (%s)" % (a_name, ("\n" + lemma_block).replace("\n", "\n     "), lemma_fname))



        # temporarily copying the lists to another list used earlier
        actual_word_list = [] + list_of_input_strings
        buckwalter_word_list = [] + list_of_b_transliterations
        lemma_list = [] + list_of_lemmas


        debug("len(actual_word_list): %s\n" % (len(actual_word_list)), DEBUG, MAX_VERBOSITY)
        debug("actual_word_list: %s\n" % (actual_word_list), DEBUG, MAX_VERBOSITY)
        debug("len(buckwalter_word_list): %s\n" % (len(buckwalter_word_list)), DEBUG, MAX_VERBOSITY)
        debug("buckwalter_word_list: %s\n" % (buckwalter_word_list), DEBUG, MAX_VERBOSITY)
        debug("len(lemma_list): %s\n" % (len(lemma_list)), DEBUG, MAX_VERBOSITY)
        debug("lemma_list: %s\n" % (lemma_list), DEBUG, MAX_VERBOSITY)



        if(len(actual_word_list) != len(buckwalter_word_list)
           or
           len(actual_word_list) != len(lemma_list)):
            debug("len(actual_word_list): %s\n" % (len(actual_word_list)), DEBUG, MAX_VERBOSITY)
            debug("len(buckwalter_word_list): %s\n" % (len(buckwalter_word_list)), DEBUG, MAX_VERBOSITY)
            debug("len(lemma_list): %s\n" % (len(lemma_list)), DEBUG, MAX_VERBOSITY)
            raise Exception("the three lists -- actual word, buckwalter word, and lemma should be the same length, or else some information might be missing from the .lemma file")

        for i in range(0, len(actual_word_list)):
            if(lemma_list[i] == "DEFAULT"
               or
               buckwalter_word_list[i] == ""):
                debug("%s %s %s\n" % (actual_word_list[i].rjust(50), buckwalter_word_list[i].rjust(50), lemma_list[i].rjust(50)), DEBUG, MAX_VERBOSITY)






        for i in range(0, len(actual_word_list)):

         lemma_lemma = list_of_lemmas[i]

         coarse_sense = ""
         if "_" in lemma_lemma and lemma_lemma != "lemma_not_set":
           try:
             lemma_lemma, coarse_sense = lemma_lemma.split("_")
           except ValueError:
             raise 

         lemma_object = lemma(list_of_input_strings[i],
                              list_of_b_transliterations[i],
                              list_of_comments[i],
                              list_of_indices[i],
                              list_of_offsets[i],
                              list_of_unvocalized_strings[i],
                              list_of_vocalized_strings[i],
                              list_of_vocalized_inputs[i],
                              list_of_pos[i],
                              list_of_glosses[i],
                              lemma_lemma, 
                              coarse_sense,
                              i)

         debug("lemma_object: %s" % (lemma_object), DEBUG, MAX_VERBOSITY)
         a_list_of_lemmas.append(lemma_object)





    f=codecs.open(output_fname, "w", encoding)
    f.close()

    sentences = []
    i=0
    w=0
    conll_file = codecs.open(conll_fname, "r", encoding)
    for line in conll_file:
        if(line.strip() == ""):

            assert len(sentences) == len(w_list[i]), "the example should contain the same number of words as the words in the parse"


            if(a_list_of_lemmas != []):
              assert len(a_list_of_lemmas) == num_words, "the list of lemmas does not match the list of words. please report this issue."



            rows=[]
            c=0
            for columns in sentences:

                if a_list_of_lemmas != []:
                    columns[WORD_COLUMN] = "%s#%s#%s#%s" % (w_list[i][c], a_list_of_lemmas[w].lemma.strip(), a_list_of_lemmas[w].unvocalized_string.strip(), a_list_of_lemmas[w].vocalized_string.strip())

                    if DEBUG:
                      if columns[LEMMA_COLUMN] == a_list_of_lemmas[w].lemma.strip():
                        print "found the same lemma"
                      else:
                        raise Exception("Something is wrong: %s %s %s" % (columns[LEMMA_COLUMN], a_list_of_lemmas[w].lemma.strip(), " ".join(columns)))
                      
                    columns[LEMMA_COLUMN] = a_list_of_lemmas[w].lemma.strip()
                else:
                    columns[WORD_COLUMN] = w_list[i][c]

                rows.append(" ".join(columns))
                c=c+1
                w=w+1

            pretty_print_table_string = pretty_print_table(rows, out_file="-")

            if output_fname == "-":
                print pretty_print_table_string
            else:
                with codecs.open(output_fname, "a", encoding) as outf:
                    outf.write("%s\n" % (pretty_print_table_string))

            sentences = []
            i=i+1

        elif(line.startswith("#")):
            if output_fname == "-":
                print line.strip()
            else:
                with codecs.open(output_fname, "a", encoding) as outf:
                    outf.write("%s\n" % (line.strip()))
        else:
            sentences.append(line.split())









if __name__ == "__main__":

    encoding = "utf8"
    if "--gb18030" in sys.argv:
        encoding="gb18030"
        sys.argv.remove("--gb18030")

    if len(sys.argv[1:]) == 2 and sys.argv[1] in ["--help", "-h"] and sys.argv[2] in transformations:
        print
        print "  ", transformations[sys.argv[2]].__doc__
    elif not sys.argv[1:] or "--help" in sys.argv[1:] or "-h" in sys.argv[1:]:
        print
        print "-"*120
        print "Usage:  python skeleton2conll.py <ontonotes-parse-file> <input-skel-file> <conll-output-file> [transformations] ..."
        print "\nAllowed transforms:"

        max_key_len = max(len(t) for t in transformations) + 1 # +1 for colon

        for key in transformations:
            print "   %s %s" %(("%s:"%key).rjust(max_key_len),
                               transformations[key].__doc__.split("\n")[0])

        print "   %s %s" % ("--text:".rjust(max_key_len),
                            "Produce text output instead of parse trees")
        print
        print
        print "Example:"
        print "python skeleton2conll.py <ontonotes-release-directory>/data/.../bc/cnn/00/cnn_0000.parse conll-2011/dev/data/english/annotations/bc/cnn/00/cnn_0000.v0_gold_skel conll-2011/dev/data/english/annotations/bc/cnn/00/cnn_0000.v0_gold_conll -edited --text"
        print "-"*120
    else:
        input_fname, conll_fname, output_fname, changes = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4:]
        start(input_fname, conll_fname, output_fname, encoding, changes)
