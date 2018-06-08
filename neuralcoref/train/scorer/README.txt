NAME
   CorScorer: Perl package for scoring coreference resolution systems
   using different metrics.


VERSION
   v8.01 -- reference implementations of MUC, B-cubed, CEAF and BLANC metrics.


CHANGES SINCE v8.0
   - fixed a bug that crashed the BLANC scorer when a duplicate singleton
     mention was present in the response.

INSTALLATION
   Requirements:
      1. Perl: downloadable from http://perl.org
      2. Algorithm-Munkres: included in this package and downloadable
         from CPAN http://search.cpan.org/~tpederse/Algorithm-Munkres-0.08

USE
   This package is distributed with two scripts to execute the scorer from
   the command line.

   Windows (tm): scorer.bat
   Linux: scorer.pl


SYNOPSIS
   use CorScorer;

   $metric = 'ceafm';

   # Scores the whole dataset
   &CorScorer::Score($metric, $keys_file, $response_file);

   # Scores one file
   &CorScorer::Score($metric, $keys_file, $response_file, $name);


INPUT
   metric: the metric desired to score the results:
     muc: MUCScorer (Vilain et al, 1995)
     bcub: B-Cubed (Bagga and Baldwin, 1998)
     ceafm: CEAF (Luo et al., 2005) using mention-based similarity
     ceafe: CEAF (Luo et al., 2005) using entity-based similarity
     blanc: BLANC (Luo et al., 2014) BLANC metric for gold and predicted mentions
     all: uses all the metrics to score

   keys_file: file with expected coreference chains in CoNLL-2011/2012 format

   response_file: file with output of coreference system (CoNLL-2011/2012 format)

   name: [optional] the name of the document to score. If name is not
     given, all the documents in the dataset will be scored. If given
     name is "none" then all the documents are scored but only total
     results are shown.


OUTPUT
   The score subroutine returns an array with four values in this order:
   1) Recall numerator
   2) Recall denominator
   3) Precision numerator
   4) Precision denominator

   Also recall, precision and F1 are printed in the standard output when variable
   $VERBOSE is not null.

   Final scores:
   Recall = recall_numerator / recall_denominator
   Precision = precision_numerator / precision_denominator
   F1 = 2 * Recall * Precision / (Recall + Precision)

   Identification of mentions
   An scorer for identification of mentions (recall, precision and F1) is also included.
   Mentions from system response are compared with key mentions. This version performs
   strict mention matching as was used in the CoNLL-2011 and 2012 shared tasks.

AUTHORS
   Emili Sapena, Universitat Politècnica de Catalunya, http://www.lsi.upc.edu/~esapena, esapena <at> lsi.upc.edu
   Sameer Pradhan, sameer.pradhan <at> childrens.harvard.edu
   Sebastian Martschat, sebastian.martschat <at> h-its.org
   Xiaoqiang Luo, xql <at> google.com

COPYRIGHT AND LICENSE
   Copyright (C) 2009-2011, Emili Sapena esapena <at> lsi.upc.edu
                 2011-2014, Sameer Pradhan sameer.pradhan <at> childrens.harvard.edu

   This program is free software; you can redistribute it and/or modify it
   under the terms of the GNU General Public License as published by the
   Free Software Foundation; either version 2 of the License, or (at your
   option) any later version. This program is distributed in the hope that
   it will be useful, but WITHOUT ANY WARRANTY; without even the implied
   warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
   GNU General Public License for more details.

   You should have received a copy of the GNU General Public License along
   with this program; if not, write to the Free Software Foundation, Inc.,
   59 Temple Place - Suite 330, Boston, MA 02111-1307, USA.

