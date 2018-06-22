#!/usr/bin/perl

BEGIN {
  $d = $0;
  $d =~ s/\/[^\/][^\/]*$//g;

  if ($d eq $0) {
    unshift(@INC, "lib");
  }
  else {
    unshift(@INC, $d . "/lib");
  }
}

use strict;
use CorScorer;

if (@ARGV < 3) {
  print q|
use: scorer.pl <metric> <keys_file> <response_file> [name]

  metric: the metric desired to score the results:
    muc: MUCScorer (Vilain et al, 1995)
    bcub: B-Cubed (Bagga and Baldwin, 1998)
    ceafm: CEAF (Luo et al, 2005) using mention-based similarity
    ceafe: CEAF (Luo et al, 2005) using entity-based similarity
    blanc: BLANC
    all: uses all the metrics to score

  keys_file: file with expected coreference chains in SemEval format

  response_file: file with output of coreference system (SemEval format)

  name: [optional] the name of the document to score. If name is not
    given, all the documents in the dataset will be scored. If given
    name is "none" then all the documents are scored but only total
    results are shown.

|;
  exit;
}

my $metric = shift(@ARGV);
if ($metric !~ /^(muc|bcub|ceafm|ceafe|blanc|all)/i) {
  print "Invalid metric\n";
  exit;
}

if ($metric eq 'all') {
  foreach my $m ('muc', 'bcub', 'ceafm', 'ceafe', 'blanc') {
    print "\nMETRIC $m:\n";
    &CorScorer::Score($m, @ARGV);
  }
}
else {
  &CorScorer::Score($metric, @ARGV);
}

