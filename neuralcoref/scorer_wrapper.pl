#!/usr/bin/perl

BEGIN {
  $d = $0;
  $d =~ s/\/[^\/][^\/]*$//g;

  if ($d eq $0) {
    unshift(@INC, "scorer/lib");
  }
  else {
    unshift(@INC, $d . "/scorer/lib");
  }
}

use strict;
use CorScorer;

my $metric = shift(@ARGV);
if ($metric !~ /^(muc|bcub|ceafm|ceafe|blanc|all)/i) {
  print "Invalid metric\n";
  exit;
}

if ($metric eq 'all') {
  foreach my $m ('muc', 'bcub', 'ceafm', 'ceafe', 'blanc') {
#    print "\nMETRIC $m:\n";
    my ($acumNR, $acumDR, $acumNP, $acumDP, $identNR, $identDR, $identNP, $identDP) = &CorScorer::Score($m, @ARGV);
	print "$acumNR $acumDR $acumNP $acumDP\n$identNR $identDR $identNP $identDP";
  }
}
else {
  my ($acumNR, $acumDR, $acumNP, $acumDP, $identNR, $identDR, $identNP, $identDP) = &CorScorer::Score($metric, @ARGV);
  print "$acumNR $acumDR $acumNP $acumDP\n$identNR $identDR $identNP $identDP";
}

