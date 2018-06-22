package CorScorer;

# Copyright (C) 2009-2011, Emili Sapena esapena <at> lsi.upc.edu
#               2011-2014, Sameer Pradhan <at> childrens.harvard.edu
#
# This program is free software; you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by the
# Free Software Foundation; either version 2 of the License, or (at your
# option) any later version. This program is distributed in the hope that
# it will be useful, but WITHOUT ANY WARRANTY; without even the implied
# warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License along
# with this program; if not, write to the Free Software Foundation, Inc.,
# 59 Temple Place - Suite 330, Boston, MA 02111-1307, USA.
#
# Modified in 2013 for v1.07 by Sebastian Martschat,
#   sebastian.martschat <at> h-its.org
#
# Revised in July, 2013 by Xiaoqiang Luo (xql <at> google.com) to create v6.0.
# See comments under $VERSION for modifications.
#
# Revised in March, 2014 by Sameer Pradhan (sameer.pradhan <at> childrens.harvard.edu)
# to implement the BLANC metric for predicted mentions


use strict;
use Algorithm::Munkres;
use Data::Dumper;

#use Algorithm::Combinatorics qw(combinations);
use Math::Combinatorics;
use Cwd;

our $VERSION = '8.01';
print "version: " . $VERSION . " " . Cwd::realpath(__FILE__) . "\n";

##
# 8.01 fixed a bug that crashed the the BLANC scoring when duplicate
#      (potentially singleton) mentions were present in the
#      response. as part of the fix, wee will allow a maximum of 10
#      duplicate mentions in response, but if there are more, than it
#      is a sign of a systematic error/manipulation and we will refuse
#      to score that run.

#  8.0 added code to compute the BLANC metric (generalized for both gold
#      and system mentions (Luo et al., 2014)
#
#  7.0 Removed code to compute *_cs metrics
#
#  6.0 The directory hosting the scorer is under v6 and internal $VERSION is
#      set to "6.0."
#      Changes:
#      - 'ceafm', 'ceafe' and 'bcub' in the previous version are renamed
#        'ceafm_cs', 'ceafe_cs', and 'bcub_cs', respectively.
#      - 'ceafm', 'ceafe' and 'bcub' are implemented without (Cai&Strube 2010)
#         modification. These metrics can handle twinless mentions and entities
#         just fine.
#
# 1.07 Modifications to implement BCUB and CEAFM
#      exactly as proposed by (Cai & Strube, 2010).
# 1.06 ?
# 1.05 Modification of IdentifMentions in order to correctly evaluate the
#     outputs with detected mentions. Based on (Cai & Strubbe, 2010)
# 1.04 Some output corrections in BLANC functions. Changed package name to "Scorer"
# 1.03 Detects mentions that start in a document but do not end
# 1.02 Corrected BCUB bug. It fails when the key file does not have any mention

# global variables
my $VERBOSE         = 0;#2;#3;
my $HEAD_COLUMN     = 8;
my $RESPONSE_COLUMN = -1;
my $KEY_COLUMN      = -1;

# Score. Scores the results of a coreference resolution system
# Input: Metric, keys file, response file, [name]
#        Metric: the metric desired to evaluate:
#                muc: MUCScorer (Vilain et al, 1995)
#                bcub: B-Cubed (Bagga and Baldwin, 1998)
#                ceafm: CEAF (Luo et al, 2005) using mention-based similarity
#                ceafe: CEAF (Luo et al, 2005) using entity-based similarity
#        keys file: file with expected coreference chains in SemEval format
#        response file: file with output of coreference system (SemEval format)
#        name: [optional] the name of the document to score. If name is not
#              given, all the documents in the dataset will be scored.
#
# Output: an array with numerators and denominators of recall and precision
#         (recall_num, recall_den, precision_num, precision_den)
#
#   Final scores:
# Recall = recall_num / recall_den
# Precision = precision_num / precision_den
# F1 = 2 * Recall * Precision / (Recall + Precision)
sub Score {
  my ($metric, $kFile, $rFile, $name) = @_;
	our $repeated_mentions = 0;

  if (lc($metric) eq 'blanc') {
    return ScoreBLANC($kFile, $rFile, $name);
  }

  my %idenTotals =
    (recallDen => 0, recallNum => 0, precisionDen => 0, precisionNum => 0);
  my ($acumNR, $acumDR, $acumNP, $acumDP) = (0, 0, 0, 0);

  if (defined($name) && $name ne 'none') {
    print "$name:\n" if ($VERBOSE);
    my $keys     = GetCoreference($kFile, $KEY_COLUMN,      $name);
    my $response = GetCoreference($rFile, $RESPONSE_COLUMN, $name);
    my (
      $keyChains, $keyChainsWithSingletonsFromResponse,
      $responseChains, $responseChainsWithoutMentionsNotInKey,
      $keyChainsOrig, $responseChainsOrig
    ) = IdentifMentions($keys, $response, \%idenTotals);
    ($acumNR, $acumDR, $acumNP, $acumDP) = Eval(
      $metric,                                $keyChains,
      $keyChainsWithSingletonsFromResponse,   $responseChains,
      $responseChainsWithoutMentionsNotInKey, $keyChainsOrig,
      $responseChainsOrig
    );
  }
  else {
    my $kIndexNames = GetFileNames($kFile);
    my $rIndexNames = GetFileNames($rFile);

    $VERBOSE = 0 if ($name eq 'none');
    foreach my $iname (keys(%{$kIndexNames})) {
      my $keys =
        GetCoreference($kFile, $KEY_COLUMN, $iname, $kIndexNames->{$iname});
      my $response = GetCoreference($rFile, $RESPONSE_COLUMN, $iname,
        $rIndexNames->{$iname});

      print "$iname:\n" if ($VERBOSE);
      my (
        $keyChains,      $keyChainsWithSingletonsFromResponse,
        $responseChains, $responseChainsWithoutMentionsNotInKey,
        $keyChainsOrig,  $responseChainsOrig
      ) = IdentifMentions($keys, $response, \%idenTotals);
      my ($nr, $dr, $np, $dp) = Eval(
        $metric,                                $keyChains,
        $keyChainsWithSingletonsFromResponse,   $responseChains,
        $responseChainsWithoutMentionsNotInKey, $keyChainsOrig,
        $responseChainsOrig
      );

      $acumNR += $nr;
      $acumDR += $dr;
      $acumNP += $np;
      $acumDP += $dp;
    }
  }

  if ($VERBOSE || $name eq 'none') {
    print "\n====== TOTALS =======\n";
    print "Identification of Mentions: ";
    ShowRPF(
      $idenTotals{recallNum},    $idenTotals{recallDen},
      $idenTotals{precisionNum}, $idenTotals{precisionDen}
    );
    print "Coreference: ";
    ShowRPF($acumNR, $acumDR, $acumNP, $acumDP);
  }

  return ($acumNR, $acumDR, $acumNP, $acumDP,
          $idenTotals{recallNum}, $idenTotals{recallDen},
          $idenTotals{precisionNum}, $idenTotals{precisionDen});
}

sub GetIndex {
  my ($ind, $i) = @_;
  if (!defined($ind->{$i})) {
    my $n = $ind->{nexti} || 0;
    $ind->{$i} = $n;
    $n++;
    $ind->{nexti} = $n;
  }

  return $ind->{$i};
}

# Get the coreference information from column $column of the file $file
# If $name is defined, only keys between "#begin document $name" and
# "#end file $name" are taken.
# The output is an array of entites, where each entity is an array
# of mentions and each mention is an array with two values corresponding
# to the mention's begin and end. For example:
# @entities = ( [ [1,3], [45,45], [57,62] ], # <-- entity 0
#               [ [5,5], [25,27], [31,31] ], # <-- entity 1
# ...
# );
# entity 0 is composed of 3 mentions: from token 1 to 3, token 45 and
# from token 57 to 62 (both included)
#
# if $name is not specified, the output is a hash including each file
# found in the document:
# $coref{$file} = \@entities
sub GetCoreference {
  my ($file, $column, $name, $pos) = @_;
  my %coref;
  my %ind;

  open(F, $file) || die "Can not open $file: $!";
  if ($pos) {
    seek(F, $pos, 0);
  }
  my $fName;
  my $getout = 0;
  do {
    # look for the begin of a file
    while (my $l = <F>) {
      chomp($l);
      $l =~ s/\r$//;    # m$ format jokes
      if ($l =~ /^\#\s*begin document (.*?)$/) {
        if (defined($name)) {
          if ($name eq $1) {
            $fName  = $name;
            $getout = 1;
            last;
          }
        }
        else {
          $fName = $1;
          last;
        }
      }
    }
    print "====> $fName:\n" if ($VERBOSE > 1);

    # Extract the keys from the file until #end is found
    my $lnumber = 0;
    my @entities;
    my @half;
    my @head;
    my @sentId;
    while (my $l = <F>) {
      chomp($l);
			$l =~ s/^\s+$//;
      next if ($l eq '');
      if ($l =~ /\#\s*end document/) {
        foreach my $h (@half) {
          if (defined($h) && @$h) {
            die "Error: some mentions in the document do not close\n";
          }
        }
        last;
      }
      my @columns = split(/\t/, $l);
      my $cInfo = $columns[$column];
      push(@head,   $columns[$HEAD_COLUMN]);
      push(@sentId, $columns[0]);
      if ($cInfo ne '_') {

        #discard double antecedent
        while ($cInfo =~ s/\((\d+\+\d)\)//) {
          print "Discarded ($1)\n" if ($VERBOSE > 1);
        }

        # one-token mention(s)
        while ($cInfo =~ s/\((\d+)\)//) {
          my $ie = GetIndex(\%ind, $1);
          push(@{$entities[$ie]}, [$lnumber, $lnumber, $lnumber]);
          print "+mention (entity $ie): ($lnumber,$lnumber)\n"
            if ($VERBOSE > 2);
        }

        # begin of mention(s)
        while ($cInfo =~ s/\((\d+)//) {
          my $ie = GetIndex(\%ind, $1);
          push(@{$half[$ie]}, $lnumber);
          print "+init mention (entity $ie): ($lnumber\n" if ($VERBOSE > 2);
        }

        # end of mention(s)
        while ($cInfo =~ s/(\d+)\)//) {
          my $numberie = $1;
          my $ie       = GetIndex(\%ind, $numberie);
          my $start    = pop(@{$half[$ie]});
          if (defined($start)) {
            my $inim  = $sentId[$start];
            my $endm  = $sentId[$lnumber];
            my $tHead = $start;

        # the token whose head is outside the mention is the head of the mention
            for (my $t = $start ; $t <= $lnumber ; $t++) {
              if ($head[$t] < $inim || $head[$t] > $endm) {
                $tHead = $t;
                last;
              }
            }
            push(@{$entities[$ie]}, [$start, $lnumber, $tHead]);
          }
          else {
            die
"Detected the end of a mention [$numberie]($ie) without begin (?,$lnumber)";
          }
          print "+mention (entity $ie): ($start,$lnumber)\n" if ($VERBOSE > 2);

        }
      }
      $lnumber++;
    }

    # verbose
    if ($VERBOSE > 1) {
      print "File $fName:\n";
      for (my $e = 0 ; $e < scalar(@entities) ; $e++) {
        print "Entity $e:";
        foreach my $mention (@{$entities[$e]}) {
          print " ($mention->[0],$mention->[1])";
        }
        print "\n";
      }
    }

    $coref{$fName} = \@entities;
  } while (!$getout && !eof(F));

  if (defined($name)) {
    return $coref{$name};
  }
  return \%coref;
}

sub GetFileNames {
  my $file = shift;
  my %hash;
  my $last = 0;
  open(F, $file) || die "Can not open $file: $!";
  while (my $l = <F>) {
    chomp($l);
    $l =~ s/\r$//;    # m$ format jokes
    if ($l =~ /^\#\s*begin document (.*?)$/) {
      my $name = $1;
      $hash{$name} = $last;
    }
    $last = tell(F);
  }
  close(F);
  return \%hash;
}

sub IdentifMentions {
	my ($keys, $response, $totals) = @_;
  my @kChains;
  my @kChainsWithSingletonsFromResponse;
  my @rChains;
  my @rChainsWithoutMentionsNotInKey;
  my %id;
  my %map;
  my $idCount = 0;
  my @assigned;
  my @kChainsOrig = ();
  my @rChainsOrig = ();

  # for each mention found in keys an ID is generated
  foreach my $entity (@$keys) {
    foreach my $mention (@$entity) {
      if (defined($id{"$mention->[0],$mention->[1]"})) {
        print "Repeated mention in the key: $mention->[0], $mention->[1] ",
          $id{"$mention->[0],$mention->[1]"}, $idCount, "\n";
      }
      $id{"$mention->[0],$mention->[1]"} = $idCount;
      $idCount++;
    }
  }

  # correct identification: Exact bound limits
  my $exact = 0;
  foreach my $entity (@$response) {

    my $i = 0;
    my @remove;
		
    foreach my $mention (@$entity) {
      if (defined($map{"$mention->[0],$mention->[1]"})) {
        print "Repeated mention in the response: $mention->[0], $mention->[1] ",
          $map{"$mention->[0],$mention->[1]"},
          $id{"$mention->[0],$mention->[1]"},
          "\n";
        push(@remove, $i);
				$main::repeated_mentions++;

				if ($main::repeated_mentions > 10)
				{
						print STDERR "Found too many repeated mentions (> 10) in the response, so refusing to score. Please fix the output.\n";
						exit 1;
				}

      }
      elsif (defined($id{"$mention->[0],$mention->[1]"})
        && !$assigned[$id{"$mention->[0],$mention->[1]"}])
      {
        $assigned[$id{"$mention->[0],$mention->[1]"}] = 1;
        $map{"$mention->[0],$mention->[1]"} =
          $id{"$mention->[0],$mention->[1]"};
        $exact++;
      }
      $i++;
    }

    # Remove repeated mentions in the response
    foreach my $i (sort { $b <=> $a } (@remove)) {
      splice(@$entity, $i, 1);
    }
  }


	# now, lets remove any empty elements in the response array after removing
	# potential repeats
	my @another_remove = ();
	my $ii;

	foreach my $eentity (@$response)
	{
			if ( @$eentity == 0)
			{
					push(@another_remove, $ii);
			}
			$ii++;
	}

	foreach my $iii (sort { $b <=> $a } (@another_remove)) {
      splice(@$response, $iii, 1);
	}


  # Partial identificaiton: Inside bounds and including the head
  my $part = 0;

  # Each mention in response not included in keys has a new ID
  my $mresp = 0;
  foreach my $entity (@$response) {
    foreach my $mention (@$entity) {
      my $ini = $mention->[0];
      my $end = $mention->[1];
      if (!defined($map{"$mention->[0],$mention->[1]"})) {
        $map{"$mention->[0],$mention->[1]"} = $idCount;
        $idCount++;
      }
      $mresp++;
    }
  }

  if ($VERBOSE) {
    print "Total key mentions: " . scalar(keys(%id)) . "\n";
    print "Total response mentions: " . scalar(keys(%map)) . "\n";
    print "Strictly correct identified mentions: $exact\n";
    print "Partially correct identified mentions: $part\n";
    print "No identified: " . (scalar(keys(%id)) - $exact - $part) . "\n";
    print "Invented: " . ($idCount - scalar(keys(%id))) . "\n";
  }

  if (defined($totals)) {
    $totals->{recallDen}      += scalar(keys(%id));
    $totals->{recallNum}      += $exact;
    $totals->{precisionDen}   += scalar(keys(%map));
    $totals->{precisionNum}   += $exact;
    $totals->{precisionExact} += $exact;
    $totals->{precisionPart}  += $part;
  }

  # The coreference chains arrays are generated again with ID of mentions
  # instead of token coordenates
  my $e = 0;
  foreach my $entity (@$keys) {
    foreach my $mention (@$entity) {
      push(@{$kChainsOrig[$e]}, $id{"$mention->[0],$mention->[1]"});
      push(@{$kChains[$e]},     $id{"$mention->[0],$mention->[1]"});
    }
    $e++;
  }
  $e = 0;
  foreach my $entity (@$response) {
    foreach my $mention (@$entity) {
      push(@{$rChainsOrig[$e]}, $map{"$mention->[0],$mention->[1]"});
      push(@{$rChains[$e]},     $map{"$mention->[0],$mention->[1]"});
    }
    $e++;
  }

  # In order to use the metrics as in (Cai & Strube, 2010):
  # 1. Include the non-detected key mentions into the response as singletons
  # 2. Discard the detected mentions not included in key resolved as singletons
  # 3a. For computing precision: put twinless system mentions in key
  # 3b. For computing recall: discard twinless system mentions in response

  my $kIndex = Indexa(\@kChains);
  my $rIndex = Indexa(\@rChains);

  # 1. Include the non-detected key mentions into the response as singletons
  my $addkey = 0;
  if (scalar(keys(%id)) - $exact - $part > 0) {
    foreach my $kc (@kChains) {
      foreach my $m (@$kc) {
        if (!defined($rIndex->{$m})) {
          push(@rChains, [$m]);
          $addkey++;
        }
      }
    }
  }

  @kChainsWithSingletonsFromResponse = @kChains;
  @rChainsWithoutMentionsNotInKey    = [];

  # 2. Discard the detected mentions not included in key resolved as singletons
  my $delsin = 0;

  if ($idCount - scalar(keys(%id)) > 0) {
    foreach my $rc (@rChains) {
      if (scalar(@$rc) == 1) {
        if (!defined($kIndex->{$rc->[0]})) {
          @$rc = ();
          $delsin++;
        }
      }
    }
  }

# 3a. For computing precision: put twinless system mentions in key as singletons
  my $addinv = 0;

  if ($idCount - scalar(keys(%id)) > 0) {
    foreach my $rc (@rChains) {
      if (scalar(@$rc) > 1) {
        foreach my $m (@$rc) {
          if (!defined($kIndex->{$m})) {
            push(@kChainsWithSingletonsFromResponse, [$m]);
            $addinv++;
          }
        }
      }
    }
  }

  # 3b. For computing recall: discard twinless system mentions in response
  my $delsys = 0;

  foreach my $rc (@rChains) {
    my @temprc;
    my $i = 0;

    foreach my $m (@$rc) {
      if (defined($kIndex->{$m})) {
        push(@temprc, $m);
        $i++;
      }
      else {
        $delsys++;
      }
    }

    if ($i > 0) {
      push(@rChainsWithoutMentionsNotInKey, \@temprc);
    }
  }

  # We clean the empty chains
  my @newrc;
  foreach my $rc (@rChains) {
    if (scalar(@$rc) > 0) {
      push(@newrc, $rc);
    }
  }
  @rChains = @newrc;

  return (
    \@kChains, \@kChainsWithSingletonsFromResponse,
    \@rChains, \@rChainsWithoutMentionsNotInKey,
    \@kChainsOrig, \@rChainsOrig
  );
}

sub Eval {
  my ($scorer, $keys, $keysPrecision, $response, $responseRecall,
    $keyChainsOrig, $responseChainsOrig)
    = @_;
  $scorer = lc($scorer);
  my ($nr, $dr, $np, $dp);
  if ($scorer eq 'muc') {
    ($nr, $dr, $np, $dp) =
      MUCScorer($keys, $keysPrecision, $response, $responseRecall);
  }
  elsif ($scorer eq 'bcub') {
    ($nr, $dr, $np, $dp) = BCUBED($keyChainsOrig, $responseChainsOrig);
  }
  elsif ($scorer eq 'ceafm') {
    ($nr, $dr, $np, $dp) = CEAF($keyChainsOrig, $responseChainsOrig, 1);
  }
  elsif ($scorer eq 'ceafe') {
    ($nr, $dr, $np, $dp) = CEAF($keyChainsOrig, $responseChainsOrig, 0);
  }
  else {
    die "Metric $scorer not implemented yet\n";
  }
  return ($nr, $dr, $np, $dp);
}

# Indexes an array of arrays, in order to easily know the position of an element
sub Indexa {
  my ($arrays) = @_;
  my %index;

  for (my $i = 0 ; $i < @$arrays ; $i++) {
    foreach my $e (@{$arrays->[$i]}) {
      $index{$e} = $i;
    }
  }
  return \%index;
}

# Consider the "links" within every coreference chain. For example,
# chain A-B-C-D has 3 links: A-B, B-C and C-D.
# Recall: num correct links / num expected links.
# Precision: num correct links / num output links

sub MUCScorer {
  my ($keys, $keysPrecision, $response, $responseRecall) = @_;

  my $kIndex = Indexa($keys);

  # Calculate correct links
  my $correct = 0;
  foreach my $rEntity (@$response) {
    next if (!defined($rEntity));

    # for each possible pair
    for (my $i = 0 ; $i < @$rEntity ; $i++) {
      my $id_i = $rEntity->[$i];
      for (my $j = $i + 1 ; $j < @$rEntity ; $j++) {
        my $id_j = $rEntity->[$j];
        if ( defined($kIndex->{$id_i})
          && defined($kIndex->{$id_j})
          && $kIndex->{$id_i} == $kIndex->{$id_j})
        {
          $correct++;
          last;
        }
      }
    }
  }

  # Links in key
  my $keylinks = 0;
  foreach my $kEntity (@$keys) {
    next if (!defined($kEntity));
    $keylinks += scalar(@$kEntity) - 1 if (scalar(@$kEntity));
  }

  # Links in response
  my $reslinks = 0;
  foreach my $rEntity (@$response) {
    next if (!defined($rEntity));
    $reslinks += scalar(@$rEntity) - 1 if (scalar(@$rEntity));
  }

  ShowRPF($correct, $keylinks, $correct, $reslinks) if ($VERBOSE);
  return ($correct, $keylinks, $correct, $reslinks);
}

# Compute precision for every mention in the response, and compute
# recall for every mention in the keys
sub BCUBED {
  my ($keys, $response) = @_;
  my $kIndex = Indexa($keys);
  my $rIndex = Indexa($response);
  my $acumP  = 0;
  my $acumR  = 0;
  foreach my $rChain (@$response) {
    foreach my $m (@$rChain) {
      my $kChain = (defined($kIndex->{$m})) ? $keys->[$kIndex->{$m}] : [];
      my $ci     = 0;
      my $ri     = scalar(@$rChain);
      my $ki     = scalar(@$kChain);

      # common mentions in rChain and kChain => Ci
      foreach my $mr (@$rChain) {
        foreach my $mk (@$kChain) {
          if ($mr == $mk) {
            $ci++;
            last;
          }
        }
      }

      $acumP += $ci / $ri if ($ri);
      $acumR += $ci / $ki if ($ki);
    }
  }

  # Mentions in key
  my $keymentions = 0;
  foreach my $kEntity (@$keys) {
    $keymentions += scalar(@$kEntity);
  }

  # Mentions in response
  my $resmentions = 0;
  foreach my $rEntity (@$response) {
    $resmentions += scalar(@$rEntity);
  }

  ShowRPF($acumR, $keymentions, $acumP, $resmentions) if ($VERBOSE);
  return ($acumR, $keymentions, $acumP, $resmentions);
}

# type = 0: Entity-based
# type = 1: Mention-based
sub CEAF {
  my ($keys, $response, $type) = @_;

  my @sim;
  for (my $i = 0 ; $i < scalar(@$keys) ; $i++) {
    for (my $j = 0 ; $j < scalar(@$response) ; $j++) {
      if (defined($keys->[$i]) && defined($response->[$j])) {
        if ($type == 0) {    # entity-based
          $sim[$i][$j] = 1 - SIMEntityBased($keys->[$i], $response->[$j]);

          # 1 - X => the library searches minima not maxima
        }
        elsif ($type == 1) {    # mention-based
          $sim[$i][$j] = 1 - SIMMentionBased($keys->[$i], $response->[$j]);
        }
      }
      else {
        $sim[$i][$j] = 1;
      }
    }

    # fill the matrix when response chains are less than key ones
    for (my $j = scalar(@$response) ; $j < scalar(@$keys) ; $j++) {
      $sim[$i][$j] = 1;
    }

    #$denrec += SIMEntityBased($kChain->[$i], $kChain->[$i]);
  }

  my @out;

  # Munkres algorithm
  assign(\@sim, \@out);

  my $numerador = 0;
  my $denpre    = 0;
  my $denrec    = 0;

  # entity-based
  if ($type == 0) {
    foreach my $c (@$response) {
      $denpre++ if (defined($c) && scalar(@$c) > 0);
    }
    foreach my $c (@$keys) {
      $denrec++ if (defined($c) && scalar(@$c) > 0);
    }
  }

  # mention-based
  elsif ($type == 1) {
    foreach my $c (@$response) {
      $denpre += scalar(@$c) if (defined($c));
    }
    foreach my $c (@$keys) {
      $denrec += scalar(@$c) if (defined($c));
    }
  }

  for (my $i = 0 ; $i < scalar(@$keys) ; $i++) {
    $numerador += 1 - $sim[$i][$out[$i]];
  }

  ShowRPF($numerador, $denrec, $numerador, $denpre) if ($VERBOSE);

  return ($numerador, $denrec, $numerador, $denpre);
}

sub SIMEntityBased {
  my ($a, $b) = @_;
  my $intersection = 0;

  # Common elements in A and B
  foreach my $ma (@$a) {
    next if (!defined($ma));
    foreach my $mb (@$b) {
      next if (!defined($mb));
      if ($ma == $mb) {
        $intersection++;
        last;
      }
    }
  }

  my $r = 0;
  my $d = scalar(@$a) + scalar(@$b);
  if ($d != 0) {
    $r = 2 * $intersection / $d;
  }

  return $r;
}

sub SIMMentionBased {
  my ($a, $b) = @_;
  my $intersection = 0;

  # Common elements in A and B
  foreach my $ma (@$a) {
    next if (!defined($ma));
    foreach my $mb (@$b) {
      next if (!defined($mb));
      if ($ma == $mb) {
        $intersection++;
        last;
      }
    }
  }

  return $intersection;
}

sub ShowRPF {
  my ($numrec, $denrec, $numpre, $denpre, $f1) = @_;

  my $precisio = $denpre ? $numpre / $denpre : 0;
  my $recall   = $denrec ? $numrec / $denrec : 0;
  if (!defined($f1)) {
    $f1 = 0;
    if ($recall + $precisio) {
      $f1 = 2 * $precisio * $recall / ($precisio + $recall);
    }
  }

  print "Recall: ($numrec / $denrec) " . int($recall * 10000) / 100 . '%';
  print "\tPrecision: ($numpre / $denpre) "
    . int($precisio * 10000) / 100 . '%';
  print "\tF1: " . int($f1 * 10000) / 100 . "\%\n";
  print
"--------------------------------------------------------------------------\n";
}

# NEW
sub ScoreBLANC {
  my ($kFile, $rFile, $name) = @_;
  my ($acumNRa, $acumDRa, $acumNPa, $acumDPa) = (0, 0, 0, 0);
  my ($acumNRr, $acumDRr, $acumNPr, $acumDPr) = (0, 0, 0, 0);
  my %idenTotals =
    (recallDen => 0, recallNum => 0, precisionDen => 0, precisionNum => 0);

  if (defined($name) && $name ne 'none') {
    print "$name:\n" if ($VERBOSE);
    my $keys     = GetCoreference($kFile, $KEY_COLUMN,      $name);
    my $response = GetCoreference($rFile, $RESPONSE_COLUMN, $name);
    my (
      $keyChains, $keyChainsWithSingletonsFromResponse,
      $responseChains, $responseChainsWithoutMentionsNotInKey,
      $keyChainsOrig, $responseChainsOrig
    ) = IdentifMentions($keys, $response, \%idenTotals);
    (
      $acumNRa, $acumDRa, $acumNPa, $acumDPa,
      $acumNRr, $acumDRr, $acumNPr, $acumDPr
    ) = BLANC_Internal($keyChainsOrig, $responseChainsOrig);
  }
  else {
    my $kIndexNames = GetFileNames($kFile);
    my $rIndexNames = GetFileNames($rFile);

    $VERBOSE = 0 if ($name eq 'none');
    foreach my $iname (keys(%{$kIndexNames})) {
      my $keys =
        GetCoreference($kFile, $KEY_COLUMN, $iname, $kIndexNames->{$iname});
      my $response = GetCoreference($rFile, $RESPONSE_COLUMN, $iname,
        $rIndexNames->{$iname});

      print "$name:\n" if ($VERBOSE);
      my (
        $keyChains,      $keyChainsWithSingletonsFromResponse,
        $responseChains, $responseChainsWithoutMentionsNotInKey,
        $keyChainsOrig,  $responseChainsOrig
      ) = IdentifMentions($keys, $response, \%idenTotals);
      my ($nra, $dra, $npa, $dpa, $nrr, $drr, $npr, $dpr) =
        BLANC_Internal($keyChainsOrig, $responseChainsOrig);

      $acumNRa += $nra;
      $acumDRa += $dra;
      $acumNPa += $npa;
      $acumDPa += $dpa;
      $acumNRr += $nrr;
      $acumDRr += $drr;
      $acumNPr += $npr;
      $acumDPr += $dpr;
    }
  }

  if ($VERBOSE || $name eq 'none') {
    print "\n====== TOTALS =======\n";
    print "Identification of Mentions: ";
    ShowRPF(
      $idenTotals{recallNum},    $idenTotals{recallDen},
      $idenTotals{precisionNum}, $idenTotals{precisionDen}
    );
    print "\nCoreference:\n";
    print "Coreference links: ";
    ShowRPF($acumNRa, $acumDRa, $acumNPa, $acumDPa);
    print "Non-coreference links: ";
    ShowRPF($acumNRr, $acumDRr, $acumNPr, $acumDPr);
    print "BLANC: ";

    my $Ra = ($acumDRa) ? $acumNRa / $acumDRa : -1;
    my $Rr = ($acumDRr) ? $acumNRr / $acumDRr : -1;
    my $Pa = ($acumDPa) ? $acumNPa / $acumDPa : 0;
    my $Pr = ($acumDPr) ? $acumNPr / $acumDPr : 0;

    my $R = ($Ra + $Rr) / 2;
    my $P = ($Pa + $Pr) / 2;

    my $Fa = ($Pa + $Ra) ? 2 * $Pa * $Ra / ($Pa + $Ra) : 0;
    my $Fr = ($Pr + $Rr) ? 2 * $Pr * $Rr / ($Pr + $Rr) : 0;

    my $f1 = ($Fa + $Fr) / 2;

    if ($Ra == -1 && $Rr == -1) {
      $R  = 0;
      $P  = 0;
      $f1 = 0;
    }
    elsif ($Ra == -1) {
      $R  = $Rr;
      $P  = $Pr;
      $f1 = $Fr;
    }
    elsif ($Rr == -1) {
      $R  = $Ra;
      $P  = $Pa;
      $f1 = $Fa;
    }

    ShowRPF($R, 1, $P, 1, $f1);
  }
  return (
    $acumNRa, $acumDRa, $acumNPa, $acumDPa,
    $acumNRr, $acumDRr, $acumNPr, $acumDPr
  );
}

sub cartesian {
  my @C = map { [$_] } @{shift @_};

  foreach (@_) {
    my @A = @$_;

    @C = map {
      my $n = $_;
      map { [$n, @$_] } @C
    } @A;
  }

  return @C;
}

sub BLANC_Internal {
  my ($keys, $response) = @_;
  my ($ga, $gr, $ba, $br) = (0, 0, 0, 0);
  my $key_coreference_links          = {};
  my $key_non_coreference_links      = {};
  my $response_coreference_links     = {};
  my $response_non_coreference_links = {};

  print "list containing list of chains in key:\n" if ($VERBOSE > 2);
  print Dumper $keys if ($VERBOSE > 2);

  print "each key chain printed individually:\n" if ($VERBOSE > 2);

  if ($VERBOSE > 2) {
    foreach my $z (@$keys) {
      print Dumper $z;
    }
  }

  print "list containing list of chains in response:\n" if ($VERBOSE > 2);
  print Dumper $response if ($VERBOSE > 2);

  print "each response chain printed individually:\n" if ($VERBOSE > 2);

  if ($VERBOSE > 2) {
    foreach my $z (@$response) {
      print Dumper $z;
    }
  }

  print
"---------------------------------------------------------------------------------"
    . "\n"
    if ($VERBOSE > 2);

  print "combinations of links for each chain in the key:\n" if ($VERBOSE > 2);
  for my $kkk (@$keys) {
    my $ccombinat = Math::Combinatorics->new(
      count => 2,
      data  => [@$kkk],
    );

    while (my @zcombo = $ccombinat->next_combination) {
      print Dumper [@zcombo] if ($VERBOSE > 2);
      my @zzcombo = sort { $a <=> $b } @zcombo;

      $key_coreference_links->{$zzcombo[0] . "-" . $zzcombo[1]} = 1;
    }

    print
"................................................................................\n"
      if ($VERBOSE > 2);
  }

  print Dumper $key_coreference_links if ($VERBOSE > 2);
  print
"********************************************************************************\n"
    if ($VERBOSE > 2);

  print
"---------------------------------------------------------------------------------"
    . "\n"
    if ($VERBOSE > 2);
  print "combinations of links for each chain in the response:\n"
    if ($VERBOSE > 2);
  for my $rrr (@$response) {
    my $ccombinat = Math::Combinatorics->new(
      count => 2,
      data  => [@$rrr],
    );

    while (my @zcombo = $ccombinat->next_combination) {
      print Dumper [@zcombo] if ($VERBOSE > 2);
      my @zzcombo = sort { $a <=> $b } @zcombo;

      $response_coreference_links->{$zzcombo[0] . "-" . $zzcombo[1]} = 1;
    }

    print
"................................................................................\n"
      if ($VERBOSE > 2);
  }

  print Dumper $response_coreference_links if ($VERBOSE > 2);
  print
"********************************************************************************\n"
    if ($VERBOSE > 2);

  my $number_chains_in_key = @$keys;
  print "number chains in key: " . $number_chains_in_key . "\n"
    if ($VERBOSE > 2);

  my @s  = (0 .. $number_chains_in_key - 1);
  my $ss = join(' ', @s);
  my @n  = split(' ', $ss);

  my $combinat = Math::Combinatorics->new(
    count => 2,
    data  => [@n],
  );

  print "combinations of 2 from: " . join(" ", @n) . "\n" if ($VERBOSE > 2);
  print "------------------------" . ("--" x scalar(@n)) . "\n"
    if ($VERBOSE > 2);

  while (my @combo = $combinat->next_combination) {

    my @kcombo = ();
    foreach my $comboo (@combo) {
      push(@kcombo, @$keys[$comboo]);
    }

    my $lkcombo = @kcombo;
    print "length: " . $lkcombo . "\n" if ($VERBOSE > 2);
    print "kcombo:\n"                  if ($VERBOSE > 2);
    print "+++++\n"                    if ($VERBOSE > 2);
    print Dumper [@kcombo] if ($VERBOSE > 2);
    my @kccar = cartesian($kcombo[0], $kcombo[1]);

    foreach my $x (@kccar) {
      print "--->>>>>>>>>>>>\n" if ($VERBOSE > 2);
      print Dumper $x if ($VERBOSE > 2);
      my @y = sort { $a <=> $b } @$x;
      print Dumper [@y] if ($VERBOSE > 2);
      $key_non_coreference_links->{@y[0] . "-" . @y[1]} = 1;
    }

    print Dumper $key_non_coreference_links if ($VERBOSE > 2);
    print "" . "\n" if ($VERBOSE > 2);

    print ".....\n" if ($VERBOSE > 2);

    print "\n" if ($VERBOSE > 2);
  }

  print "\n" if ($VERBOSE > 2);
  my $number_chains_in_response = @$response;
  print "number chains in response: " . $number_chains_in_response . "\n"
    if ($VERBOSE > 2);

  my @s  = (0 .. $number_chains_in_response - 1);
  my $ss = join(' ', @s);
  my @n  = split(' ', $ss);

  my $combinat = Math::Combinatorics->new(
    count => 2,
    data  => [@n],
  );

  print "combinations of 2 from: " . join(" ", @n) . "\n" if ($VERBOSE > 2);
  print "------------------------" . ("--" x scalar(@n)) . "\n"
    if ($VERBOSE > 2);

  while (my @combo = $combinat->next_combination) {
    my @kcombo = ();
    foreach my $comboo (@combo) {
      push(@kcombo, @$response[$comboo]);
    }

    my $lkcombo = @kcombo;
    print "length: " . $lkcombo . "\n" if ($VERBOSE > 2);
    print "kcombo:\n"                  if ($VERBOSE > 2);
    print "+++++\n"                    if ($VERBOSE > 2);
    print Dumper [@kcombo] if ($VERBOSE > 2);
    my @kccar = cartesian($kcombo[0], $kcombo[1]);

    foreach my $x (@kccar) {
      print "--->>>>>>>>>>>>\n" if ($VERBOSE > 2);
      print Dumper $x if ($VERBOSE > 2);
      my @y = sort { $a <=> $b } @$x;
      print Dumper [@y] if ($VERBOSE > 2);
      $response_non_coreference_links->{@y[0] . "-" . @y[1]} = 1;
    }

    print Dumper $response_non_coreference_links if ($VERBOSE > 2);
    print "" . "\n" if ($VERBOSE > 2);

    print ".....\n" if ($VERBOSE > 2);
    print "\n"      if ($VERBOSE > 2);
  }

  print "\n" if ($VERBOSE > 2);

  print
"<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<\n"
    if ($VERBOSE > 2);
  print Dumper $key_coreference_links          if ($VERBOSE > 2);
  print Dumper $response_coreference_links     if ($VERBOSE > 2);
  print Dumper $key_non_coreference_links      if ($VERBOSE > 2);
  print Dumper $response_non_coreference_links if ($VERBOSE > 2);
  print
"<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<\n"
    if ($VERBOSE > 2);

  my @union_cl = my @isect_cl = ();
  my %union_cl = my %isect_cl = ();

  my @kcl = keys %$key_coreference_links;
  my @rcl = keys %$response_coreference_links;

  print Dumper @kcl if ($VERBOSE > 2);
  print
"++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++\n"
    if ($VERBOSE > 2);
  print Dumper @rcl if ($VERBOSE > 2);
  print
"++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++\n"
    if ($VERBOSE > 2);

  foreach my $e (@kcl, @rcl) { $union_cl{$e}++ && $isect_cl{$e}++ }

  @union_cl = keys %union_cl;
  @isect_cl = keys %isect_cl;

  print Dumper @isect_cl if ($VERBOSE > 2);
  print
"********************************************************************************\n"
    if ($VERBOSE > 2);

  my @union_ncl = my @isect_ncl = ();
  my %union_ncl = my %isect_ncl = ();

  my @kncl = keys %$key_non_coreference_links;
  my @rncl = keys %$response_non_coreference_links;

  print Dumper @kncl if ($VERBOSE > 2);
  print
"++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++\n"
    if ($VERBOSE > 2);
  print Dumper @rncl if ($VERBOSE > 2);
  print
"++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++\n"
    if ($VERBOSE > 2);

  foreach my $e (@kncl, @rncl) { $union_ncl{$e}++ && $isect_ncl{$e}++ }

  @union_ncl = keys %union_ncl;
  @isect_ncl = keys %isect_ncl;

  print Dumper @isect_ncl if ($VERBOSE > 2);
  print
"++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++\n"
    if ($VERBOSE > 2);

  my $num_isect_cl = @isect_cl;
  print
"    number of links in the intersection of key and response coreference links: "
    . $num_isect_cl . "\n"
    if ($VERBOSE > 2);

  my $num_isect_ncl = @isect_ncl;
  print
"number of links in the intersection of key and response non-coreference links: "
    . $num_isect_ncl . "\n"
    if ($VERBOSE > 2);

  my $num_key_coreference_links = keys %$key_coreference_links;
  print "number of key coreference links: " . $num_key_coreference_links . "\n"
    if ($VERBOSE > 2);

  my $num_response_coreference_links = keys %$response_coreference_links;
  print "number of response coreference links: "
    . $num_response_coreference_links . "\n"
    if ($VERBOSE > 2);

  my $num_key_non_coreference_links = keys %$key_non_coreference_links;
  print "number of key non-coreference links: "
    . $num_key_non_coreference_links . "\n"
    if ($VERBOSE > 2);

  my $num_response_non_coreference_links =
    keys %$response_non_coreference_links;
  print "number of response non-coreference links: "
    . $num_response_non_coreference_links . "\n"
    if ($VERBOSE > 2);

  my ($r_blanc, $p_blanc, $f_blanc) = ComputeBLANCFromCounts(
    $num_isect_cl,                   $num_key_coreference_links,
    $num_response_coreference_links, $num_isect_ncl,
    $num_key_non_coreference_links,  $num_response_non_coreference_links
  );

  print "   blanc recall: " . $r_blanc . "\n" if ($VERBOSE > 2);
  print "blanc precision: " . $p_blanc . "\n" if ($VERBOSE > 2);
  print "  blanc score: " . $f_blanc . "\n"   if ($VERBOSE > 2);
  print
">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>\n"
    if ($VERBOSE > 2);

  return (
    $num_isect_cl,  $num_key_coreference_links,
    $num_isect_cl,  $num_response_coreference_links,
    $num_isect_ncl, $num_key_non_coreference_links,
    $num_isect_ncl, $num_response_non_coreference_links
  );
}

################################################################################
# Compute BLANC recall, precision and F-measure from counts.
# Parameters:
#    (#correct_coref_links, #key_coref_links, #response_coref_links,
#     #correct_noncoref_links, #key_noncoref_links, #response_noncoref_links).
# Returns: (recall, precision, F-measure).
################################################################################
sub ComputeBLANCFromCounts {
  my (
    $num_isect_cl,                   $num_key_coreference_links,
    $num_response_coreference_links, $num_isect_ncl,
    $num_key_non_coreference_links,  $num_response_non_coreference_links
  ) = @_;

  my $kcl_recall =
    ($num_key_coreference_links == 0)
    ? 0
    : ($num_isect_cl / $num_key_coreference_links);
  my $kcl_precision =
    ($num_response_coreference_links == 0)
    ? 0
    : ($num_isect_cl / $num_response_coreference_links);

  print
"<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<\n"
    if ($VERBOSE > 2);
  print "       coreference recall: " . $kcl_recall . "\n"    if ($VERBOSE > 2);
  print "    coreference precision: " . $kcl_precision . "\n" if ($VERBOSE > 2);

  my $fcl =
    ($kcl_recall + $kcl_precision == 0)
    ? 0
    : (2 * $kcl_recall * $kcl_precision / ($kcl_recall + $kcl_precision));
  print "      coreference f-score: " . $fcl . "\n" if ($VERBOSE > 2);

  my $kncl_recall =
    ($num_key_non_coreference_links == 0)
    ? 0
    : ($num_isect_ncl / $num_key_non_coreference_links);
  my $kncl_precision =
    ($num_response_non_coreference_links == 0)
    ? 0
    : ($num_isect_ncl / $num_response_non_coreference_links);

  print
"--------------------------------------------------------------------------------\n"
    if ($VERBOSE > 2);
  print "   non-coreference recall: " . $kncl_recall . "\n" if ($VERBOSE > 2);
  print "non-coreference precision: " . $kncl_precision . "\n"
    if ($VERBOSE > 2);

  my $fncl =
    ($kncl_recall + $kncl_precision == 0)
    ? 0
    : (2 * $kncl_recall * $kncl_precision / ($kncl_recall + $kncl_precision));
  print "  non-coreference f-score: " . $fncl . "\n" if ($VERBOSE > 2);
  print
"--------------------------------------------------------------------------------\n"
    if ($VERBOSE > 2);

  my $r_blanc = -1;
  my $p_blanc = -1;
  my $f_blanc = -1;

  if ($num_key_coreference_links == 0 && $num_key_non_coreference_links == 0) {
    $r_blanc = 0;
    $p_blanc = 0;
    $f_blanc = 0;
  }
  elsif ($num_key_coreference_links == 0 || $num_key_non_coreference_links == 0)
  {
    if ($num_key_coreference_links == 0) {
      $r_blanc = $kncl_recall;
      $p_blanc = $kncl_precision;
      $f_blanc = $fncl;
    }
    elsif ($num_key_non_coreference_links == 0) {
      $r_blanc = $kcl_recall;
      $p_blanc = $kcl_precision;
      $f_blanc = $fcl;
    }
  }
  else {
    $r_blanc = ($kcl_recall + $kncl_recall) / 2;
    $p_blanc = ($kcl_precision + $kncl_precision) / 2;
    $f_blanc = ($fcl + $fncl) / 2;
  }

  return ($r_blanc, $p_blanc, $f_blanc);
}

1;
