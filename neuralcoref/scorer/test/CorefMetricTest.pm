package CorefMetricTest;
use strict;
use warnings;
use Exporter;

our @ISA= qw(Exporter);
our @EXPORT = qw(ComputeScoreFromCounts DiffExpectedAndActual);

################################################################################
# Compute recall, precision and F1.
# 
# Input: (numerator_counts_for_recall, denominator_counts_for_recall,
#         numerator_counts_for_precision, denominator_counts_for_precision)
# Output: (recall, precision, F1)
################################################################################
sub ComputeScoreFromCounts {
  # The first 4 are also coref link counts when using BLANC.
  my ($recall_numerator, $recall_denominator, 
      $precision_numerator, $precision_denominator, @noncoref_counts) = @_;
  # The coref recall, precision, and F1 when using BLANC.
  my ($recall, $precision, $F1) = 
    RPFFromCounts($recall_numerator, $recall_denominator, 
                  $precision_numerator, $precision_denominator);

  # BLANC: @noncoref_counts=
  #   (noncoref_numerator_recall, noncoref_denominator_recall, 
  #    noncoref_numerator_precision, noncoref_denominator_precision) 
  if (scalar(@noncoref_counts) == 4) {
    ($recall, $precision, $F1) = CorScorer::ComputeBLANCFromCounts(
	$recall_numerator, $recall_denominator, $precision_denominator,
	$noncoref_counts[0], $noncoref_counts[1], $noncoref_counts[3]);
  }
  $recall = ($recall < 0) ? 0 : $recall;
  $precision = ($precision < 0) ? 0 : $precision;
  $F1 = ($F1 < 0) ? 0 : $F1;
  return ($recall, $precision, $F1);
}

sub RPFFromCounts
{
  my ($recall_numerator, $recall_denominator, 
      $precision_numerator, $precision_denominator, @nonCorefCounts) = @_;
  my ($recall, $precision, $F1) = (-1, -1, 0);
  if ($recall_denominator > 0) {
    $recall = $recall_numerator / $recall_denominator; 
  }
  if ($precision_denominator > 0) {
    $precision = $precision_numerator / $precision_denominator;
  }

  if (($recall + $precision) > 0) {
    $F1 = 2 * $recall * $precision / ($recall + $precision);
  }
  
  return ($recall, $precision, $F1);
}

# deprecated -- see CorScorer::ComputeBLANCFromCounts().
sub ComputeBLANCRPF
{
  my ($coref_recall, $coref_precision, $coref_F1,
      $noncoref_recall, $noncoref_precision, $noncoref_F1) = @_;

  my ($recall, $precision, $F1);

  if ($coref_recall < 0 && $noncoref_recall < 0) {
    # no key mention.
    $recall = $precision = $F1 = 0;
  } elsif ($coref_recall < 0) {
    # key: all links are non-coref (mentions are all singltons).
    $recall = $noncoref_recall;
    $precision = ($noncoref_precision < 0) ? 0 : $noncoref_precision;
    $F1 = $noncoref_F1;
  } elsif ($noncoref_recall < 0) {
    # key: all links are coref (all mentions are in one entity).
    $recall = $coref_recall;
    $precision = ($coref_precision < 0) ? 0 : $coref_precision;
    $F1 = $coref_F1;
  } else {
    #key contains both coref and non-coref links.
    if ($coref_precision < 0 && $noncoref_precision < 0) {
      # no response.
      $recall = $precision = $F1 = 0;
    } else {
      if ($coref_precision < 0) {
        # response: all links are non-coref, or response mentions are all
        # singletons.
        $coref_precision = 0;
      } elsif ($noncoref_precision < 0) {
        # response: all links are coref, or all mentions are in one entity.
        $noncoref_precision = 0;
      }
      $recall = ($coref_recall + $noncoref_recall)/2;
      $precision  = ($coref_precision + $noncoref_precision)/2;
      $F1 = ($coref_F1 + $noncoref_F1)/2;
    }
  }

  return ($recall, $precision, $F1);
}

##############################################################################
# Compute the sum of the duifference between the expected recall, precision, 
# F1 and the actual one. 
##############################################################################
sub DiffExpectedAndActual {
  my ($expected, $actual) = @_;
  if (scalar(@$expected) != scalar(@$actual)) {
    print STDERR "Expected and actual have diff dimensions: \n";
    print STDERR "   Expected: ", join(" ", @$expected), "\n";
    print STDERR "     Actual: ", join(" ", @$actual), "\n";
    return 1.0e5;
  }
  my $sum = 0.0;
  my $i = 0;
  foreach my $e (@$expected) {
    $sum += abs($e - $actual->[$i]);
    ++$i;
  }
  return $sum;
}

1;

