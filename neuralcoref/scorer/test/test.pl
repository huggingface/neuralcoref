#!/usr/bin/perl

BEGIN {
    $d = $0;
    $d =~ s/\/[^\/][^\/]*$//g;
    push(@INC, $d);
    push(@INC, $d . "/../lib");
}

use strict;
use CorScorer;
use CorefMetricTest;
use CorefMetricTestConfig;

my $error_tolerance = 1.e-4;
my $script_dir = $0;
$script_dir =~ s/\/[^\/][^\/]*$//g;

foreach my $test_case (@CorefMetricTestConfig::TestCases) {
  my $id = $test_case->{'id'};
  my @key_response_files = ($script_dir . "/" . $test_case->{'key_file'}, 
                            $script_dir . "/" . $test_case->{'response_file'});
  print "\nTesting case ($id): keyFile=", $key_response_files[0], 
        " responseFile=", $key_response_files[1], "\n";
  my $expected_metrics = $test_case->{'expected_metrics'};
  foreach my $metric_name (sort keys %$expected_metrics) {
    my $expected_values = $expected_metrics->{$metric_name};
    *::SAVED_STDOUT = *STDOUT;
    *STDOUT = *::SUPRRES_STDOUT;
    my @actual_counts = &CorScorer::Score($metric_name, @key_response_files);
    # Compute R,P,and F1 from raw counts.
    my @actual_values = CorefMetricTest::ComputeScoreFromCounts(@actual_counts);
    *STDOUT = *::SAVED_STDOUT;
    my $diff = CorefMetricTest::DiffExpectedAndActual($expected_values, \@actual_values);
    printf "  metric: %+10s", $metric_name;
    if ($diff < $error_tolerance) {
      print " => PASS\n";
    } else {
      print " => FAIL\n";
      print "    Expected (recall, prec, F1) = (", join(" ", @$expected_values), ")\n";
      print "    Actual (recall, prec, F1) = (", join(" ", @actual_values), ")\n";
      #exit(1);
    }
  }
}

