@rem = '--*-Perl-*--
@echo off
if "%OS%" == "Windows_NT" goto WinNT
perl -x -S "%0" %1 %2 %3 %4 %5 %6 %7 %8 %9
goto endofperl
:WinNT
perl -x -S %0 %*
if NOT "%COMSPEC%" == "%SystemRoot%\system32\cmd.exe" goto endofperl
if %errorlevel% == 9009 echo You do not have Perl in your PATH.
if errorlevel 1 goto script_failed_so_exit_with_non_zero_val 2>nul
goto endofperl
@rem ';
#!perl
#line 15

BEGIN {
    $d = $0;
    $d =~ s/\/[^\/][^\/]*$//g;
    push(@INC, $d."/lib");
}

use strict;
use CorScorer;

if (@ARGV < 3) {
  print q|
  use: scorer.bat <metric> <keys_file> <response_file> [name]
  
  metric: the metric desired to score the results:
     muc: MUCScorer (Vilain et al, 1995)
     bcub: B-Cubed (Bagga and Baldwin, 1998)
     ceafm: CEAF (Luo et al, 2005) using mention-based similarity
     ceafe: CEAF (Luo et al, 2005) using entity-based similarity
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

my $metric = shift (@ARGV);
if ($metric !~ /^(muc|bcub|ceafm|ceafe|all)/i) {
  print "Invalid metric\n";
  exit;
}


if ($metric eq 'all') {
  foreach my $m ('muc', 'bcub', 'ceafm', 'ceafe') {
    print "\nMETRIC $m:\n";
    &CorScorer::Score( $m, @ARGV );
  }
}
else {
  &CorScorer::Score( $metric, @ARGV );
}

__END__
:endofperl
