################################################################################
# This is the test configuration file. Test cases are stored in an 
# array, each element consisting of:
#   (1) id: a unique identifier for the test case.
#   (2) key_file: the key file to be tested in the CoNLL format.
#   (3) response_file: the response file to be tested in the CoNLL format.
#   (4) expected_metrics: is a hash label from a metric name (identical to those
#                         used in the scorer.{pl|bat}) to an array of expected
#                         metric values. All metrics have 3 expected numbers:
#                         (recall, precision, F-measure).
################################################################################

package CorefMetricTestConfig;
use strict;
use warnings;
use Exporter;

our @ISA= qw( Exporter );

# these are exported by default.
our @EXPORT = qw(TestCases);

#
# Values following metric names are [recall, precision, F1]
#
our @TestCases = (
{ id => "A1", 
  key_file => "DataFiles/TC-A.key",
  response_file => "DataFiles/TC-A-1.response",
  expected_metrics => { "muc" => [1, 1, 1], 
                        "bcub" => [6/6, 6/6, 1],
                        "ceafm" => [1, 1, 1],
                        "ceafe" => [1, 1, 1],
                        "blanc" => [1, 1, 1] }
},
{ id => "A2", 
  key_file => "DataFiles/TC-A.key", 
  response_file => "DataFiles/TC-A-2.response",
  expected_metrics => { "muc" => [1/3, 1/1, 0.5], 
                        "bcub" => [(7/3)/6, 3/3, 14/25],
                        "ceafm" => [0.5, 1, 0.66667],
                        "ceafe" => [0.6, 0.9, 0.72],
                        "blanc" => [0.21591, 1, 0.35385] }
},
{ id => "A3", 
  key_file => "DataFiles/TC-A.key", 
  response_file => "DataFiles/TC-A-3.response",
  expected_metrics => { "muc" => [3/3, 3/5, 0.75], 
                        "bcub" => [6/6, (4+7/12)/9, 110/163],
                        "ceafm" => [1, 0.66667, 0.8],
                        "ceafe" => [0.88571, 0.66429, 0.75918],
                        "blanc" => [1, 0.42593, 0.59717] }
},
{ id => "A4", 
  key_file => "DataFiles/TC-A.key",
  response_file => "DataFiles/TC-A-4.response",
  expected_metrics => { "muc" => [1/3, 1/3, 1/3], 
                        "bcub" => [(3+1/3)/6, (1+4/3+1/2)/7, 2*(5/9)*(17/42)/((5/9)+(17/42))],
                        "ceafm" => [0.66667, 0.57143, 0.61538],
                        "ceafe" => [0.73333, 0.55, 0.62857],
                        "blanc" => [0.35227, 0.27206, 0.30357] }
},
{ id => "A5", 
  key_file => "DataFiles/TC-A.key", 
  response_file => "DataFiles/TC-A-5.response",
  expected_metrics => { "muc" => [1/3, 1/4, 2/7], 
                        "bcub" => [(3+1/3)/6, 2.5/8, 2*(5/9)*(5/16)/((5/9)+(5/16))],
                        "ceafm" => [0.66667, 0.5, 0.57143],
                        "ceafe" => [0.68889, 0.51667, 0.59048],
                        "blanc" => [0.35227, 0.19048, 0.24716] }
},
{ id => "A6", 
  key_file => "DataFiles/TC-A.key", 
  response_file => "DataFiles/TC-A-6.response",
  expected_metrics => { "muc" => [1/3, 1/4, 2/7],
                        "bcub" => [(10/3)/6, (1+4/3+1/2)/8, 2*(5/9)*(17/48)/((5/9)+(17/48))],
                        "ceafm" => [0.66667, 0.5, 0.57143],
                        "ceafe" => [0.73333, 0.55, 0.62857],
                        "blanc" => [0.35227, 0.20870, 0.25817] }
},
{ id => "A7", 
  key_file => "DataFiles/TC-A.key", 
  response_file => "DataFiles/TC-A-7.response",
  expected_metrics => { "muc" => [1/3, 1/3, 1/3], 
                        "bcub" => [(10/3)/6, (1+4/3+1/2)/7, 2*(5/9)*(17/42)/((5/9)+(17/42))],
                        "ceafm" => [0.66667, 0.57143, 0.61538],
                        "ceafe" => [0.73333, 0.55, 0.62857],
                        "blanc" => [0.35227, 0.27206, 0.30357] }
},
{ id => "A8", 
  key_file => "DataFiles/TC-A.key", 
  response_file => "DataFiles/TC-A-8.response",
  expected_metrics => { "muc" => [1/3, 1/3, 1/3], 
                        "bcub" => [(10/3)/6, (1+4/3+1/2)/7, 2*(5/9)*(17/42)/((5/9)+(17/42))],
                        "ceafm" => [0.66667, 0.57143, 0.61538],
                        "ceafe" => [0.73333, 0.55, 0.62857],
                        "blanc" => [0.35227, 0.27206, 0.30357] }
},
{ id => "A9", 
  key_file => "DataFiles/TC-A.key", 
  response_file => "DataFiles/TC-A-9.response",
  expected_metrics => { "muc" => [1/3, 1/3, 1/3],
                        "bcub" => [(10/3)/6, (1+4/3+1/2)/7, 2*(5/9)*(17/42)/((5/9)+(17/42))],
                        "ceafm" => [0.66667, 0.57143, 0.61538],
                        "ceafe" => [0.73333, 0.55, 0.62857],
                        "blanc" => [0.35227, 0.27206, 0.30357] }
},
{ id => "A10", 
  key_file => "DataFiles/TC-A.key", 
  response_file => "DataFiles/TC-A-10.response",
  expected_metrics => { "muc" => [0, 0, 0], 
                        "bcub" => [3/6, 6/6, 2/3],
                        #”ceafm" => [1, 1, 1],
                        #”ceafe" => [1, 1, 1],
                        "blanc" => [0.5, 0.36667, 0.42308] }
},
{ id => "A11", 
  key_file => "DataFiles/TC-A.key", 
  response_file => "DataFiles/TC-A-11.response",
  expected_metrics => { "muc" => [3/3, 3/5, 6/8], 
                        "bcub" => [6/6, (1/6+2*2/6+3*3/6)/6, 14/25],
                        #”ceafm" => [1, 1, 1],
                        #”ceafe" => [1, 1, 1],
                        "blanc" => [0.5, 0.13333, 0.21053] }
},
{ id => "A12", 
  key_file => "DataFiles/TC-A.key", 
  response_file => "DataFiles/TC-A-12.response",
  expected_metrics => { "muc" => [0, 0, 0], 
                        "bcub" => [(1+1/2+2/3)/6, 4/7, 2*(13/36)*(4/7)/((13/36)+(4/7))],
                        #”ceafm" => [1, 1, 1],
                        #”ceafe" => [1, 1, 1],
                        "blanc" => [0.22727, 0.11905, 0.15625] }
},
{ id => "A13", 
  key_file => "DataFiles/TC-A.key", 
  response_file => "DataFiles/TC-A-13.response",
  expected_metrics => { "muc" => [1/3, 1/6, 2/9], 
                        "bcub" => [(1+1/2+2*2/3)/6, (1/7+1/7+2*2/7)/7, 2*(17/36)*(6/49)/((17/36)+(6/49))],
                        #”ceafm" => [1, 1, 1],
                        #”ceafe" => [1, 1, 1],
                        "blanc" => [0.125, 0.02381, 0.04] }
},
{ id => "B1", 
  key_file => "DataFiles/TC-B.key", 
  response_file => "DataFiles/TC-B-1.response",
  expected_metrics => { #"muc" => [1, 1, 1], 
                        #"bcub" => [1, 1, 1],
                        #”ceafm" => [1, 1, 1],
                        #”ceafe" => [1, 1, 1],
                        "blanc" => [1/2 * (1/4 + 1/3), 1/2 * (1/4 + 1/3), 1/2 * (1/4 + 1/3)] }
},
{ id => "C1", 
  key_file => "DataFiles/TC-C.key", 
  response_file => "DataFiles/TC-C-1.response",
  expected_metrics => { #"muc" => [1, 1, 1], 
                        #"bcub" => [1, 1, 1],
                        #”ceafm" => [1, 1, 1],
                        #”ceafe" => [1, 1, 1],
                        "blanc" => [1/2 * (2/5 + 10/16), 1/2 * (2/5 + 10/16), 1/2 * (2/5 + 10/16)] }
},
{ id => "D1", 
		key_file => "DataFiles/TC-D.key", 
		response_file => "DataFiles/TC-D-1.response",
 		expected_metrics => { "muc" => [9/9, 9/10, 2*(9/9)*(9/10)/(9/9+9/10)], 
                          "bcub" => [12/12, 16/21, 2*(12/12)*(16/21)/(12/12+16/21)],
                          #"ceafm" => [1, 1, 1],
                          #"ceafe" => [1, 1, 1],
                          #"blanc" => [1, 1, 1]
                        }
},
{ id => "E1", 
		key_file => "DataFiles/TC-E.key", 
		response_file => "DataFiles/TC-E-1.response",
 		expected_metrics => { "muc" => [9/9, 9/10, 2*(9/9)*(9/10)/(9/9+9/10)], 
                          "bcub" => [1, 7/12, 2*1*(7/12)/(1+7/12)],
                          #"ceafm" => [1, 1, 1],
                          #"ceafe" => [1, 1, 1],
                          #"blanc" => [1, 1, 1]
                        }
},
{ id => "F1", 
		key_file => "DataFiles/TC-F.key", 
		response_file => "DataFiles/TC-F-1.response",
 		expected_metrics => { "muc" => [2/3, 2/2, 2*(2/3)*(2/2)/(2/3+2/2)] ,
                          #"bcub" => ,
                          #"ceafm" => ,
                          #"ceafe" => ,
                          #"blanc" => 
                        }
},
{ id => "G1", 
		key_file => "DataFiles/TC-G.key", 
		response_file => "DataFiles/TC-G-1.response",
 		expected_metrics => { "muc" => [2/2, 2/3, 2*(2/2)*(2/3)/(2/2+2/3)],
                          #"bcub" => ,
                          #"ceafm" => ,
                          #"ceafe" => ,
                          #"blanc" => 
                        }
},
{ id => "H1", 
		key_file => "DataFiles/TC-H.key", 
		response_file => "DataFiles/TC-H-1.response",
 		expected_metrics => { "muc" => [1, 1, 1],
                          #"bcub" => ,
                          #"ceafm" => ,
                          #"ceafe" => ,
                          #"blanc" => 
                        }
},
{ id => "I1", 
		key_file => "DataFiles/TC-I.key", 
		response_file => "DataFiles/TC-I-1.response",
 		expected_metrics => { "muc" => [2/3, 2/2, 2*(2/3)*(2/2)/(2/3+2/2)],
                          #"bcub" => ,
                          #"ceafm" => ,
                          #"ceafe" => ,
                          #"blanc" => 
                        }
},
{ id => "J1", 
		key_file => "DataFiles/TC-J.key", 
		response_file => "DataFiles/TC-J-1.response",
 		expected_metrics => { "muc" => [1/2, 1/1, 2*(1/2)*(1/1)/(1/2+1/1)],
                          #"bcub" => ,
                          #"ceafm" => ,
                          #"ceafe" => ,
                          #"blanc" => 
                        }
},
{ id => "K1", 
		key_file => "DataFiles/TC-K.key", 
		response_file => "DataFiles/TC-K-1.response",
 		expected_metrics => { "muc" => [3/6, 3/6, 3/6],
                          #"bcub" => ,
                          #"ceafm" => ,
                          #"ceafe" => ,
                          #"blanc" => 
                        }
},
{ id => "L1", 
		key_file => "DataFiles/TC-L.key", 
		response_file => "DataFiles/TC-L-1.response",
 		expected_metrics => { "muc" => [2/5, 2/4, 2*(2/5)*(2/4)/(2/5+2/4)],
                          #"bcub" => ,
                          #"ceafm" => ,
                          #"ceafe" => ,
                          #"blanc" => 
                        }
},
{ id => "M1", 
		key_file => "DataFiles/TC-M.key", 
		response_file => "DataFiles/TC-M-1.response",
 		expected_metrics => { "muc" => [1, 1, 1],
                          "bcub" => [1, 1, 1],
                          "ceafm" => [1, 1, 1],
                          "ceafe" => [1, 1, 1],
                          "blanc" => [1, 1, 1] }
},
{ id => "M2", 
		key_file => "DataFiles/TC-M.key", 
		response_file => "DataFiles/TC-M-2.response",
 		expected_metrics => { "muc" => [0, 0, 0],
                          #"bcub" => ,
                          #"ceafm" => ,
                          #"ceafe" => ,
                          "blanc" => [0, 0, 0] }
},
{ id => "M3", 
		key_file => "DataFiles/TC-M.key", 
		response_file => "DataFiles/TC-M-3.response",
 		expected_metrics => { #"muc" => ,
                          #"bcub" => ,
                          #"ceafm" => ,
                          #"ceafe" => ,
                          "blanc" => [0.26667, 1, 0.42105] }
},
{ id => "M4", 
		key_file => "DataFiles/TC-M.key", 
		response_file => "DataFiles/TC-M-4.response",
 		expected_metrics => { #"muc" => ,
                          #"bcub" => ,
                          #"ceafm" => ,
                          #"ceafe" => ,
                          "blanc" => [0.2, 0.2, 0.2] }
},
{ id => "M5", 
		key_file => "DataFiles/TC-M.key", 
		response_file => "DataFiles/TC-M-5.response",
 		expected_metrics => { "muc" => [0, 0, 0],
                          #"bcub" => ,
                          #"ceafm" => ,
                          #"ceafe" => ,
                          "blanc" => [0, 0, 0] }
},
{ id => "M6", 
		key_file => "DataFiles/TC-M.key", 
		response_file => "DataFiles/TC-M-6.response",
 		expected_metrics => { #"muc" => ,
                          #"bcub" => ,
                          #"ceafm" => ,
                          #"ceafe" => ,
                          "blanc" => [0.06667, 0.25, 0.10526] }
},
{ id => "N1", 
		key_file => "DataFiles/TC-N.key", 
		response_file => "DataFiles/TC-N-1.response",
 		expected_metrics => { "muc" => [0, 0, 0],
                          #"bcub" => [1, 1, 1],
                          #"ceafm" => [1, 1, 1],
                          #"ceafe" => [1, 1, 1],
                          "blanc" => [1, 1, 1] }
},
{ id => "N2", 
		key_file => "DataFiles/TC-N.key", 
		response_file => "DataFiles/TC-N-2.response",
 		expected_metrics => { "muc" => [0, 0, 0],
                          #"bcub" => ,
                          #"ceafm" => ,
                          #"ceafe" => ,
                          "blanc" => [0, 0, 0] }
},
{ id => "N3", 
		key_file => "DataFiles/TC-N.key", 
		response_file => "DataFiles/TC-N-3.response",
 		expected_metrics => { #"muc" => ,
                          #"bcub" => ,
                          #"ceafm" => ,
                          #"ceafe" => ,
                          "blanc" => [0.73333, 1, 0.84615] }
},
{ id => "N4", 
		key_file => "DataFiles/TC-N.key", 
		response_file => "DataFiles/TC-N-4.response",
 		expected_metrics => { "muc" => [0, 0, 0],
                          #"bcub" => ,
                          #"ceafm" => ,
                          #"ceafe" => ,
                          "blanc" => [0.2, 0.2, 0.2] }
},
{ id => "N5", 
		key_file => "DataFiles/TC-N.key", 
		response_file => "DataFiles/TC-N-5.response",
 		expected_metrics => { #"muc" => ,
                          #"bcub" => ,
                          #"ceafm" => ,
                          #"ceafe" => ,
                          "blanc" => [0, 0, 0] }
},
{ id => "N6", 
		key_file => "DataFiles/TC-N.key", 
		response_file => "DataFiles/TC-N-6.response",
 		expected_metrics => { #"muc" => ,
                          #"bcub" => ,
                          #"ceafm" => ,
                          #"ceafe" => ,
                          "blanc" => [0.13333, 0.18182, 0.15385] }
}

);

1;
