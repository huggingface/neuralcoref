package Algorithm::Munkres;

use 5.006;
use strict;
use warnings;

require Exporter;

our @ISA = qw(Exporter);

our @EXPORT = qw( assign );

our $VERSION = '0.08';

#Variables global to the package
my @mat = ();
my @mask = ();
my @colcov = ();
my @rowcov = ();
my $Z0_row = 0;
my $Z0_col = 0;
my @path = ();

#The exported subroutine.
#Expected Input: Reference to the input matrix (MxN)
#Output: Mx1 matrix, giving the column number of the value assigned to each row. (For more explaination refer perldoc)
sub assign
{
    #reference to the input matrix
    my $rmat = shift;
    my $rsolution_mat = shift;
    my ($row, $row_len) = (0,0);

    # re-initialize that global variables
    @mat = ();
    @mask = ();
    @colcov = ();
    @rowcov = ();
    $Z0_row = 0;
    $Z0_col = 0;
    @path = ();

    #variables local to the subroutine
    my $step = 0;
    my ($i, $j) = (0,0);

    #the input matrix
    my @inp_mat = @$rmat;

    #copy the orginal matrix, before applying the algorithm to the matrix
    foreach (@inp_mat)
    {
	push @mat, [ @$_ ];
    }

    #check if the input matrix is well-formed i.e. either square or rectangle.
    $row_len = $#{$mat[0]};
    foreach my $row (@mat)
    {
	if($row_len != $#$row)
	{
	    die "Please check the input matrix.\nThe input matrix is not a well-formed matrix!\nThe input matrix has to be rectangular or square matrix.\n";
	}
    }

    #check if the matrix is a square matrix, 
    #if not convert it to square matrix by padding zeroes.
    if($#mat < $#{$mat[0]})
    {
	# Add rows
	my $diff = $#{$mat[0]} - $#mat;
	for (1 .. $diff)
	{
	    push @mat, [ (0) x @{$mat[0]} ];
	}
    }
    elsif($#mat > $#{$mat[0]})
    {
	# Add columns
	my $diff = $#mat - $#{$mat[0]};
	for (0 .. $#mat)
	{
	    push @{$mat[$_]}, (0) x $diff;
	}
    }

    #initialize mask, column cover and row cover matrices
    clear_covers();

    for($i=0;$i<=$#mat;$i++)
    {
 	push @mask, [ (0) x @mat ];
    }

    #The algorithm can be grouped in 6 steps.
    &stepone();
    &steptwo();
    $step = &stepthree();
    while($step == 4)
    {
	$step = &stepfour();
	while($step == 6)
	{
	    &stepsix();
	    $step = &stepfour();	    
	}
	&stepfive();
	$step = &stepthree();
    }

    #create the output matrix
    for my $i (0 .. $#mat)
    {
	for my $j (0 .. $#{$mat[$i]})
	{
	    if($mask[$i][$j] == 1)
	    {
		$rsolution_mat->[$i] = $j;
	    }
	}
    }


#Code for tracing------------------
    <<'ee';
    print "\nInput Matrix:\n";
    for($i=0;$i<=$#mat;$i++)
    {
	for($j=0;$j<=$#mat;$j++)
	{
	    print $mat[$i][$j] . "\t";
	}
	print "\n";
    }
    
    print "\nMask Matrix:\n";
    for($i=0;$i<=$#mat;$i++)
    {
	for($j=0;$j<=$#mat;$j++)
	{
	    print $mask[$i][$j] . "\t";
	}
	print "\n";
    }

    print "\nOutput Matrix:\n";
    print "$_\n" for @$rsolution_mat;
ee

#----------------------------------

}

#Step 1 - Find minimum value for every row and subtract this min from each element of the row.
sub stepone
{
#    print "Step 1 \n";

    #Find the minimum value for every row
    for my $row (@mat)
    {
	my $min = $row->[0];
	for (@$row)
	{
	    $min = $_ if $min > $_;
	}    
	
        #Subtract the minimum value of the row from each element of the row.
	@$row = map {$_ - $min} @$row;
    }
#    print "Step 1 end \n";
}

#Step 2 - Star the zeroes, Create the mask and cover matrices. Re-initialize the cover matrices for next steps.
#To star a zero: We search for a zero in the matrix and than cover the column and row in which it occurs. Now this zero is starred.
#A next starred zero can occur only in those columns and rows which have not been previously covered by any other starred zero.
sub steptwo
{
#    print "Step 2 \n";
 
    my ($i, $j) = (0,0);

    for($i=0;$i<=$#mat;$i++)
    {
	for($j=0;$j<=$#{$mat[$i]};$j++)
	{
	    if($mat[$i][$j] == 0 && $colcov[$j] == 0 && $rowcov[$i] == 0)
	    {
		$mask[$i][$j] = 1;
		$colcov[$j] = 1;
		$rowcov[$i] = 1;
	    }
	}
    }
    #Re-initialize the cover matrices
    &clear_covers();
#    print "Step 2 end\n";
}

#Step 3 - Check if each column has a starred zero. If yes then the problem is solved else proceed to step 4
sub stepthree
{
#    print "Step 3 \n";

    my $cnt = 0;

    for my $i (0 .. $#mat)
    {
	for my $j (0 .. $#mat)
	{
	    if($mask[$i][$j] == 1)
	    {
		$colcov[$j] = 1;
		$cnt++;
	    }
	}
    }
    if($cnt > $#mat)
    {
#       print "Step 3 end. Next expected step 7 \n";
       return 7;
    }
    else
    {
#       print "Step 3 end. Next expected step 4 \n";
       return 4;
    }

}

#Step 4 - Try to find a zero which is not starred and whose columns and rows are not yet covered. 
#If such a zero found, prime it, try to find a starred zero in its row, 
#                                                 if not found proceed to step 5 
#                                                 else continue
#Else proceed to step 6.
sub stepfour
{
#    print "Step 4 \n";

    while(1)
    {
	my ($row, $col) = &find_a_zero();
	if ($row < 0)
	{
	    # No zeroes
	    return 6;
	}

	$mask[$row][$col] = 2;
	my $star_col = &find_star_in_row($row);
	if ($star_col >= 0)
	{
	    $col = $star_col;
	    $rowcov[$row] = 1;
	    $colcov[$col] = 0;
	}
	else
	{
	    $Z0_row = $row;
	    $Z0_col = $col;
	    return 5;
	}
    }
}

#Tries to find yet uncovered zero
sub find_a_zero
{
    for my $i (0 .. $#mat)
    {
	next if $rowcov[$i];

	for my $j (reverse(0 .. $#mat))  # Prefer large $j
	{
	    next if $colcov[$j];
	    return ($i, $j) if $mat[$i][$j] == 0;
	}
    }

    return (-1, -1);
}

#Tries to find starred zero in the given row and returns the column number
sub find_star_in_row
{
    my $row = shift;

    for my $j (0 .. $#mat)
    {
	if($mask[$row][$j] == 1)
	{
	    return $j;
	}
    }
    return -1;
}

#Step 5 - Try to find a starred zero in the column of the uncovered zero found in the step 4.
#If starred zero found, try to find a prime zero in its row.
#Continue finding starred zero in the column and primed zero in the row until, 
#we get to a primed zero which does not have a starred zero in its column.
#At this point reduce the non-zero values of mask matrix by 1. i.e. change prime zeros to starred zeroes.
#Clear the cover matrices and clear any primes i.e. values=2 from mask matrix.
sub stepfive
{
#    print "Step 5 \n";

    my $cnt = 0;
    my $done = 0;

    $path[$cnt][0] = $Z0_row;
    $path[$cnt][1] = $Z0_col;
    
    while($done == 0)
    {
	my $row = &find_star_in_col($path[$cnt][1]);
	if($row > -1)
	{
	    $cnt++;
	    $path[$cnt][0] = $row;
	    $path[$cnt][1] = $path[$cnt - 1][1];
	}
	else
	{
	    $done = 1;
	}
	if($done == 0)
	{
	    my $col = &find_prime_in_row($path[$cnt][0]);
	    $cnt++;
	    $path[$cnt][0] = $path[$cnt - 1][0];
	    $path[$cnt][1] = $col;
	}
    }
    &convert_path($cnt);
    &clear_covers();
    &erase_primes();

#    print "Step 5 end \n";
}

#Tries to find starred zero in the given column and returns the row number
sub find_star_in_col
{
    my $col = shift;

    for my $i (0 .. $#mat)
    {
	return $i if $mask[$i][$col] == 1;
    }
    
    return -1;
}

#Tries to find primed zero in the given row and returns the column number
sub find_prime_in_row
{
    my $row = shift;

    for my $j (0 .. $#mat)
    {
	return $j if $mask[$row][$j] == 2;
    }
    
    return -1;
}

#Reduces non-zero value in the mask matrix by 1.
#i.e. converts all primes to stars and stars to none.
sub convert_path
{
    my $cnt = shift;

    for my $i (0 .. $cnt)
    {
	for ( $mask[$path[$i][0]][$path[$i][1]] ) {
	    $_ = ( $_ == 1 ) ? 0 : 1;
	}
    }
}

#Clears cover matrices
sub clear_covers
{
    @rowcov = @colcov = (0) x @mat;
}

#Changes all primes i.e. values=2 to 0.
sub erase_primes
{
    for my $row (@mask)
    {
	for my $j (0 .. $#$row)
	{
	    $row->[$j] = 0 if $row->[$j] == 2;
	}
    }
}

#Step 6 - Find the minimum value from the rows and columns which are currently not covered.
#Subtract this minimum value from all the elements of the columns which are not covered.
#Add this minimum value to all the elements of the rows which are covered.
#Proceed to step 4.
sub stepsix
{
#    print "Step 6 \n";
    my ($i, $j);
    my $minval = 0;

    $minval = &find_smallest();
    
    for($i=0;$i<=$#mat;$i++)
    {
	for($j=0;$j<=$#{$mat[$i]};$j++)
	{
	    if($rowcov[$i] == 1)
	    {
		$mat[$i][$j] += $minval;
	    }
	    if($colcov[$j] == 0)
	    {
		$mat[$i][$j] -= $minval;
	    }
	}
    }

#    print "Step 6 end \n";
}

#Finds the minimum value from all the matrix values which are not covered.
sub find_smallest
{
    my $minval;

    for my $i (0 .. $#mat)
    {
	next if $rowcov[$i];

	for my $j (0 .. $#mat)
	{
	    next if $colcov[$j];
	    if( !defined($minval) || $minval > $mat[$i][$j])
	    {
		$minval = $mat[$i][$j];
	    }
	}
    }
    return $minval;
}


1;
__END__

=head1 NAME

    Algorithm::Munkres - Perl extension for Munkres' solution to 
    classical Assignment problem for square and rectangular matrices 
    This module extends the solution of Assignment problem for square
    matrices to rectangular matrices by padding zeros. Thus a rectangular 
    matrix is converted to square matrix by padding necessary zeros.

=head1 SYNOPSIS

use Algorithm::Munkres;

    @mat = (
	 [2, 4, 7, 9],
	 [3, 9, 5, 1],
	 [8, 2, 9, 7],
	 );

assign(\@mat,\@out_mat);

    Then the @out_mat array will have the output as: (0,3,1,2),
    where 
    0th element indicates that 0th row is assigned 0th column i.e value=2
    1st element indicates that 1st row is assigned 3rd column i.e.value=1
    2nd element indicates that 2nd row is assigned 1st column.i.e.value=2
    3rd element indicates that 3rd row is assigned 2nd column.i.e.value=0


=head1 DESCRIPTION

    Assignment Problem: Given N jobs, N workers and the time taken by 
    each worker to complete a job then how should the assignment of a 
    Worker to a Job be done, so as to minimize the time taken. 

	Thus if we have 3 jobs p,q,r and 3 workers x,y,z such that:
	    x  y  z		
	 p  2  4  7
	 q  3  9  5
	 r  8  2  9
        
        where the cell values of the above matrix give the time required
        for the worker(given by column name) to complete the job(given by 
        the row name) 
    
	then possible solutions are:	
		 	 Total
	 1. 2, 9, 9       20
	 2. 2, 2, 5        9
	 3. 3, 4, 9       16
	 4. 3, 2, 7       12
	 5. 8, 9, 7       24
	 6. 8, 4, 5       17

    Thus (2) is the optimal solution for the above problem.
    This kind of brute-force approach of solving Assignment problem 
    quickly becomes slow and bulky as N grows, because the number of 
    possible solution are N! and thus the task is to evaluate each 
    and then find the optimal solution.(If N=10, number of possible
    solutions: 3628800 !)
    Munkres' gives us a solution to this problem, which is implemented 
    in this module.

    This module also solves Assignment problem for rectangular matrices 
    (M x N) by converting them to square matrices by padding zeros. ex:
    If input matrix is:
	 [2, 4, 7, 9],
	 [3, 9, 5, 1],
	 [8, 2, 9, 7]
    i.e 3 x 4 then we will convert it to 4 x 4 and the modified input 
    matrix will be:
	 [2, 4, 7, 9],
	 [3, 9, 5, 1],
	 [8, 2, 9, 7],
 	 [0, 0, 0, 0]

=head1 EXPORT

    "assign" function by default.

=head1 INPUT

    The input matrix should be in a two dimensional array(array of 
    array) and the 'assign' subroutine expects a reference to this 
    array and not the complete array. 
    eg:assign(\@inp_mat, \@out_mat);
    The second argument to the assign subroutine is the reference 
    to the output array.

=head1 OUTPUT

    The assign subroutine expects references to two arrays as its 
    input paramenters. The second parameter is the reference to the
    output array. This array is populated by assign subroutine. This 
    array is single dimensional Nx1 matrix.
    For above example the output array returned will be:
     (0,
     2,
     1)

    where 
    0th element indicates that 0th row is assigned 0th column i.e value=2
    1st element indicates that 1st row is assigned 2nd column i.e.value=5
    2nd element indicates that 2nd row is assigned 1st column.i.e.value=2

=head1 SEE ALSO

    1. http://216.249.163.93/bob.pilgrim/445/munkres.html

    2. Munkres, J. Algorithms for the assignment and transportation 
       Problems. J. Siam 5 (Mar. 1957), 32-38

    3. François Bourgeois and Jean-Claude Lassalle. 1971.
       An extension of the Munkres algorithm for the assignment 
       problem to rectangular matrices.
       Communication ACM, 14(12):802-804

=head1 AUTHOR

    Anagha Kulkarni, University of Minnesota Duluth
    kulka020 <at> d.umn.edu
	
    Ted Pedersen, University of Minnesota Duluth
    tpederse <at> d.umn.edu

=head1 COPYRIGHT AND LICENSE

Copyright (C) 2007-2008, Ted Pedersen and Anagha Kulkarni

This program is free software; you can redistribute it and/or
modify it under the terms of the GNU General Public License
as published by the Free Software Foundation; either version 2
of the License, or (at your option) any later version.
This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program; if not, write to the Free Software
Foundation, Inc., 59 Temple Place - Suite 330, Boston, MA  02111-1307, USA.

=cut
