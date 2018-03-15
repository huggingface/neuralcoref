=head1 NAME

Math::Combinatorics - Perform combinations and permutations on lists

=head1 SYNOPSIS

Available as an object oriented API.

  use Math::Combinatorics;

  my @n = qw(a b c);
  my $combinat = Math::Combinatorics->new(count => 2,
                                          data => [@n],
                                         );

  print "combinations of 2 from: ".join(" ",@n)."\n";
  print "------------------------".("--" x scalar(@n))."\n";
  while(my @combo = $combinat->next_combination){
    print join(' ', @combo)."\n";
  }

  print "\n";

  print "permutations of 3 from: ".join(" ",@n)."\n";
  print "------------------------".("--" x scalar(@n))."\n";
  while(my @permu = $combinat->next_permutation){
    print join(' ', @permu)."\n";
  }

  output:

Or available via exported functions 'permute', 'combine', and 'factorial'.

  use Math::Combinatorics;

  my @n = qw(a b c);
  print "combinations of 2 from: ".join(" ",@n)."\n";
  print "------------------------".("--" x scalar(@n))."\n";
  print join("\n", map { join " ", @$_ } combine(2,@n)),"\n";
  print "\n";
  print "permutations of 3 from: ".join(" ",@n)."\n";
  print "------------------------".("--" x scalar(@n))."\n";
  print join("\n", map { join " ", @$_ } permute(@n)),"\n";


Output:

  combinations of 2 from: a b c
  ------------------------------
  a b
  a c
  b c

  permutations of 3 from: a b c
  ------------------------------
  a b c
  a c b
  b a c
  b c a
  c a b
  c b a

Output from both types of calls is the same, but the object-oriented approach consumes
much less memory for large sets.

=head1 DESCRIPTION

Combinatorics is the branch of mathematics studying the enumeration, combination,
and permutation of sets of elements and the mathematical relations that characterize
their properties.  As a jumping off point, refer to:

 http://mathworld.wolfram.com/Combinatorics.html

This module provides a pure-perl implementation of nCk, nCRk, nPk, nPRk, !n and n!
(combination, multiset, permutation, string, derangement, and factorial, respectively).
Functional and object-oriented usages allow problems such as the following to be solved:

=over

=item combine - nCk

 http://mathworld.wolfram.com/Combination.html

"Fun questions to ask the pizza parlor wait staff: how many possible combinations
of 2 toppings can I get on my pizza?".

=item derange - !n

 http://mathworld.wolfram.com/Derangement.html

"A derangement of n ordered objects, denoted !n, is a permutation in which none of the
objects appear in their "natural" (i.e., ordered) place."

=item permute - nPk

 http://mathworld.wolfram.com/Permutation.html

"Master Mind Game: ways to arrange pieces of different colors in a
certain number of positions, without repetition of a color".

=back

Object-oriented usage additionally allows solving these problems by calling L</new()>
with a B<frequency> vector:

=over

=item string - nPRk

 http://mathworld.wolfram.com/String.html

"Morse signals: diferent signals of 3 positions using the two symbols - and .".

 $o = Math::Combinatorics->new( count=>3 , data=>[qw(. -)] , frequency=>[3,3] );
 while ( my @x = $o->next_multiset ) {
   my $p = Math::Combinatorics->new( data=>\@x , frequency=>[map{1} @x] );
   while ( my @y = $p->next_string ) {
     #do something
   }
 }

=item multiset/multichoose - nCRk

 http://mathworld.wolfram.com/Multiset.html

"ways to extract 3 balls at once of a bag with 3 black and 3 white balls".

 $o = Math::Combinatorics->new( count=>3 , data=>[qw(white black)] , frequency=>[3,3] );
 while ( my @x = $o->next_multiset ) {
   #do something
 }

=back

=head2 EXPORT

the following export tags will bring a single method into the caller's
namespace.  no symbols are exported by default.  see pod documentation below for
method descriptions.

  combine
  derange
  multiset
  permute
  string
  factorial

=head1 AUTHOR

Allen Day <allenday@ucla.edu>, with algorithmic contributions from Christopher Eltschka and
Tye.

Copyright (c) 2004-2005 Allen Day. All rights reserved. This program is free software; you
can redistribute it and/or modify it under the same terms as Perl itself.

=head1 ACKNOWLEDGEMENTS

A sincere thanks to everyone for helping to make this a better module.  After initial
development I've only had time to accept patches and improvements.  Math::Combinatorics
continues to be developed and improved by the community.  Contributors of note include:

For adding new features: Carlos Rica, David Coppit, Carlos Segre, Lyon Lemmens

For bug reports: Ying Yang, Joerg Beyer, Marc Logghe, Yunheng Wang,
Torsten Seemann, Gerrit Haase, Joern Behre, Lyon Lemmens, Federico Lucifredi

=head1 BUGS / TODO

Report them to the author.

 * Need more extensive unit tests.

   * tests for new()'s frequency argment

 * A known bug (more of a missing feature, actually) does not allow parameterization of k
 for nPk in permute().  it is assumed k == n.  L</permute()> for details.  You can work
 around this by making calls to both L</permute()> and L</combine()>

 * Lots of really interesting stuff from Mathworld.Wolfram.com.  MathWorld rocks!  Expect
 to see implementation of more concepts from their site, e.g.:

   http://mathworld.wolfram.com/BellNumber.html
   http://mathworld.wolfram.com/StirlingNumberoftheSecondKind.html
   http://mathworld.wolfram.com/Word.html

 * Other combinatorics stuff
   http://en.wikipedia.org/wiki/Catalan_number
   http://en.wikipedia.org/wiki/Stirling_number

=head1 SEE ALSO

L<Set::Scalar>

L<Set::Bag>

L<String::Combination> (alas misnamed, it actually returns permutations on a string).

 http://perlmonks.thepen.com/29374.html

 http://groups.google.com/groups?selm=38568F79.13680B86%40physik.tu-muenchen.de&output=gplain


=cut

package Math::Combinatorics;

use strict;
use Data::Dumper;
require Exporter;

our @ISA = qw(Exporter);
our @EXPORT = qw( combine derange factorial permute );
our $VERSION = '0.09';

=head1 EXPORTED FUNCTIONS

=head2 combine()

 Usage   : my @combinations = combine($k,@n);
 Function: implements nCk (n choose k), or n!/(k!*(n-k!)).
           returns all unique unorderd combinations of k items from set n.
           items in n are assumed to be character data, and are
           copied into the return data structure (see "Returns" below).
 Example : my @n = qw(a b c);
           my @c = combine(2,@n);
           print join "\n", map { join " ", @$_ } @c;
           # prints:
           # b c
           # a c
           # a b
 Returns : a list of arrays, where each array contains a unique combination
           of k items from n
 Args    : a list of items to be combined
 Notes   : data is internally assumed to be alphanumeric.  this is necessary
           to efficiently generate combinations of large sets.  if you need
           combinations of non-alphanumeric data, or on data
           C<sort {$a cmp $b}> would not be appropriate, use the
           object-oriented API.  See L</new()> and the B<compare> option.

           Identical items are assumed to be non-unique.  That is, calling
           C<combine(1,'a','a') yields two sets: {a}, and {a}.  See
           L</next_multiset() if this is not the desired behavior.

=cut

sub combine {
  my($k,@n) = @_;

  my @result = ();

  my $c = __PACKAGE__->new(data => [@n], count => $k);
  while(my(@combo) = $c->next_combination){
    push @result, [@combo];
  }

  return @result;
}

=head2 derange()

 Usage   : my @deranges = derange(@n);
 Function: implements !n, a derangement of n items in which none of the
           items appear in their originally ordered place.
 Example : my @n = qw(a b c);
           my @d = derange(@n);
           print join "\n", map { join " ", @$_ } @d;
           # prints:
           # a c b
           # b a c
           # b c a
           # c a b
           # c b a
 Returns : a list of arrays, where each array contains a derangement of
           k items from n (where k == n).
 Args    : a list of items to be deranged.
 Note    : k should really be parameterizable.  this will happen
           in a later version of the module.  send me a patch to
           make that version come out sooner.
 Notes   : data is internally assumed to be alphanumeric.  this is necessary
           to efficiently generate combinations of large sets.  if you need
           combinations of non-alphanumeric data, or on data
           C<sort {$a cmp $b}> would not be appropriate, use the
           object-oriented API.  See L</new()>, and the B<compare> option.

=cut

sub derange {
  my(@n) = @_;

  my @result = ();

  my $c = __PACKAGE__->new(data => [@n]);
  while(my(@derange) = $c->next_derangement){
    push @result, [@derange];
  }

  return @result;
}

=head2 next_derangement()

 Usage   : my @derangement = $c->next_derangement();
 Function: get derangements for @data.
 Returns : returns a permutation of items from @data (see L</new()>),
           where none of the items appear in their natural order.  repeated calls
           retrieve all unique derangements of @data elements.  a returned empty
           list signifies all derangements have been iterated.
 Args    : none.

=cut

sub next_derangement {
  my $self = shift;
  my $data = $self->data();

  my $cursor = $self->_permutation_cursor();
  my $values = @$cursor;
  if($self->{pin}){
    $self->{pin} = 0;

    my $i;
    for ($i = 1; $i < $values; $i += 2) {
      $$cursor[$i - 1] = $i;
      $$cursor[$i]     = $i - 1;
    }
    if ($values % 2 != 0) {
      $$cursor[$values - 1] = $values - 3;
      $$cursor[$values - 2] = $values - 1;
    }
    goto RESULT;
  }
  else {
    my $values = @$cursor;
    my $i;
    my @found;  # stores for each element if it has been found previously
    for ($i = 0; $i < $values; $i++) { $found[$i] = 0 }
    my $e;
    my $elemfound = 0;
    for ($i = $values - 1; $i > -1; $i--) {
      $found[$$cursor[$i]] = 1;
      if ($i > $values - 3) { # $values-1 or $values-2
        if ($i == $values - 2) {
          #print "i=$i (values-2)\n";##
          $e = $$cursor[$i + 1];
          if ($e > $$cursor[$i] && $e != $i
              && $$cursor[$i] != $i + 1) {
            $$cursor[$i + 1] = $$cursor[$i];
            $$cursor[$i] = $e;
            #print "!\n";##
            goto RESULT;
          }
        }
        next;
      }
      for ($e = $$cursor[$i] + 1; $e < $values; $e++) {
        if ($found[$e] && $e != $i) {
          $elemfound = 1;
          last;
        }
      }
      last if ($elemfound);
    }
    if ($elemfound) {
      $$cursor[$i] = $e;
      $found[$e] = 0;
      $i++;
      my $j;
      my @elems;
      for ($j = 0; $j < $values; $j++) {
        if ($found[$j]) { push(@elems, $j) }
      }
      for ($j = 0; $j < @elems; $j++) {
        if ($elems[$j] != $i) {
          # if the next is the last and it will be wrong:
          if ($j + 2 == @elems
              && $elems[$j + 1] == $i + 1) {
            # interchange them:
            $$cursor[$i] = $elems[$j + 1];
            $$cursor[$i + 1] = $elems[$j];
            last;
          }
          $$cursor[$i] = $elems[$j];
        }
        elsif ($j + 1 < @elems) {
          # use the next element:
          $$cursor[$i] = $elems[$j + 1];
          $elems[$j + 1] = $elems[$j];
        }
        else { die() }
        $i++;
      }
      goto RESULT;
    }
    return ();
  }
  RESULT:
  # map cursor to data array
  my @result;
  foreach my $c (@$cursor){
    push @result, $${ $data->[$c] };
  }
  return @result;
}

=head2 factorial()

 Usage   : my $f = factorial(4); #returns 24, or 4*3*2*1
 Function: calculates n! (n factorial).
 Returns : undef if n is non-integer or n < 0
 Args    : a positive, non-zero integer
 Note    : this function is used internally by combine() and permute()

=cut

sub factorial {
  my $n = shift;
  return undef unless $n >= 0 and $n == int($n);

  my $f;

  for($f = 1 ; $n > 0 ; $n--){
    $f *= $n
  }

  return $f;
}

=head2 permute()

 Usage   : my @permutations = permute(@n);
 Function: implements nPk (n permute k) (where k == n), or n!/(n-k)!
            returns all unique permutations of k items from set n
           (where n == k, see "Note" below).  items in n are assumed to
           be character data, and are copied into the return data
           structure.
 Example : my @n = qw(a b c);
           my @p = permute(@n);
           print join "\n", map { join " ", @$_ } @p;
           # prints:
           # b a c
           # b c a
           # c b a
           # c a b
           # a c b
           # a b c
 Returns : a list of arrays, where each array contains a permutation of
           k items from n (where k == n).
 Args    : a list of items to be permuted.
 Note    : k should really be parameterizable.  this will happen
           in a later version of the module.  send me a patch to
           make that version come out sooner.
 Notes   : data is internally assumed to be alphanumeric.  this is necessary
           to efficiently generate combinations of large sets.  if you need
           combinations of non-alphanumeric data, or on data
           C<sort {$a cmp $b}> would not be appropriate, use the
           object-oriented API.  See L</new()>, and the B<compare> option.

           Identical items are assumed to be non-unique.  That is, calling
           C<permute('a','a') yields two sets: {a,a}, and {a,a}.  See
           L</next_string() if this is not the desired behavior.

=cut

sub permute {
  my(@n) = @_;

  my @result = ();

  my $c = __PACKAGE__->new(data => [@n]);
  while(my(@permu) = $c->next_permutation){
    push @result, [@permu];
  }

  return @result;
}

=head1 CONSTRUCTOR

=cut

=head2 new()

 Usage   : my $c = Math::Combinatorics->new( count => 2,       #treated as int
                                             data => [1,2,3,4] #arrayref or anonymous array
                                           );
 Function: build a new Math::Combinatorics object.
 Returns : a Math::Combinatorics object
 Args    : count     - required for combinatoric functions/methods.  number of elements to be
                       present in returned set(s).
           data      - required for combinatoric B<AND> permutagenic functions/methods.  this is the
                       set elements are chosen from.  B<NOTE>: this array is modified in place; make
                       a copy of your array if the order matters in the caller's space.
           frequency - optional vector of data frequencies.  must be the same length as the B<data>
                       constructor argument.  These two constructor calls here are equivalent:

                         $a = 'a';
                         $b = 'b';

                         Math::Combinatorics->new( count=>2, data=>[\$a,\$a,\$a,\$a,\$a,\$b,\$b] );
                         Math::Combinatorics->new( count=>2, data=>[\$a,\$b], frequency=>[5,2] );

                       so why use this?  sometimes it's useful to have multiple identical entities in
                       a set (in set theory jargon, this is called a "bag", See L<Set::Bag>).
           compare   - optional subroutine reference used in sorting elements of the set.  examples:

                       #appropriate for character elements
                       compare => sub { $_[0] cmp $_[1] }
                       #appropriate for numeric elements
                       compare => sub { $_[0] <=> $_[1] }
                       #appropriate for object elements, perhaps
                       compare => sub { $_[0]->value <=> $_[1]->value }

                     The default sort mechanism is based on references, and cannot be predicted.
                     Improvements for a more flexible compare() mechanism are most welcome.

=cut

sub new {
  my($class,%arg) = @_;
  my $self = bless {}, $class;

  $self->{compare} = $arg{compare} || sub { $_[0] cmp $_[1] };
  $self->{count}   = $arg{count};

  #convert bag to set
  my $freq            = $arg{frequency};
  if(ref($freq) eq 'ARRAY' and scalar(@$freq) == scalar(@{$arg{data}})){
    $self->{frequency}++;
    my @bag = @{$arg{data}};
    my @set = ();

    #allow '0 but defined' elements (Yunheng Wang)
    foreach my $type ( @bag ) {
      my $f = shift @$freq;
      next if $f < 1;
      for(1..$f){
        #we push on a reference to make sure, for instance, that objects
        #are identical and not copied
        push @set, \$type;
      }
    }
    $arg{data} = \@set;
  }
  elsif(!ref($freq)){
    $arg{data} = [map { \$_ } @{$arg{data}}];
  }

#warn join ' ', @{$arg{data}};

  #OK, this is hokey, but I don't have time to fix it properly right now.
  #We want to allow both user-specified sorting as well as our own
  #reference-based internal sorting -- the latter only because unit tests
  #are failing if we don't have it.  Additionally, we don't want to require
  #the triple derefernce necessary for comparison of the pristine data in
  #the user-supplied compare coderef.  The solution for now is to do an
  #if/else.  If you're staring at this please fix it!
  my $compare = $self->{compare};
  if ( defined $arg{compare} ) {
    $self->{data} = [sort {&$compare($$$a,$$$b)} map {\$_} @{$arg{data}}];
  }
  else {
    $self->{data} = [sort {&$compare($a,$b)} map {\$_} @{$arg{data}}];
  }

#warn Dumper($self->{data});

  $self->{cin} = 1;
  $self->{pin} = 1;

  return $self;
}

=head1 OBJECT METHODS

=cut

=head2 next_combination()

 Usage   : my @combo = $c->next_combination();
 Function: get combinations of size $count from @data.
 Returns : returns a combination of $count items from @data (see L</new()>).
           repeated calls retrieve all unique combinations of $count elements.
           a returned empty list signifies all combinations have been iterated.
 Note    : this method may only be used if a B<frequency> argument is B<NOT>
           given to L</new()>, otherwise use L</next_multiset()>.
 Args    : none.

=cut

sub next_combination {
  my $self = shift;
  if ( $self->{frequency} ) {
    print STDERR "must use next_multiset() if 'frequency' argument passed to constructor\n";
    return ();
  }
  return $self->_next_combination;
}

sub _next_combination {
  my $self = shift;
  my $data = $self->data();
  my $combo_end = $self->count();

  my $begin = 0;
  my $end = $#{$data} + 1;

  my @result;

  return () if scalar(@$data) < $self->count();

  if($self->{cin}){
    $self->{cin} = 0;

    for(0..$self->count-1){
      push @result, $${ $data->[$_] };
    }
#warn 1;
    return @result;
  }

  if ($combo_end == $begin || $combo_end == $end) {
    return ();
  }

  my $combo = $combo_end;
  my $total_set;

  --$combo;
  $total_set = $self->upper_bound($combo_end,$end,$data->[$combo]);
  if ($total_set != $end) {
    $self->swap($combo,$total_set);

    for(0..$self->count-1){
      push @result, $${ $data->[$_] };
    }
#warn 2;
    return @result;
  }

  --$total_set;
  $combo = $self->lower_bound($begin, $combo_end, $data->[$total_set]);

  if ($combo == $begin) {
    $self->rotate($begin, $combo_end, $end);
#warn 3;
    return ();
  }

  my $combo_next = $combo;
  --$combo;
  $total_set = $self->upper_bound($combo_end, $end, $data->[$combo]);

  my $sort_pos = $end;
  $sort_pos += $combo_end - $total_set - 1;

  $self->rotate($combo_next, $total_set, $end);
  $self->rotate($combo, $combo_next, $end);
  $self->rotate($combo_end, $sort_pos, $end);

  for(0..$self->count-1){
    push @result, $${ $data->[$_] };
  }
#warn 4;
  return @result;
}

=head2 next_multiset()

 Usage   : my @multiset = $c->next_multiset();
 Function: get multisets for @data.
 Returns : returns a multiset of items from @data (see L</new()>).
           a multiset is a special type of combination where the set from which
           combinations are drawn contains items that are indistinguishable.  use
           L</next_multiset()> when a B<frequency> argument is passed to L</new()>.
           repeated calls retrieve all unique multisets of @data elements.  a
           returned empty list signifies all multisets have been iterated.
 Note    : this method may only be used if a B<frequency> argument is given to
           L</new()>, otherwise use L</next_combination()>.
 Args    : none.

=cut

sub next_multiset {
  my $self = shift;

  if ( ! $self->{frequency} ) {
    print STDERR "must use next_combination() if 'frequency' argument not passed to constructor\n";
    return ();
  }

  my $data = $self->data();
  my $compare = $self->compare();

  while ( my @combo = $self->_next_combination ) {
    my $x = join '', map {scalar($$_)} sort @$data;
    my $y = join '', map {scalar($_) } sort @combo;

    next if $self->{'cache_multiset'}{$y}++;
    return @combo;
  }
  $self->{'cache_multiset'} = undef;
  return ();
}

=head2 next_permutation()

 Usage   : my @permu = $c->next_permutation();
 Function: get permutations of elements in @data.
 Returns : returns a permutation of items from @data (see L</new()>).
           repeated calls retrieve all unique permutations of @data elements.
           a returned empty list signifies all permutations have been iterated.
 Note    : this method may only be used if a B<frequency> argument is B<NOT>
           given to L</new()>, otherwise use L</next_string()>.
 Args    : none.

=cut

sub next_permutation {
  my $self = shift;
  if ( $self->{frequency} ) {
    print STDERR "must use next_string() if 'frequency' argument passed to constructor\n";
    return ();
  }
  return $self->_next_permutation;
}

sub _next_permutation {
  my $self = shift;
  my $data = $self->data();

  if($self->{pin}){
    $self->{pin} = 0;
    return map {$$$_} @$data;
  }

  my $cursor = $self->_permutation_cursor();

  my $last= $#{$cursor};

  if($last < 1){
    return ();
  }

  # Find last item not in reverse-sorted order:
  my $i = $last - 1;
  $i-- while  0 <= $i  &&  $cursor->[$i] >= $cursor->[$i+1];

  if($i == -1){
    return ();
  }


  # Re-sort the reversely-sorted tail of the list:
  @{$cursor}[$i+1..$last] = reverse @{$cursor}[$i+1..$last]
    if $cursor->[$i+1] > $cursor->[$last];

  # Find next item that will make us "greater":
  my $j = $i+1;
  $j++ while  $cursor->[$i] >= $cursor->[$j];

  # Swap:
  @{$cursor}[$i,$j] = @{$cursor}[$j,$i];

  # map cursor to data array
  my @result;
  foreach my $c (@$cursor){
    push @result, $${ $data->[$c] };
  }
  return @result;
}

=head2 next_string()

 Usage   : my @string = $c->next_string();
 Function: get strings for @data.
 Returns : returns a multiset of items from @data (see L</new()>).
           a multiset is a special type of permutation where the set from which
           combinations are drawn contains items that are indistinguishable.  use
           L</next_permutation()> when a B<frequency> argument is passed to L</new()>.
           repeated calls retrieve all unique multisets of @data elements.  a
           returned empty list signifies all strings have been iterated.
 Note    : this method may only be used if a B<frequency> argument is given to
           L</new()>, otherwise use L</next_permutation()>.
 Args    : none.

=cut

sub next_string {
  my $self = shift;
  my $data = $self->data();

  if ( ! $self->{frequency} ) {
    print STDERR "must use next_permutation() if 'frequency' argument not passed to constructor\n";
    return ();
  }


  while ( my @permu = $self->_next_permutation ) {
    my $x = join '', map {scalar($$_)} @$data;
    my $y = join '', map {scalar($_) } @permu;

    next if $self->{'cache_string'}{$y}++;
    return @permu;
  }

  $self->{'cache_string'} = undef;
  return ();
}

=head1 INTERNAL FUNCTIONS AND METHODS

=head2 sum()

 Usage   : my $sum = sum(1,2,3); # returns 6
 Function: sums a list of integers.  non-integer list elements are ignored
 Returns : sum of integer items in arguments passed in
 Args    : a list of integers
 Note    : this function is used internally by combine()

=cut

sub sum {
  my $sum = 0;
  foreach my $i (@_){
    $sum += $i if $i == int($i);
  }
  return $sum;
}

=head2 compare()

 Usage   : $obj->compare()
 Function: internal, undocumented.  holds a comparison coderef.
 Returns : value of compare (a coderef)


=cut

sub compare {
  my($self,$val) = @_;
  return $self->{'compare'};
}


=head2 count()

 Usage   : $obj->count()
 Function: internal, undocumented.  holds the "k" in nCk or nPk.
 Returns : value of count (an int)

=cut

sub count {
  my($self) = @_;
  return $self->{'count'};
}


=head2 data()

 Usage   : $obj->data()
 Function: internal, undocumented.  holds the set "n" in nCk or nPk.
 Returns : value of data (an arrayref)

=cut

sub data {
  my($self) = @_;
  return $self->{'data'};
}


=head2 swap()

internal, undocumented.

=cut

sub swap {
  my $self = shift;
  my $first = shift;
  my $second = shift;
  my $data = $self->data();

  my $temp = $data->[$first];
  $data->[$first] = $data->[$second];
  $data->[$second] = $temp;
}

=head2 reverse()

internal, undocumented.

=cut

sub reverse {
  my $self = shift;
  my $first = shift;
  my $last = shift;
  my $data = $self->data();

  while (1) {
    if ($first == $last || $first == --$last) {
      return;
    } else {
      $self->swap($first++, $last);
    }
  }
}

=head2 rotate()

internal, undocumented.

=cut

sub rotate {
  my $self = shift;
  my $first = shift;
  my $middle = shift;
  my $last = shift;
  my $data = $self->data();

  if ($first == $middle || $last == $middle) {
    return;
  }

  my $first2 = $middle;

  do {
    $self->swap($first++, $first2++);

    if ($first == $middle) {
      $middle = $first2;
    }
  } while ($first2 != $last);

  $first2 = $middle;

  while ($first2 != $last) {
    $self->swap($first++, $first2++);
    if ($first == $middle) {
      $middle = $first2;
    } elsif ($first2 == $last) {
      $first2 = $middle;
    }
  }
}

=head2 upper_bound()

internal, undocumented.

=cut

sub upper_bound {
  my $self = shift;
  my $first = shift;
  my $last = shift;
  my $value = shift;
  my $compare = $self->compare();
  my $data = $self->data();

  my $len = $last - $first;
  my $half;
  my $middle;

  while ($len > 0) {
    $half = $len >> 1;
    $middle = $first;
    $middle += $half;

    if (&$compare($value,$data->[$middle]) == -1) {
      $len = $half;
    } else {
      $first = $middle;
      ++$first;
      $len = $len - $half - 1;
    }
  }

  return $first;
}

=head2 lower_bound()

internal, undocumented.

=cut

sub lower_bound {
  my $self = shift;
  my $first = shift;
  my $last = shift;
  my $value = shift;
  my $compare = $self->compare();
  my $data = $self->data();

  my $len = $last - $first;
  my $half;
  my $middle;

  while ($len > 0) {
    $half = $len >> 1;
    $middle = $first;
    $middle += $half;

    if (&$compare($data->[$middle],$value) == -1) {
      $first = $middle;
      ++$first;
      $len = $len - $half - 1;
    } else {
      $len = $half;
    }
  }

  return $first;
}

=head2 _permutation_cursor()

 Usage   : $obj->_permutation_cursor()
 Function: internal method.  cursor on permutation iterator order.
 Returns : value of _permutation_cursor (an arrayref)
 Args    : none

=cut

sub _permutation_cursor {
  my($self,$val) = @_;

  if(!$self->{'_permutation_cursor'}){
    my $data = $self->data();
    my @tmp = ();
    my $i = 0;
    push @tmp, $i++ foreach @$data;
    $self->{'_permutation_cursor'} = \@tmp;
  }

  return $self->{'_permutation_cursor'};
}

1;

