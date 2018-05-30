#!/bin/bash

function usage {
cat <<EOF



----------------------------------------------------------------------------------------------------
Usage:
-----

${0##*/}  <conll-file>


Description:
-----------

Takes a *conll file as input and prints out the corresponding coreference file

----------------------------------------------------------------------------------------------------




EOF
exit;
}


function message
{
  echo "----------------------------------------------------------------------------------------------------"
  echo
  echo $* 1>&2
  echo
  echo "----------------------------------------------------------------------------------------------------"

}



function r { echo ${1%.*}; }
function t { echo ${1##*/}; }
function e { echo $(t ${1##*.}); }
function h { echo ${1%/*}; }

# define helper function: run a command and print its exit code
function erun () {
  debug=0
  if [[ $1 == "-d" ]]; then
    debug=1
    shift;
  fi


  if [[ $DEBUG -eq 1 ]]; then
      debug=1
  fi




  verbose=0
  if [[ $1 == "-v" ]]; then
    verbose=1
    shift;
  fi


  if [[ $VERBOSE -eq 1 ]]; then
      verbose=1
  fi






  if [[ $debug -eq 1 ]]; then
    echo "debug mode ..."
    echo "eval $1"
  else
    echo "normal mode ..."
    if [[ $verbose -eq 1 ]]; then
      echo -e "\nrun: $1\n-------------"
    fi

    eval $1
  fi


  local code=$?
  if [ $code -ne 0 ]; then
	  echo "Exit code: $code"
	  exit $code
  fi
}




# handle the valid command line options
DEBUG=0
VERBOSE=0
DEBUG_OPTION=""
while getopts vdh opt
do
  case "$opt" in
    v)
      VERBOSE=1;;

    d)
      DEBUG=1;;

    \?)
      usage
      exit 1;;

    h)
      usage
      exit 0;;

    :)
      echo "option -$OPTARG requires an argument"
      usage
      exit 1;;

    esac
done
shift `expr $OPTIND - 1`


# at this point $* contains the arguments after interpreting the options

d=$1

# if no arguments are specified, then just print usage
if [[ $# -eq 0 ]]; then
    usage
fi


# debugging
if [[ $DEBUG -eq 1 ]]; then
    echo "debugging mode is on ..." 1>&2
    DEBUG_OPTION="-d"
fi





for file in $(find $d -name "*_conll"); do

  if [[ $file =~ "data/english/annotations" ]]; then
    LANGUAGE=english
  elif [[ $file =~ "data/chinese/annotations" ]]; then
    LANGUAGE=chinese
  else
    LANGUAGE=arabic
  fi

  echo "language: $LANGUAGE"
  coref=${file/_conll/_coref}
  echo "$file -> $coref ..."
  erun -v "python conll2coreference.py -l $LANGUAGE $file > $coref"
#  conll2coreference.py -l $LANGUAGE $file > $coref
done








# complain if the exit status of the last command executed is non-zero
if [[ $? != 0 ]]; then echo "the last command exited with a non-zero status" 1>&2; fi



