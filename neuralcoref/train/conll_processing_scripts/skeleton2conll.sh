#!/bin/bash

function usage {
cat <<EOF



----------------------------------------------------------------------------------------------------
Usage:
-----

${0##*/}  -D <ontonotes-release-data-directory> <top-level-directory>


Description:
-----------

<ontonotes-release-data-directory>: Location of the data directory under the OntoNotes 4.0 release
             <top-level-directory>: The directory inside which the *_skel files exist and need to
                                    be convered to .conll files

----------------------------------------------------------------------------------------------------




EOF
exit;
}


function message
{
  (echo "----------------------------------------------------------------------------------------------------";
   echo "" ; 
   echo $* ;
   echo "" ;
   echo "----------------------------------------------------------------------------------------------------") 1>&2

}

function warning
{
  message "$*"
}

function error
{
  message "$*"
  exit
}


function r { echo ${1%.*}; }
function t { echo ${1##*/}; }
function e { echo $(t ${1##*.}); }
function h { echo ${1%/*}; }



# define helper function: run a command and print its exit code
function erun ()
{
    local debug;
    local verbose;
    debug=0;
    if [[ $1 == "-d" ]]; then
        debug=1;
        shift;
    fi;
    verbose=0;
    if [[ $1 == "-v" ]]; then
        verbose=1;
        shift;
    fi;
    if [[ $DEBUG -eq 1 ]]; then
        debug=1;
    fi;
    if [[ $VERBOSE -eq 1 ]]; then
        verbose=1;
    fi;
    if [[ $debug -eq 1 ]]; then
        echo "eval $1";
    else
        if [[ $verbose -eq 1 ]]; then
            echo "-> $1";
        fi;
        eval $1;
    fi;
    local code=$?;
    if [ $code -ne 0 ]; then
        echo "Exit code: $code";
        break;
    fi
}



# handle the valid command line options
DEBUG=0
TESTING=false
VERBOSE=0
DEBUG_OPTION=""
EDITED=""
while getopts D:dhT opt
do
  case "$opt" in
    v)
      VERBOSE=1;;

    d)
      DEBUG=1
      DEBUG_OPTION="-d";;

    D)
      ON_DATA_DIR="$OPTARG"
      ON_DATA_DIR=${ON_DATA_DIR%/}

      if [[ -z $ON_DATA_DIR ]]; then
        error "please specify a valid ontonotes data directory using the -D option"
        usage
      fi;;

    T)
      # this option is used internally for testing
      TESTING=true;;

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
d=${d%/}


# if the conll release directory is not correct
if [[ $(t $d) != "conll-2012" ]]; then
  error "please make sure that you are pointing to the directory 'conll-2012'"
fi



# if we are testing the release, we do not want to clobber the
# true _conll files
if $TESTING; then
  EXT="_skel2conll"
else
  EXT="_conll"
fi


# if no arguments are specified, then just print usage
if [[ $# -eq 0 ]]; then
    usage
fi




for language in arabic english chinese; do
  # set the EDITED option only for english
  if [[ $language == "english" ]]; then
    EDITED="-edited"
  else
    EDITED=""
  fi

  for partition in train development test; do
    for skel in $(find $d/v?/data/$partition/data/$language/ -name "*_skel"); do
      gold_parse=$ON_DATA_DIR/$(r ${skel/*data\//}).parse

      if [[ ! -e $gold_parse ]]; then
        error "could not find the gold parse [$gold_parse] in the ontonotes distribution ... exiting ..."
        exit
      fi

      conll=${skel/_skel/$EXT}
      erun -v "python $d/v?/scripts/skeleton2conll.py $gold_parse $skel $conll $EDITED --text"
    done
  done
done





# complain if the exit status of the last command executed is non-zero
if [[ $? != 0 ]]; then echo "the last command exited with a non-zero status" 1>&2; fi



