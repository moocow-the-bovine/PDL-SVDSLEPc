## File: PDL-SVDSLEPc/Config.PL
## Description: user variables for PDL::SVDSLEPc package

##-- load cache (disabling to see if this fixes cpantesters failures (v0.14, Tue, 17 Nov 2015 14:56:15 +0100)
if (0 && -e "./Config.cache") {
  do "./Config.cache";
  if ($@) {
    warn("$0: could not load cache data from 'Config.cache': $@");
  }
}

##--
## $val = cprompt($key, $message)
## $val = cprompt($key, $message, $default)
##   + prompt for and sets configuration value $cconfig{$key}
##   + uses ExtUtils::MakeMaker::prompt() for cpan smoker-friendliness
sub cprompt {
  my ($key, $msg, $default)=@_;
  return $cconfig{$key} if (defined($cconfig{$key}));
  $default = '' if (!defined($default)); 
  my $answer = ExtUtils::MakeMaker::prompt("<Config.PL> $msg [$default] ?");
  chomp($answer);
  return $cconfig{$key} = ($answer eq '' ? $default : $answer);
}

##--------------------------------------------------------------
## Sources: slepc

## $SLEPC_DIR : root directory of SLEPc installation
my @SLEPC_DIRS = (
		  $cconfig{SLEPC_DIR}, ##-- cached
		  $ENV{SLEPC_DIR},     ##-- SLEPc environment variable
		  glob("slepc-*"),
		  "/usr/local/lib/slepc",
		  "/usr/lib/slepc",
		 );
my $SLEPC_DIR;
foreach my $d (@SLEPC_DIRS) {
  if (-d $d && -r "$d/conf/slepc_common") {
    $SLEPC_DIR = $d;
    last;
  }
}

##-- prompt and check
cprompt("SLEPC_DIR", "SLEPc root directory", $SLEPC_DIR);
if (!$cconfig{SLEPC_DIR} || !-d $cconfig{SLEPC_DIR} || !-r "$cconfig{SLEPC_DIR}/conf/slepc_common") {
  warn("Warning: could not find SLEPc sources in SLEPC_DIR=".($cconfig{SLEPC_DIR}||''));
  #return;
}

##--------------------------------------------------------------
## Sources: petsc

## $PETSC_DIR : root directory of PETSc installation
my @PETSC_DIRS = (
		  $cconfig{PETSC_DIR}, ##-- cached
		  $ENV{PETSC_DIR},     ##-- SLEPc environment variable
		  glob("petsc-*"),
		  "/usr/local/lib/petsc",
		  "/usr/lib/petsc",
		 );
my $PETSC_DIR;
foreach my $d (@PETSC_DIRS) {
  if (-d $d && -r "$d/conf/petscvariables") {
    $PETSC_DIR = $d;
    last;
  }
}

##-- prompt and check
cprompt("PETSC_DIR", "PETSc root directory", $PETSC_DIR);
if (!$cconfig{PETSC_DIR} || !-d $cconfig{PETSC_DIR} || !-r "$cconfig{PETSC_DIR}/conf/petscvariables") {
  warn("Warning: could not find PETSc sources in PETSC_DIR=".($cconfig{PETSC_DIR}||''));
  #return;
}

##--------------------------------------------------------------
## save cache file
use Data::Dumper;
open(CACHE,">Config.cache")
  or die("$0: open failed for Config.cache: $!");
print CACHE
  ("## Automatically generated, remove to re-configure!\n",
   Data::Dumper->Dump([\%cconfig],['*cconfig']),
   "1; ##-- be happy\n",
  );
close CACHE;

1; ##-- return nicely

