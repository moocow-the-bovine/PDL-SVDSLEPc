use PDL::Core::Dev;
use ExtUtils::MakeMaker;

##-- read in user variables
do "Config.PL";
die "$0: reading 'Config.PL' failed: $@" if ($@);

##-- pdlmaker stuff
PDL::Core::Dev->import();
require "pdlmaker.plm";

$package = ["svdslepc.pd",SVDSLEPc,PDL::SVDSLEPc];
%hash = pdlmaker_init($package);

##----------------------------------------------------------------------
## ExtUtils::MakeMaker hash
$hash{AUTHOR} = 'Bryan Jurish';
$hash{ABSTRACT} = 'PDL interface to SLEPc sparse singular value decomposition';
$hash{PREREQ_PM} = {
		    PDL=>0,
		   };
$hash{CONFIGURE_REQUIRES} = {
			     'PDL'=>0,
			     'ExtUtils::MakeMaker'=>0,
			     'Data::Dumper'=>0,
			    };
$hash{BUILD_REQUIRES}{'Test::More'} = 0;

$hash{DIR} = [];
$hash{VERSION_FROM} = 'svdslepc.pd';
$hash{LICENSE} = 'perl';
$hash{realclean}{FILES} .= '*~ *.tmp README.txt Config.cache';

#push(@{$hash{LIBS}},    ' '.$cconfig{SLEPC_LIBS});
#$hash{INC}           .= ' '.$cconfig{SLEPC_INC};
#$hash{LDFLAGS}       .= ' '.$cconfig{SLEPC_LDFLAGS};
#$hash{OPTIMIZE}      .= ' '.$cconfig{SLEPC_CFLAGS};

##-- include $Config{ccflags} explicitly to avoid errors like:
## t/00_basic.t .. Not a CODE reference at /usr/lib/perl/5.14/DynaLoader.pm line 207.
use Config;
$hash{CCFLAGS}       .= " $Config{ccflags} ".$cconfig{SLEPC_DEFS};

WriteMakefile(%hash);

##----------------------------------------------------------------------
sub MY::cflags {
  package MY;
  return shift->SUPER::cflags(@_) . <<EOF;

SLEPC_DIR = $::cconfig{SLEPC_DIR}
PETSC_DIR = $::cconfig{PETSC_DIR}

include \$(SLEPC_DIR)/conf/slepc_variables

INC += \$(CCPPFLAGS)
LDFLAGS += \$(SLEPC_LIB)
LDDLFLAGS += \$(SLEPC_LIB)
#OPTIMIZE += 
#CCFLAGS +=

EOF
}

##----------------------------------------------------------------------
sub MY::depend {
  package MY;
  my $inherited = shift->SUPER::depend(@_);
  $inherited .= (
		 "\n\n"
		 ##--
		 #."SVDLIBC.o: svdutil.h svdlib.h\n\n"
		);
}

##----------------------------------------------------------------------
sub MY::special_targets {
  package MY;
  my $inherited = shift->SUPER::special_targets(@_);
  #$inherited =~ s/^(\.SUFFIXES\s*:.*)$/$1 .pod .rpod .pm .man .txt .html;/
  $inherited .= <<EOF

##-- readme
readme: README.txt

##-- debug
slepc-debug:
	echo "SLEPC_LIB = \$(SLEPC_LIB)"
	echo "LDFLAGS = \$(LDFLAGS)"


EOF
}

##----------------------------------------------------------------------
## avoid applying 'processPL' rules to 'Config.PL'
sub MY::processPL { return ''; }
