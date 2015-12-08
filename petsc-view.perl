#!/usr/bin/perl -w

use PDL;
use PDL::CCS;
use PDL::CCS::IO::PETSc;
use PDL::CCS::IO::MatrixMarket;
use Getopt::Long;

my ($help);
my $mode = 'ccs';
GetOptions('help|h' => \$help,
	   'mm|m' => sub { $mode='mm' },
	   'pdl|print|p|dense|d' => sub { $mode='dense' },
	   'ccs|c' => sub { $mode='ccs' },
	   'info|i!' => sub { $mode='info' },
	   'none|null|noout|n' => sub { $mode='null' },
	  );
if ($help || !@ARGV) {
  print STDERR <<EOF;

Usage: $0 [OPTIONS] PETSC_MATRIX_FILE

Options:
  -help      # this help message
  -ccs       # output CCS PDL text (default)
  -dense     # output dense PDL text
  -info      # just print pdl info
  -mm        # output in MatrixMarket format
  -noout     # suppress pdl output

EOF
  exit 1;
}

##-- open input file
my $infile = shift(@ARGV);
defined(my $ccs = ccs_rpetsc($infile))
  or die("$0: failed to read $infile as PETSc matrix: $!");

if ($mode eq 'ccs') {
  print $ccs;
}
elsif ($mode eq 'dense') {
  print $ccs->todense;
}
elsif ($mode eq 'info') {
  print "INFO: dims=(", join(',', $ccs->dims), "); which: ", $ccs->_whichND->info, "; vals: ", $ccs->_vals->info, "\n";
}
elsif ($mode eq 'mm') {
  $ccs->writemm(\*STDOUT, {start=>0});
}
