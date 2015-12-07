#!/usr/bin/perl -w

use lib qw(./blib/lib ./blib/arch);
use PDL;
use PDL::SVDSLEPc;
use PDL::CCS;
use PDL::MatrixOps;

use Benchmark qw(timethese cmpthese);

use vars qw($eps $test_a);
BEGIN{
  $, = ' ';
  $eps=1e-6;
  #$test_a = 1; ##-- test convenience wrappers?
}

##---------------------------------------------------------------------
## test: data
use vars qw($p $whichi $whichv $ptr $colids $nzvals $a $m $n $d);
sub tdata {
  $p = $a = pdl(double, [
			 [10,0,0,0,-2,0,0],
			 [3,9,0,0,0,3,1],
			 [0,7,8,7,0,0,0],
			 [3,0,8,7,5,0,1],
			 [0,8,0,9,9,13,0],
			 [0,4,0,0,2,-1,1],
			]);
  udata();
}

##-- gendata($n,$m,$density) : generate random data
sub gendata {
  ($n,$m,$density,$mult)=@_;
  $density = .5 if (!defined($density));
  $mult    = 1  if (!defined($mult));
  $p = random(double,$n,$m);
  $p->where($p>$density) .= 0;
  $p *= $mult;
  udata();
}

##-- gendata($n,$m,$density) : generate random data
sub gendatag {
  ($n,$m,$density,$mult)=@_;
  $density = .5 if (!defined($density));
  $mult    = 1  if (!defined($mult));
  $p = grandom(double,$n,$m);
  $p->where($p>$density) .= 0;
  $p *= $mult;
  udata();
}

##-- update cccs on changed $p
sub udata {
  ($n,$m) = $p->dims;
  $d      = pdl(long,[$p->dims])->min;
  #$d = $n;
  ##
  ##-- ccs encode
  ($ptr,$colids,$nzvals) = ccsencode($p);
  #$ptr = $ptr->convert(longlong);
  #$colids = $colids->convert(longlong);
  ##
  ##-- coo-encode
  $whichi = $p->whichND->vv_qsortvec;
  $whichv = $p->indexND($whichi);
  ##
  ##-- HACK: allocate an extra slot in $ptr
  $ptr->reshape($ptr->nelem+1);
  $ptr->set(-1, $colids->nelem);
}

##-- common subs
sub svdreduce {
  my ($u,$s,$v, $d) = @_;
  $d = $s->dim(0) if (!defined($d) || $d > $s->dim(0));
  my $end = $d-1;
  return ($u->slice("0:$end,:"),$s->slice("0:$end"),$v->slice("0:$end,"));
}
sub svdcompose {
  my ($u,$s,$v) = @_;
  #return $u x stretcher($s) x $v->xchg(0,1);   ##-- by definition
  return ($u * $s)->matmult($v->xchg(0,1));     ##-- pdl-ized, more efficient
}
sub svdcomposet {
  my ($ut,$s,$vt) = @_;
  return svdcompose($ut->xchg(0,1),$s,$vt->xchg(0,1));
}
sub svdwanterr {
  my ($a,$u,$s,$v) = @_;
  return (($a-svdcompose($u,$s,$v))**2)->flat->sumover;
}


##---------------------------------------------------------------------
## test: constants
sub test_const {
  local $,= '';
  print "slepc_version = ", PDL::SVDSLEPc::slepc_version(), "\n";
  print "petsc_version = ", PDL::SVDSLEPc::petsc_version(), "\n";
  print "library_version [list] = ", PDL::SVDSLEPc::library_version(), "\n";
  print "library_version [sclr] = ", scalar(PDL::SVDSLEPc::library_version()), "\n";
  print "MPI_Comm_size = ", PDL::SVDSLEPc::MPI_Comm_size(), "\n"; ##-- crashes
}
#test_const; exit 0;

##---------------------------------------------------------------------
## test: svd help
sub test_help {
  local $,= '';
  print "slepc_svd_help():\n"; slepc_svd_help();

  ##-- redirect to stderr (ugly, but works)
  if (0) {
    open(my $oldout, '>&STDOUT');
    open(STDOUT, ">&STDERR");
    slepc_svd_help();
    *STDOUT = *$oldout;
  }
}
#test_help; exit 0;

##---------------------------------------------------------------------
## test: option passing
sub test_opts {
  my $u = zeroes($d,$m);
  my $s = zeroes($d);
  my $v = zeroes($d,$n);
  my @opts = qw(-svd_nsv 4 -help);
  _slepc_svd_crs($ptr,$colids,$nzvals, $u,$s,$v, \@opts);
}
#test_opts; exit 0;

##---------------------------------------------------------------------
## test: svd computation
sub test_svd {
  tdata();
  my $u = zeroes($d,$m);
  my $s = zeroes($d);
  my $v = zeroes($d,$n);
  my @opts = (@_);
  _slepc_svd_crs($ptr,$colids,$nzvals, $u,$s,$v, \@opts);
  print STDERR "sigma = ", $s, "\n";

  my ($u0,$s0,$v0) = svd($a);
}
test_svd(@ARGV); exit 0;


##---------------------------------------------------------------------
## DUMMY
##---------------------------------------------------------------------
foreach $i (0..3) {
  print "--dummy($i)--\n";
}

