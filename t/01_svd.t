# -*- Mode: CPerl -*-
# t/01_svd.t: test SLEPc svd
use Test::More tests=>16, todo=>[];

$TEST_DIR = './t';
#use lib qw(../blib/lib ../blib/arch); $TEST_DIR = '.'; # for debugging

##----------------------------------------------------------------------
## load modules
use PDL;
use PDL::SVDSLEPc;

##----------------------------------------------------------------------
## setup
$a = pdl(double,
	 [[10,0,0,0,-2,0,0],
	  [3,9,0,0,0,3,1],
	  [0,7,8,7,0,0,0],
	  [3,0,8,7,5,0,1],
	  [0,8,0,9,9,13,0],
	  [0,4,0,0,2,-1,1]]);

$ptr=pdl(long,[0,3,7,9,12,16,19, 22]);
$colids=pdl(long,[0,1,3,1,2,4,5,2,3,2,3,4,0,3,4,5,1,4,5,1,3,5]);
$nzvals=pdl(double,[10,3,3,9,7,8,4,8,8,7,7,9,-2,5,9,2,3,13,-1,1,1,1]);

($n,$m) = $a->dims;


##----------------------------------------------------------------------
## common subs
sub min2 {
  return $_[0]<$_[1] ? $_[0] : $_[1];
}
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
sub svdtest {
  my ($label, $u,$s,$v, $eps_s,$eps_v) = @_;
  my $ns = $s->nelem;
  $eps_s = .01    if (!defined($eps_s));
  $eps_v = $eps_s if (!defined($eps_v));
  ok( all($s->approx($s_want->slice("0:".($ns-1)),$eps_s)), "${label} - s [eps=$eps_s]" );
  ok( all(svdcompose($u,$s,$v)->approx($a,$eps_v)), "${label} - vals [eps=$eps_v]" );
}

##-- expected values
$d  = min2($n,$m);
$d1 = $d-1;
$s_want = pdl(double,
	      [23.3228474410401, 12.9401616781924, 10.9945440916999, 9.08839598479767, 3.84528764361343, 1.1540470359863, 0]);

##----------------------------------------------------------------------
## builtin

##-- test 1..2 : builtin svd, d=min{m,n}
($u,$s,$v) = svdreduce(svd($a),$d);
svdtest("PDL::svd - d=$d=min{m,n}", $u,$s,$v, .01);

##-- test 3..4 : builtin svd, d<min{m,n}
($u,$s,$v) = svdreduce(svd($a),$d1);
svdtest("PDL::svd - d=$d1<min{m,n}", $u,$s,$v, .01,.5);

##----------------------------------------------------------------------
## _slepc_svd_crs(): basic

##-- test 5..6 : _slepc_svd_crs(), d=min{m,n}
_slepc_svd_crs($ptr,$colids,$nzvals, $u=zeroes($d,$m),$s=zeroes($d),$v=zeroes($d,$n), []);
svdtest("_slepc_svd_crs - d=$d=min{m,n}", $u,$s,$v, .01,.01);

##-- test 7..8 : _slepc_svd_crs(), d<{m,n}
_slepc_svd_crs($ptr,$colids,$nzvals, $u=zeroes($d1,$m),$s=zeroes($d1),$v=zeroes($d1,$n), []);
svdtest("_slepc_svd_crs - d=$d1<min{m,n}", $u,$s,$v, .01,.5);

##----------------------------------------------------------------------
## _slepc_svd_crs(): transposed

##-- test 9..10 : _slepc_svd_crs(), transposed, d=min{m,n}
my $xptr    = pdl(long,[0,2,6,9,14,18,22]);
my $xcolids = pdl(long,[0,4,0,1,5,6,1,2,3,0,2,3,4,6,1,3,4,5,1,4,5,6]);
my $xnzvals = pdl(double,[10,-2,3,9,3,1,7,8,7,3,8,7,5,1,8,9,9,13,4,2,-1,1]);
_slepc_svd_crs($xptr,$xcolids,$xnzvals, $v=zeroes($d,$n),$s=zeroes($d),$u=zeroes($d,$m), []);
svdtest("_slepc_svd_crs - transposed - d=$d=min{m,n}", $u,$s,$v, .01,.01);

##-- test 11..12 : _slepc_svd_crs(), transposed, d<{m,n}
_slepc_svd_crs($xptr,$xcolids,$xnzvals, $v=zeroes($d1,$n),$s=zeroes($d1),$u=zeroes($d1,$m), []);
svdtest("_slepc_svd_crs - transposed - d=$d1<min{m,n}", $u,$s,$v, .01,.5);

##----------------------------------------------------------------------
## _slepc_svd_crs(): lanczos, trlanczos

##-- test 13..14 : _slepc_svd_crs(), d=min{m,n}
_slepc_svd_crs($ptr,$colids,$nzvals, $u=zeroes($d,$m),$s=zeroes($d),$v=zeroes($d,$n), [qw(-svd_type lanczos)]);
svdtest("_slepc_svd_crs - lanczos", $u,$s,$v, .01,.01);

##-- test 15..16 : _slepc_svd_crs(), d<{m,n}
_slepc_svd_crs($ptr,$colids,$nzvals, $u=zeroes($d,$m),$s=zeroes($d),$v=zeroes($d,$n), [qw(-svd_type trlanczos)]);
svdtest("_slepc_svd_crs - trlanczos", $u,$s,$v, .01,.01);


# end of t/01_svd.t

