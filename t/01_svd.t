# -*- Mode: CPerl -*-
# t/01_svd.t: test SLEPc svd
use Test::More tests=>4, todo=>[];

$TEST_DIR = './t';
#use lib qw(../blib/lib ../blib/arch); $TEST_DIR = '.'; # for debugging

##-- load modules
use PDL;
use PDL::SVDSLEPc;

##-- setup
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

##-- $d==$n: expect
$d  = $n < $m ? $n : $m;
$d1 = $d-1;

$s_want = pdl(double,
	      [23.3228474410401, 12.9401616781924, 10.9945440916999, 9.08839598479767, 3.84528764361343, 1.1540470359863, 0]);
$s1_want = $s_want->slice("0:".($d1-1));

##-- test 1..2 : builtin svd
($u,$s,$v) = svd($a);
($u,$s,$v) = svdreduce($u,$s,$v, $d);
ok( all($s->approx($s_want->slice("0:".($d-1)),.01)), "svd,d=$d:s" );
ok( all(svdcompose($u,$s,$v)->approx($a,.01)), "svd,d=$d:vals" );

##-- test 3..4 : _slepc_svd_crs()
_slepc_svd_crs($ptr,$colids,$nzvals, $u=zeroes($d,$m),$s=zeroes($d),$v=zeroes($d,$n), []);
ok( all($s->approx($s_want->slice("0:".($d-1)),.5)), "_slepc_svd_crs,d=$d:s" );
ok( all(svdcompose($u,$s,$v)->approx($a,.5)), "_slepc_svd_crs,d=$d:vals" );


# end of t/01_svd.t

