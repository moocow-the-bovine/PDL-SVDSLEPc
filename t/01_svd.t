# -*- Mode: CPerl -*-
# t/01_svd.t: test SLEPc svd
use Test::More tests=>28, todo=>[];

$TEST_DIR = './t';
#use lib qw(../blib/lib ../blib/arch); $TEST_DIR = '.'; # for debugging

# load common subs
use Test;
do "$TEST_DIR/common.plt";
use PDL;
use PDL::SVDLIBC;

##-- setup
$a = pdl(double,
	 [[10,0,0,0,-2,0,0],
	  [3,9,0,0,0,3,1],
	  [0,7,8,7,0,0,0],
	  [3,0,8,7,5,0,1],
	  [0,8,0,9,9,13,0],
	  [0,4,0,0,2,-1,1]]);

$ptr=pdl(long,[0,3,7,9,12,16,19, 22]);
$rowids=pdl(long,[0,1,3,1,2,4,5,2,3,2,3,4,0,3,4,5,1,4,5,1,3,5]);
$nzvals=pdl(double,[10,3,3,9,7,8,4,8,8,7,7,9,-2,5,9,2,3,13,-1,1,1,1]);

($n,$m) = $a->dims;


##-- common pars
$iters = pdl(long,14);
$end   = pdl(double,[-1e-30,1e-30]);
$kappa = pdl(double,1e-6);

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
$d  = $n;
$d1 = $n-2;

$s_want = pdl(double,
	      [23.32284744104,12.9401616781924,10.9945440916999,9.08839598479768,3.84528764361343,1.1540470359863,0]);
$s1_want = $s_want->slice("0:".($d1-1));


##-- test 1..2 : svdlas2, d=n
svdlas2($ptr,$rowids,$nzvals, $m,
	$iters, $end, $kappa,
	($ut=zeroes(double,$m,$d)),
	($s=zeroes(double,$d)),
	($vt=zeroes(double,$d,$n)),
       );
isok("svdlas2,d=n:s",    all($s->approx($s_want)));
isok("svdlas2,d=n:data", all(svdcomposet($ut,$s,$vt)->approx($a)));

##-- test 3..4 : svdlas2a, d=n
($ut,$s,$vt) = svdlas2a($ptr,$rowids,$nzvals);
isok("svdlas2a,d=n:s",    all($s->approx($s_want)));
isok("svdlas2a,d=n:data", all(svdcomposet($ut,$s,$vt)->approx($a)));

##-- test 5..6 : svdlas2a, d<n
($ut,$s,$vt) = svdlas2a($ptr,$rowids,$nzvals, $m,$d1);
isok("svdlas2a,d<n:s",    all($s->approx($s1_want)));
isok("svdlas2a,d<n:data", all(svdcomposet($ut,$s,$vt)->approx($a,0.5)));

##-- test 7..8 : svdlas2w, d=n
my $whichi = $a->whichND->qsortvec->xchg(0,1);
my $whichv = $a->indexND($whichi->xchg(0,1));
svdlas2w($whichi,$whichv, $n,$m,
	 $iters, $end, $kappa,
	 ($ut=zeroes(double,$m,$d)),
	 ($s=zeroes(double,$d)),
	 ($vt=zeroes(double,$d,$n)),
	);
isok("svdlas2w,d=n:s",    all($s->approx($s_want)));
isok("svdlas2w,d=n:data", all(svdcomposet($ut,$s,$vt)->approx($a)));

##-- test 9..10 : svdlas2aw, d=n
($ut,$s,$vt) = svdlas2aw($whichi,$whichv);
isok("svdlas2aw,d=n:s",    all($s->approx($s_want)));
isok("svdlas2aw,d=n:data", all(svdcomposet($ut,$s,$vt)->approx($a)));

##-- test 11..12 : svdlas2aw, d<n
($ut,$s,$vt) = svdlas2aw($whichi,$whichv, $n,$m,$d1);
isok("svdlas2aw,d<n:s",    all($s->approx($s1_want)));
isok("svdlas2aw,d<n:data", all(svdcomposet($ut,$s,$vt)->approx($a,0.5)));

##-- test 13..14 : svdlas2aw, d=n, transpsosed whichND
$whichi = $a->whichND->qsortvec;
$whichv = $a->indexND($whichi);
($ut,$s,$vt) = svdlas2aw($whichi,$whichv);
isok("svdlas2aw,whichT,d=n:s",    all($s->approx($s_want)));
isok("svdlas2aw,whichT,d=n:data", all(svdcomposet($ut,$s,$vt)->approx($a)));


##-- test 15..16 : svdlas2d, d=n
svdlas2d($a,
	 $iters, $end, $kappa,
	 ($ut=zeroes(double,$m,$d)),
	 ($s=zeroes(double,$d)),
	 ($vt=zeroes(double,$d,$n)),
	);
isok("svdlas2d,d=n:s",    all($s->approx($s_want)));
isok("svdlas2d,d=n:data", all(approx($ut->xchg(0,1) x stretcher($s) x $vt, $a)));

##-- test 17..18 : svdlas2ad
($ut,$s,$vt) = svdlas2ad($a);
isok("svdlas2ad,d=n:s",    all($s->approx($s_want)));
isok("svdlas2ad,d=n:data", all(approx($ut->xchg(0,1) x stretcher($s) x $vt, $a)));

##-- test 19..20 : svdlas2ad, d<n
($ut,$s,$vt) = svdlas2ad($a,$d1);
isok("svdlas2a,d<n:s",    all($s->approx($s1_want)));
isok("svdlas2a,d<n:data", all(svdcomposet($ut,$s,$vt)->approx($a,0.5)));

##-- test 21..24: decode+error (PDL::MatrixOps::svd(), full)
($u,$s,$v) = svd($a);
isok("svdindexND,d=n", all(svdindexND($u,$s,$v, $whichi)->approx($whichv,1e-5)));
isok("svdindexNDt,d=n", all(svdindexNDt($u->xchg(0,1),$s,$v->xchg(0,1), $whichi)->approx($whichv,1e-5)));
isok("svdindexccs,d=n", all(svdindexccs($u,$s,$v, $ptr,$rowids)->approx($whichv,1e-5)));
isok("svderror,d=n", svderror($u,$s,$v, $ptr,$rowids,$nzvals)->approx( svdwanterr($a,$u,$s,$v) ));

##-- test 25..28: decode+error (PDL::MatrixOps::svd(), whichND, reduced);
($ur,$sr,$vr) = svdreduce($u,$s,$v,$d1);
isok("svdindexND,d<n", all(svdindexND($ur,$sr,$vr, $whichi)->approx($whichv,0.5)));
isok("svdindexNDt,d<n", all(svdindexNDt($ur->xchg(0,1),$sr,$vr->xchg(0,1), $whichi)->approx($whichv,0.5)));
isok("svdindexccs,d<n", all(svdindexccs($ur,$sr,$vr, $ptr,$rowids)->approx($whichv,0.5)));
isok("svderror,d<n", svderror($ur,$sr,$vr, $ptr,$rowids,$nzvals)->approx( svdwanterr($a,$ur,$sr,$vr) ));

print "\n";

BEGIN { plan tests=>28, todo=>[]; }
# end of t/01_svd.t

