#!/usr/bin/perl -w

use lib qw(./blib/lib ./blib/arch);
use PDL;
use PDL::SVDSLEPc;
use PDL::CCS;
use PDL::CCS::IO::PETSc;
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
use vars qw($p $whichi $whichv $ptr $colids $nzvals $a $m $n $d $d1);
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
  $d1     = $d-1;
  ##
  ##-- ccs encode
  ($ptr,$colids,$nzvals)  = ccsencode($p);
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

  ##-- DEBUG: save petsc matrix
  #$a->toccs->wpetsc("a.petsc");
  #$a->toccs->xchg(0,1)->make_physically_indexed->wpetsc("at.petsc");
}

##-- common subs
sub min2 { return $_[0]<$_[1] ? $_[0] : $_[1]; }
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
## test: svd computation: basic
sub svderrs {
  my ($label,$x,$y) = @_;
  my $e = ($x->flat->abs-$y->flat->abs)->abs;
  return sprintf("absolute error: %--16s (min/max/avg) = %8.2g / %8.2g / %8.2g\n", $label, $e->minmax, $e->avg);
}
sub test_svd {
  tdata();
  print STDERR "data(m=".$a->dim(0).",n=".$a->dim(1)."); d=$d1\n";

  my ($u0,$s0,$v0) = svdreduce(svd($a),$d1);
  if (1) {
    print STDERR
      ("builtin:\n",
       " + s0(".$s0->nelem.") = $s0\n",
       " + u0(".join(',',$u0->dims).") = $u0",
       " + v0(".join(',',$v0->dims).") = $v0");
  }

  my $u = zeroes($d1,$m);
  my $s = zeroes($d1);
  my $v = zeroes($d1,$n);
  my @opts = (@_);
  _slepc_svd_crs($ptr,$colids,$nzvals, $u,$s,$v, \@opts);
  if (1) {
    #print STDERR "a=$a; ptr=$ptr; colids=$colids; nzvals=$nzvals\n";
    print STDERR
      ("slepc:\n",
       " + s(".$s->nelem.") = $s\n",
       " + u(".join(',',$u->dims).") = $u",
       " + v(".join(',',$v->dims).") = $v");
  }

  ##-- check errors
  local $,='';
  print STDERR
    (svderrs("u0-u",$u0,$u),
     svderrs("s0-s",$s0,$s),
     svderrs("v0-v",$v0,$v),
     svderrs("a-(u0,s0,v0)",svdcompose($u0,$s0,$v0),$a),
     svderrs("a-(u,s,v)",svdcompose($u,$s,$v),$a),
    );
}
#test_svd(@ARGV); exit 0;

##---------------------------------------------------------------------
## test: svd computation: transpose
sub test_svdx {
  tdata();

  my $ax = $a->xchg(0,1);
  my $dx = min2($ax->dims)-1;
  my ($xptr,$xcolids,$xvals) = ccsencode($ax);
  $xptr->reshape($xptr->nelem+1);
  $xptr->set(-1, $xcolids->nelem);
  print STDERR "xdata(xm=".$ax->dim(0).",xn=".$ax->dim(1)."); d=$dx\n";

  my ($u0,$s0,$v0) = svdreduce(svd($ax),$dx);
  if (0) {
    print STDERR
      ("builtin:\n",
       " + s0(".$s0->nelem.") = $s0\n",
       " + u0(".join(',',$u0->dims).") = $u0",
       " + v0(".join(',',$v0->dims).") = $v0");
  }
  #exit 0;

  my $u = zeroes($dx,$ax->dim(1));
  my $s = zeroes($dx);
  my $v = zeroes($dx,$ax->dim(0));
  my @opts = (@_);
  _slepc_svd_crs($xptr,$xcolids,$xvals, $u,$s,$v, \@opts);
  if (0) {
    #print STDERR "a=$a; ptr=$ptr; colids=$colids; nzvals=$nzvals\n";
    print STDERR
      ("slepc:\n",
       " + s(".$s->nelem.") = $s\n",
       " + u(".join(',',$u->dims).") = $u",
       " + v(".join(',',$v->dims).") = $v");
  }

  ##-- check errors
  local $,='';
  print STDERR
    (svderrs("u0-u",$u0,$u),
     svderrs("s0-s",$s0,$s),
     svderrs("v0-v",$v0,$v),
     svderrs("a-(u0,s0,v0)",svdcompose($u0,$s0,$v0),$a->xchg(0,1)),
     svderrs("a-(u,s,v)",svdcompose($u,$s,$v),$a->xchg(0,1)),
    );
}
test_svdx(@ARGV); exit 0;


##---------------------------------------------------------------------
## DUMMY
##---------------------------------------------------------------------
foreach $i (0..3) {
  print "--dummy($i)--\n";
}

