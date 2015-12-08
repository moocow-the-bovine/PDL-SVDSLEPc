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
  ($m,$n) = $p->dims;
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
  #_slepc_svd_int(null,null,null, null,null,null, 0,0,0,['-help']);

  ##-- redirect stderr (ugly, but works)
  if (0) {
    my ($buf);
    open(my $oldout, '>&STDOUT');
    open(STDOUT, ">&STDERR")
      or die("$0: failed to redirect STDOUT to STDERR: $!");
    slepc_svd_help();
    *STDOUT = *$oldout;
  }

  ##-- redirect string (?)
  if (0) {
    my ($buf);
    open(my $oldout, '>&STDOUT');
    open(my $bufh, ">", \$buf)
      or die("$0: failed to open string-buffer handle: $!");
    slepc_svd_help();
    *STDOUT = *$oldout;
    print "help = $buf";
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
  _slepc_svd_int($ptr,$colids,$nzvals, $u,$s,$v, $m,$n,$d,\@opts);
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

  ($u,$s,$v) = _slepc_svd_int($ptr,$colids,$nzvals, $m,$n,$d1,[@_]);
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
  if (1) {
    print STDERR
      ("builtin-x:\n",
       " + s0(".$s0->nelem.") = $s0\n",
       " + u0(".join(',',$u0->dims).") = $u0",
       " + v0(".join(',',$v0->dims).") = $v0");
  }
  #exit 0;

  ($u,$s,$v) = _slepc_svd_int($xptr,$xcolids,$xvals, $n,$m,$dx,[@_]);
  if (1) {
    #print STDERR "a=$a; ptr=$ptr; colids=$colids; nzvals=$nzvals\n";
    print STDERR
      ("slepc-x:\n",
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
     svderrs("a-(u0,s0,v0)",svdcompose($u0,$s0,$v0),$ax),
     svderrs("a-(u,s,v)",svdcompose($u,$s,$v),$ax),
    );
}
#test_svdx(@ARGV); exit 0;

##---------------------------------------------------------------------
## test: svd computation: ccs
sub test_svd_ccs {
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

  ($u,$s,$v) = $a->toccs->slepc_svd($d1);
  if (1) {
    #print STDERR "a=$a; ptr=$ptr; colids=$colids; nzvals=$nzvals\n";
    print STDERR
      ("slepc-ccs:\n",
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
test_svd_ccs(@ARGV); exit 0;

##---------------------------------------------------------------------
## test: wrapper

#BEGIN { *PDL::SVDSLEPc::slepc_svd = *PDL::slepc_svd = \&slepc_svd; }
sub slepc_svd0 {
  my ($rowptr,$colids,$nzvals, @args) = @_;

  ##-- parse arguments into @pdls=($u,$s,$v), @dims=($m,$n,$d), @opts=(...)
  my (@pdls,@dims,@opts);
  foreach my $arg (@args) {
    if (@pdls < 3 && UNIVERSAL::isa($arg,'PDL')) {
      ##-- output pdl
      push(@pdls,$arg);
    }
    elsif (@dims < 3 && ((UNIVERSAL::isa($arg,'PDL') && $arg->nelem==1) || !ref($arg))) {
      ##-- dimension argument
      push(@dims, UNIVERSAL::isa($arg,'PDL') ? $arg->sclr : $arg);
    }
    elsif (UNIVERSAL::isa($arg,'ARRAY')) {
      ##-- option array
      push(@opts,@$arg);
    }
    elsif (UNIVERSAL::isa($arg,'HASH')) {
      ##-- option hash: pass boolean flags as ("-FLAG"=>undef), e.g. "-svd_view"=>undef
      push(@opts, map {((/^\-/ ? $_ : "-$_"),(defined($arg->{$_}) ? $arg->{$_} : qw()))} keys %$arg);
    }
    else {
      ##-- extra parameter: warn
      warn(__PACKAGE__ . "::slepc_svd(): ignoring extra parameter '$arg'");
    }
  }

  ##-- extrac -svd_nsv ($d) option
  my $nsv = undef;
  foreach (0..($#opts-1)) {
    $nsv = $opts[$_+1] if ($opts[$_] eq '-svd_nsv');
  }

  ##-- extract arguments
  my ($u,$s,$v) = @pdls;
  my ($m,$n,$d) = @dims;
  $m = defined($v) && !$v->isempty ? $v->dim(1) : $rowptr->nelem-1  if (!defined($m));
  $n = defined($u) && !$u->isempty ? $u->dim(1) : $colids->max+1 if (!defined($n));
  $d = (defined($u) && !$u->isempty ? $u->dim(0)
	: (defined($s) && !$s->isempty ? $s->dim(0)
	   : (defined($v) && !$v->isempty ? $v->dim(0)
	      : (defined($nsv) ? $nsv
		 : $m < $n ? $m : $n))))
    if (!defined($d));

  ##-- create output piddles
  $u = zeroes(double, $d,$n) if (!defined($u) || $u->isempty);
  $s = zeroes(double, $d)    if (!defined($s) || $s->isempty);
  $v = zeroes(double, $d,$m) if (!defined($v) || $v->isempty);

  ##-- call guts
  _slepc_svd_int($rowptr,$colids,$nzvals, $u,$s,$v, $m,$n,$d, \@opts);
  return ($u,$s,$v);
}

our ($u0,$s0,$v0);
sub test_svd_wrap {
  tdata();
  print STDERR "data(m=".$a->dim(0).",n=".$a->dim(1)."); d=$d1\n";

  ($u0,$s0,$v0) = svdreduce(svd($a),$d1);
  if (1) {
    print STDERR
      ("builtin:\n",
       " + s0(".$s0->nelem.") = $s0\n",
       " + u0(".join(',',$u0->dims).") = $u0",
       " + v0(".join(',',$v0->dims).") = $v0");
  }

  ($u,$s,$v) = slepc_svd($ptr,$colids,$nzvals, [@_,'-svd_nsv'=>$d1]);
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
#test_svd_wrap(@ARGV); exit 0;


##---------------------------------------------------------------------
## DUMMY
##---------------------------------------------------------------------
foreach $i (0..3) {
  print "--dummy($i)--\n";
}

