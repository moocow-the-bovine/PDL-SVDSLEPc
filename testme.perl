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
use vars qw($p $whichi $whichv $ptr $rowids $nzvals $a);
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
  ##
  ##-- ccs encode
  ($ptr,$rowids,$nzvals) = ccsencode($p);
  #$ptr = $ptr->convert(longlong);
  #$rowids = $rowids->convert(longlong);
  ##
  ##-- dok encode
  $whichi = $p->whichND->vv_qsortvec;
  $whichv = $p->indexND($whichi);
  ##
  ##-- HACK: allocate an extra slot in $ptr
  $ptr->reshape($ptr->nelem+1);
  $ptr->set(-1, $rowids->nelem);
}

##---------------------------------------------------------------------
## test: constants
sub test_const {
  local $,= '';
  print "slepc_version = ", PDL::SVDSLEPc::slepc_version(), "\n";
  print "petsc_version = ", PDL::SVDSLEPc::petsc_version(), "\n";
  print "library_version [list] = ", PDL::SVDSLEPc::library_version(), "\n";
  print "library_version [sclr] = ", scalar(PDL::SVDSLEPc::library_version()), "\n";
}
test_const;

##---------------------------------------------------------------------
## DUMMY
##---------------------------------------------------------------------
foreach $i (0..3) {
  print "--dummy($i)--\n";
}

