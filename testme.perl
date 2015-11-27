#!/usr/bin/perl -w

use lib qw(./blib/lib ./blib/arch);
use PDL;
use PDL::SVDLIBC;
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

use vars qw($ptr1 $rowids1 $nzvals1);
sub tccs {
  ($n,$m) = $p->dims;
  $nnz = $p->flat->nnz;
  _svdccsencode($p,
		($ptr1=zeroes(long,$n+1)-1),
		($rowids1=zeroes(long,$nnz)-1),
		($nzvals1=zeroes(long,$nnz)-1));
}

use vars qw($iters $end $kappa $ut $ul $ssl $vt $vl $pbr $pbri);
sub tlas2 {
  #$d=4;
  tdata() if (!defined($p));
  $d = $p->dim(0);

  #$test_a=1;
  if ($test_a) {
    ##-- convenience method
    ($u2t,$s2,$v2t) = svdlas2a($ptr,$rowids,$nzvals);
  } else {
    ##-- guts
    svdlas2($ptr,$rowids,$nzvals, $m,
	    ($iters=pdl(14)),
	    ($end=pdl(double,[-1e-30,1e-30])),
	    ($kappa=pdl(1e-6)),
	    ($u2t = $u2t = zeroes(double,$m,$d)),
	    ($s2  = $s2  = zeroes(double,$d)),
	    ($v2t = $v2t = zeroes(double,$n,$d)), ##-- $d-by-$n
	   );
    ## u2t(m=6,d=7), s2(d=7), v2t(n=7,d=7)
  }

  ##-- stretch, tranpose
  use vars qw($s2s);
  $u2  = $u  = $u2t->xchg(0,1);
  $s2s = $ss = stretcher($s2);
  $v2  = $v  = $v2t->xchg(0,1);
  tcheck('las2', $u2, $s2, $v2);
  return;
}
#tlas2();

use vars qw($uwt $sw $vwt $uw $ssw $vw);
sub tlas2w {
  #$d=4;
  tdata() if (!defined($p));
  $d = $p->dim(0);

  $test_a=1;
  if ($test_a) {
    ##-- convenience method
    ($uwt,$sw,$vwt) = svdlas2aw($whichi,$whichv);
  } else {
    ##-- guts
    svdlas2w($whichi->xchg(0,1),$whichv, $n,$m,
	    ($iters=pdl(14)),
	    ($end=pdl(double,[-1e-30,1e-30])),
	    ($kappa=pdl(1e-6)),
	    ($uwt = zeroes(double,$m,$d)),
	    ($sw  = zeroes(double,$d)),
	    ($vwt = zeroes(double,$n,$d)), ##-- $d-by-$n
	   );
    ## uwt(m=6,d=7), sw(d=7), vwt(n=7,d=7)
  }

  ##-- stretch, tranpose
  use vars qw($s2s);
  $uw  = $uwt->xchg(0,1);
  $ssw = stretcher($sw);
  $vw  = $vwt->xchg(0,1);
  tcheck('las2w', $uw, $sw, $vw);
  return;
}
#tlas2w();


use vars qw($udt $ud $sd $ssd $vdt $vd);
sub tlas2d { #-- ok
  tdata() if (!defined($p));
  $d = $p->dim(0);

  #$test_a=1;
  if ($test_a) {
    ##-- convenience wrappers
    ($udt,$sd,$vdt) = svdlas2ad($p);
  } else {
    ##-- guts
    svdlas2d($p,
	     ($iters=pdl(14)),
	     ($end=pdl(double,[-1e-30,1e-30])),
	     ($kappa=pdl(1e-6)),
	     ($udt = $ut = zeroes(double,$m,$d)),
	     ($sd  = $s  = zeroes(double,$d)),
	     ($vdt = $vt = zeroes(double,$n,$d)),
	    );
    ## udt(m=6,d=7), sd(d=7), vdt(n=7,d=7)
  }

  ##-- stretch, tranpose
  $ud  = $u  = $udt->xchg(0,1);
  $vd  = $v  = $vdt->xchg(0,1);
  $ssd = $ss = stretcher($sd);
  tcheck('las2d', $u, $sd, $v);
  return;
}
#tlas2d();

sub tbuiltin {
  ##-- test: compare w/ builtin svd
  ($ub,$sb,$vb) = svd($p);
  $ssb = stretcher($sb);
}


##-- check copy-ability
sub tcheck {
  my ($label,$u,$s,$v) = @_;
  $label = '(nolabel)' if (!$label);

  ##-- hack
  $u->inplace->setnantobad->inplace->setbadtoval(0);
  $s->inplace->setnantobad->inplace->setbadtoval(0);
  $v->inplace->setnantobad->inplace->setbadtoval(0);

  ##-- test: copy
  $p2  = $u x stretcher($s) x $v->xchg(0,1);

  ##-- check
  print "$label : ", (all($p2->approx($p),$eps) ? "ok" : "NOT ok"), "\n";
}

sub checkall {
  tdata() if (!defined($p));

  tbuiltin;
  tcheck('builtin', $ub,$sb,$vb);

  tlas2d;
  tcheck('las2d', $ud, $sd, $vd);

  tlas2; ##-- strangeness with SVDLIBC 'long' vs. PDL 'int' or 'longlong' (now using: int)
  tcheck('las2', $u2, $s2, $v2);
}
#checkall;




##--------------------------------------------------------------------
## Restricted

use vars qw($ubr $sbr $ssbr $vbr $ur $sr $ssr $vr $pr $pri);
sub tbuiltinr {
  $dr=4 if (!defined($dr));
  $d=$dr;
  $d_1 = $d-1;

  ($ub,$sb,$vb)=($u,$s,$v)=svd($p);
  $ssb = $ss = stretcher($sb);

  ##-- restrict
  $ubr  = $ur  = $ub->slice("0:$d_1,:");
  $sbr  = $sr  = $sb->slice("0:$d_1");
  $ssbr = $ssr = stretcher($sbr);
  $vbr  = $vr  = $v->slice("0:$d_1,:");
}


use vars qw($ulr $slr $sslr $vlr);
sub tlas2dr {
  tdata() if (!defined($p));
  $dr=4 if (!defined($dr));
  $d=$dr;
  $d_1 = $d-1;

  svdlas2d($p,
	   ($iters=pdl(14)),
	   ($end=pdl(double,[-1e-30,1e-30])),
	   ($kappa=pdl(1e-6)),
	   ($udtr=zeroes(double,$m,$d)),
	   ($sdr=zeroes(double,$d)),
	   ($vdtr=zeroes(double,$n,$d)), ##-- $n-by-$d
	  );

  ##-- stretch, tranpose
  $udr  = $ur  = $udtr->xchg(0,1);
  $vdr  = $vr  = $vdtr->xchg(0,1);
  $ssdr = $ssr = stretcher($sdr);

  ##-- apply restriction
  #$plr  = $pr  = $p x $vlr;
  #$plri = $pri = ($vr x $ssr x $ur->xchg(0,1))->slice(":,:,(0)")->xchg(0,1);
}

sub tlas2r {
  tdata if (!defined($p));
  $dr=4 if (!defined($dr));
  $d=$dr;
  $d_1 = $d-1;

  svdlas2($ptr,$rowids,$nzvals, $m,
	  ($iters=pdl(14)),
	  ($end=pdl(double,[-1e-30,1e-30])),
	  ($kappa=pdl(1e-6)),
	  ($u2rt=zeroes(double,$m,$d)),
	  ($s2r=zeroes(double,$d)),
	  ($v2rt=zeroes(double,$n,$d)), ##-- $n-by-$d
	 );

  ##-- stretch, tranpose
  $u2r  = $ur  = $u2rt->xchg(0,1);
  $v2r  = $vr  = $v2rt->xchg(0,1);
  $ss2r = $ssr = stretcher($s2r);

  ##-- apply restriction
  #$plr  = $pr  = $p x $vlr;
  #$plri = $pri = ($vr x $ssr x $ur->xchg(0,1))->slice(":,:,(0)")->xchg(0,1);
}


##-- tcheckr($label, $ur,$sr,$vr)
##   + uses current values of $dr, $d_1, $ur, $ssr, $vr  : ?!?!?!
use vars qw($ss2r $ssdr);
sub tcheckr {
  my ($label,$ur,$sr,$vr)=@_;
  $label = '(nolabel)' if (!$label);

  ##-- bad-value hack
  $ur->inplace->setnantobad->inplace->setbadtoval(0);
  $sr->inplace->setnantobad->inplace->setbadtoval(0);
  $vr->inplace->setnantobad->inplace->setbadtoval(0);

  ##-- apply restriction
  $plr  = $pr  = $p x $vr;
  $plri = $pri = ($vr x stretcher($sr) x $ur->xchg(0,1))->slice(":,:,(0)")->xchg(0,1);

  print
    ("$label : decomp(",
     (all($plr->approx($ur x stretcher($sr))) ? "ok" : "NOT ok"),
     "), ",
     "avg(err)=",sprintf("%.2g", abs($plri-$p)->avg), "\n",
    );
}


sub checkallr {
  tdata() if (!defined($p));

  tbuiltinr;
  tcheckr('builtin', $ubr,$sbr,$vbr);

  tlas2dr;
  tcheckr('las2d', $udr, $sdr, $vdr);

  tlas2r;
  tcheckr('las2', $u2r, $s2r, $v2r);
}
#checkallr;

##--------------------------------------------------------------------
## Lookup

sub tcheckvals {
  my ($label,$which,$vals,$eps) = @_;
  my $want = $p->indexND($which);
  $label = '(nolabel)' if (!$label);
  $eps   = 1e-5 if (!$eps);
  print
    ("$label: vals: ", (all($vals->approx($want,$eps)) ? "ok": "NOT ok"), "\n",
     "  + avg(err)=",sprintf("%.2g", abs($vals-$want)->avg), "\n",
     "  + avg(err%)=",sprintf("%.2g %%", 100*(abs($vals-$want)/$want)->avg), "\n",
     "  + avg(s^2)=",sprintf("%.2g", (($vals-$want)**2)->avg), "\n",
     "  + sum(s^2)=",sprintf("%.2g", (($vals-$want)**2)->sum), "\n",
    );
}

sub svdreduce {
  my ($u,$s,$v,$d) = @_;
  $d = $s->nelem if (!defined($d) || $d > $s->nelem);
  my $end = $d-1;
  return ($u->slice("0:$end,:"),$s->slice("0:$end"),$v->slice("0:$end,"));
}

sub tindexnd {
  tdata() if (!defined($p));

  ##-- want data
  my $want = $whichv;
  my ($u,$s,$v,$vals, $ur,$sr,$vr, $ut,$vt, $whichp);

  ##-- svd
  ($u,$s,$v) = svd($a);
  $vals = svdindexND($u,$s,$v, $whichi);
  tcheckvals("svd+svdindexND(d=n)", $whichi,$vals);
  ##
  $vals = svdindexNDt($u->xchg(0,1),$s,$v->xchg(0,1), $whichi);
  tcheckvals("svd+svdindexNDt(d=n)", $whichi,$vals);

  ##-- svd+reduce
  ($ur,$sr,$vr) = svdreduce($u,$s,$v, 5);
  $vals = svdindexND($ur,$sr,$vr, $whichi);
  tcheckvals("svd+svdindexNDt(d<n)", $whichi,$vals,0.5);
  ##
  $vals = svdindexNDt($ur->xchg(0,1),$sr,$vr->xchg(0,1), $whichi);
  tcheckvals("svd+svdindexNDt(d<n)", $whichi,$vals,0.5);

  ##-- svdlas2ad()
  ($ut,$s,$vt) = svdlas2ad($a);
  $vals = svdindexNDt($ut,$s,$vt, $whichi);
  tcheckvals("svdlas2ad+svdindexNDt(d=n)", $whichi,$vals);

  ##-- svdlas2ad() + reduce
  ($ut,$s,$vt) = svdlas2ad($a, 5);
  $vals = svdindexNDt($ut,$s,$vt, $whichi);
  tcheckvals("svdlas2ad+svdindexNDt(d<n)", $whichi,$vals,0.5);

  ##-- svdlas2a()
  ($ut,$s,$vt) = svdlas2a($ptr,$rowids,$nzvals, $m);
  $vals = svdindexNDt($ut,$s,$vt, $whichi);
  tcheckvals("svdlas2a+svdindexNDt(d=n)", $whichi,$vals);

  ##-- svdlas2a() + reduce
  ($ut,$s,$vt) = svdlas2a($ptr,$rowids,$nzvals, $m,5);
  $vals = svdindexNDt($ut,$s,$vt, $whichi);
  tcheckvals("svdlas2a+svdindexNDt(d<n)", $whichi,$vals,0.5);

  ##-- svdlas2a() + reduce: indexccs()
  ($ut,$s,$vt) = svdlas2a($ptr,$rowids,$nzvals, $m,5);
  $whichp = ccswhichND($ptr,$rowids);
  $vals   = svdindexccs($ut->xchg(0,1),$s,$vt->xchg(0,1), $ptr,$rowids);
  tcheckvals("svdlas2a+indexccs(d<n)", $whichp,$vals,0.5);
}
tindexnd(); exit 0;

##--------------------------------------------------------------------
## sum-of-squared errors

sub tcheckerr {
  my ($label,$u,$s,$v,$err, $eps) = @_;
  my $want = (($a-($u * $s)->matmult($v->xchg(0,1)))**2)->flat->sumover;
  $label = '(nolabel)' if (!$label);
  $eps   = 1e-5 if (!$eps);
  print "$label: err: ", ($want->approx($err) ? "ok" : "NOT ok"), "\n",
}

sub tsvderror {
  tdata() if (!defined($p));
  my ($u,$s,$v, $err, $ur,$sr,$vr, $svdvals,$err_nz,$err_z);

  ##-- svd
  ($u,$s,$v) = svd($a);
  $err = svderror($u,$s,$v, $ptr,$rowids,$nzvals);
  tcheckerr("svd,d=n", $u,$s,$v,$err);

  ##-- svd:reduced
  ($ur,$sr,$vr) = svdreduce($u,$s,$v,5);
  $err = svderror($ur,$sr,$vr, $ptr,$rowids,$nzvals);
  tcheckerr("svd,d<n", $ur,$sr,$vr,$err);

  ##-- svd:reduced, approx
  $svdvals = svdindexND($ur,$sr,$vr, $whichi);
  $err_nz  = ($nzvals-$svdvals)->pow(2)->sumover;
  $err
  
}
tsvderror(); exit 0;

sub tsvderror_dbg {
  tdata() if (!defined($p));
  my ($u,$s,$v, $err0,$err);

  ##-- svd
  ($u,$s,$v) = svd($a);
  $err0 = ($a-($u * $s)->matmult($v->xchg(0,1)))**2;
  $err  = svderror_dbg($u,$s,$v, $ptr,$rowids,$nzvals);
}
#tsvderror_dbg(); exit 0;


##---------------------------------------------------------------------
## DUMMY
##---------------------------------------------------------------------
foreach $i (0..3) {
  print "--dummy($i)--\n";
}

