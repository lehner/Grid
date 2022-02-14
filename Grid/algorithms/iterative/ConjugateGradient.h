/*************************************************************************************

Grid physics library, www.github.com/paboyle/Grid

Source file: ./lib/algorithms/iterative/ConjugateGradient.h

Copyright (C) 2015

Author: Azusa Yamaguchi <ayamaguc@staffmail.ed.ac.uk>
Author: Peter Boyle <paboyle@ph.ed.ac.uk>
Author: paboyle <paboyle@ph.ed.ac.uk>

This program is free software; you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation; either version 2 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License along
with this program; if not, write to the Free Software Foundation, Inc.,
51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.

See the full license in the file "LICENSE" in the top level distribution
directory
*************************************************************************************/
			   /*  END LEGAL */
#ifndef GRID_CONJUGATE_GRADIENT_H
#define GRID_CONJUGATE_GRADIENT_H

//BJ: Settings variables
extern int bj_asynch_setting;
extern int bj_max_iter_diff;
extern int bj_restart_length;
extern int bj_synchronous_restarts;
extern int bj_me;

//BJ: Working variables
extern std::vector<std::vector<std::vector<Grid::CommsRequest_t>>> bj_reqs;
extern int bj_asynch;
extern int bj_iteration;
extern int bj_startsend_calls;
extern int bj_completesend_calls;
extern int bj_old_comms;

NAMESPACE_BEGIN(Grid);

/////////////////////////////////////////////////////////////
// Base classes for iterative processes based on operators
// single input vec, single output vec.
/////////////////////////////////////////////////////////////

template <class Field>
class ConjugateGradient : public OperatorFunction<Field> {
public:

  using OperatorFunction<Field>::operator();

  bool ErrorOnNoConverge;  // throw an assert when the CG fails to converge.
                           // Defaults true.
  RealD Tolerance;
  Integer MaxIterations;
  Integer IterationsToComplete; //Number of iterations the CG took to finish. Filled in upon completion
  RealD TrueResidual;
  
  ConjugateGradient(RealD tol, Integer maxit, bool err_on_no_conv = true)
    : Tolerance(tol),
      MaxIterations(maxit),
      ErrorOnNoConverge(err_on_no_conv){};

  void operator()(LinearOperatorBase<Field> &Linop, const Field &src, Field &psi) {

    psi.Checkerboard() = src.Checkerboard();

    conformable(psi, src);

    RealD cp, c, a, d, b, ssq, qq;
    //RealD b_pred;

    Field p(src);
    Field mmp(src);
    Field r(src);

    // Initial residual computation & set up
    RealD guess = norm2(psi);
    assert(std::isnan(guess) == 0);
    
    Linop.HermOpAndNorm(psi, mmp, d, b);
    
    r = src - mmp;
    p = r;

    a = norm2(p);
    cp = a;
    ssq = norm2(src);

    // Handle trivial case of zero src
    if (ssq == 0.){
      psi = Zero();
      IterationsToComplete = 1;
      TrueResidual = 0.;
      return;
    }

    std::cout << GridLogIterative << std::setprecision(8) << "ConjugateGradient: guess " << guess << std::endl;
    std::cout << GridLogIterative << std::setprecision(8) << "ConjugateGradient:   src " << ssq << std::endl;
    std::cout << GridLogIterative << std::setprecision(8) << "ConjugateGradient:    mp " << d << std::endl;
    std::cout << GridLogIterative << std::setprecision(8) << "ConjugateGradient:   mmp " << b << std::endl;
    std::cout << GridLogIterative << std::setprecision(8) << "ConjugateGradient:  cp,r " << cp << std::endl;
    std::cout << GridLogIterative << std::setprecision(8) << "ConjugateGradient:     p " << a << std::endl;

    RealD rsq = Tolerance * Tolerance * ssq;

    // Check if guess is really REALLY good :)
    if (cp <= rsq) {
      TrueResidual = std::sqrt(a/ssq);
      std::cout << GridLogMessage << "ConjugateGradient guess is converged already " << std::endl;
      IterationsToComplete = 0;	
      return;
    }

    std::cout << GridLogIterative << std::setprecision(8)
              << "ConjugateGradient: k=0 residual " << cp << " target " << rsq << std::endl;

    GridStopWatch LinalgTimer;
    GridStopWatch InnerTimer;
    GridStopWatch AxpyNormTimer;
    GridStopWatch LinearCombTimer;
    GridStopWatch MatrixTimer;
    GridStopWatch SolverTimer;

    SolverTimer.Start();
    int k;
    for (k = 1; k <= MaxIterations; k++) {
	  bj_iteration = k;
      c = cp;

      MatrixTimer.Start();
      Linop.HermOp(p, mmp);
      MatrixTimer.Stop();

      LinalgTimer.Start();

      InnerTimer.Start();
      ComplexD dc  = innerProduct(p,mmp);
      InnerTimer.Stop();
      d = dc.real();
      a = c / d;

      AxpyNormTimer.Start();
      cp = axpy_norm(r, -a, mmp, r);
      AxpyNormTimer.Stop();
      b = cp / c;

      LinearCombTimer.Start();
      {
	autoView( psi_v , psi, AcceleratorWrite);
	autoView( p_v   , p,   AcceleratorWrite);
	autoView( r_v   , r,   AcceleratorWrite);
	accelerator_for(ss,p_v.size(), Field::vector_object::Nsimd(),{
	    coalescedWrite(psi_v[ss], a      *  p_v(ss) + psi_v(ss));
	    coalescedWrite(p_v[ss]  , b      *  p_v(ss) + r_v  (ss));
	});
      }
      LinearCombTimer.Stop();
      LinalgTimer.Stop();

      std::cout << GridLogIterative << "ConjugateGradient: Iteration " << k
                << " residual " << sqrt(cp/ssq) << " target " << Tolerance << std::endl;

	  FILE * fp = fopen("residualcg.txt", "a");
	  fprintf(fp, "%d;%f;%f\n", k, cp, sqrt(cp/ssq));
	  fclose(fp);

      // Stopping condition
      if (cp <= rsq) {
        SolverTimer.Stop();
        Linop.HermOpAndNorm(psi, mmp, d, qq);
        p = mmp - src;

        RealD srcnorm = std::sqrt(norm2(src));
        RealD resnorm = std::sqrt(norm2(p));
        RealD true_residual = resnorm / srcnorm;

        std::cout << GridLogMessage << "ConjugateGradient Converged on iteration " << k 
		  << "\tComputed residual " << std::sqrt(cp / ssq)
		  << "\tTrue residual " << true_residual
		  << "\tTarget " << Tolerance << std::endl;

        std::cout << GridLogIterative << "Time breakdown "<<std::endl;
	std::cout << GridLogIterative << "\tElapsed    " << SolverTimer.Elapsed() <<std::endl;
	std::cout << GridLogIterative << "\tMatrix     " << MatrixTimer.Elapsed() <<std::endl;
	std::cout << GridLogIterative << "\tLinalg     " << LinalgTimer.Elapsed() <<std::endl;
	std::cout << GridLogIterative << "\tInner      " << InnerTimer.Elapsed() <<std::endl;
	std::cout << GridLogIterative << "\tAxpyNorm   " << AxpyNormTimer.Elapsed() <<std::endl;
	std::cout << GridLogIterative << "\tLinearComb " << LinearCombTimer.Elapsed() <<std::endl;

        if (ErrorOnNoConverge) assert(true_residual / Tolerance < 10000.0);

	IterationsToComplete = k;	
	TrueResidual = true_residual;

        return;
      }
    }
    // Failed. Calculate true residual before giving up                                                         
    Linop.HermOpAndNorm(psi, mmp, d, qq);
    p = mmp - src;

    TrueResidual = sqrt(norm2(p)/ssq);

    std::cout << GridLogMessage << "ConjugateGradient did NOT converge "<<k<<" / "<< MaxIterations<< std::endl;

    if (ErrorOnNoConverge) assert(0);
    IterationsToComplete = k;

  }
};
NAMESPACE_END(Grid);
#endif
