    /*************************************************************************************

    Grid physics library, www.github.com/paboyle/Grid 

    Source file: ./tests/Test_wilson_cg_unprec.cc

    Copyright (C) 2015

Author: Azusa Yamaguchi <ayamaguc@staffmail.ed.ac.uk>
Author: Peter Boyle <paboyle@ph.ed.ac.uk>

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

    See the full license in the file "LICENSE" in the top level distribution directory
    *************************************************************************************/
    /*  END LEGAL */
#include <Grid/Grid.h>

//BJ: Settings variables
int bj_asynch_setting = 0;
int bj_max_iter_diff = 0;
int bj_restart_length = 0;
int bj_synchronous_restarts = 0;

//BJ: Working variables
std::vector<std::vector<std::vector<Grid::CommsRequest_t>>> bj_reqs;
int bj_asynch = 0;
int bj_iteration = 0;
int bj_call_count = 0;

using namespace std;
using namespace Grid;
 ;

template<class d>
struct scal {
  d internal;
};

  Gamma::Algebra Gmu [] = {
    Gamma::Algebra::GammaX,
    Gamma::Algebra::GammaY,
    Gamma::Algebra::GammaZ,
    Gamma::Algebra::GammaT
  };

int main (int argc, char ** argv) {
	
  Grid_init(&argc,&argv);

  //BJ: Read in parameters from file
  std::ifstream fin;
  fin.open("settings.txt");
  std::string param_name;
  int param_value;
  
  fin >> param_name >> param_value;
  bj_asynch_setting = param_value;
  if (bj_asynch_setting == 1) {bj_asynch = 1;}
  std::cout << "BJ settings: " << param_name << " " << param_value << "\n";
  fin >> param_name >> param_value;
  bj_max_iter_diff = param_value;
  std::cout << "BJ settings: " << param_name << " " << param_value << "\n";
  fin >> param_name >> param_value; //restart_length - unused in this test
  fin >> param_name >> param_value;
  int max_iterations = param_value;
  std::cout << "BJ settings: " << param_name << " " << param_value << "\n";
  fin >> param_name >> param_value; //synchronous_restarts - unused in this test

  fin.close();

  Coordinate latt_size   = GridDefaultLatt();
  Coordinate simd_layout = GridDefaultSimd(Nd,vComplex::Nsimd());
  Coordinate mpi_layout  = GridDefaultMpi();
  GridCartesian               Grid(latt_size,simd_layout,mpi_layout);
  GridRedBlackCartesian     RBGrid(&Grid);

  std::vector<int> seeds({1,2,3,4});
  GridParallelRNG          pRNG(&Grid);  pRNG.SeedFixedIntegers(seeds);

  LatticeFermion src(&Grid); random(pRNG,src);
  RealD nrm = norm2(src);
  LatticeFermion result(&Grid); result=Zero();
  LatticeGaugeField Umu(&Grid); SU<Nc>::HotConfiguration(pRNG,Umu);

  double volume=1;
  for(int mu=0;mu<Nd;mu++){
    volume=volume*latt_size[mu];
  }  
  
  RealD mass=0.5;
  WilsonFermionR Dw(Umu,Grid,RBGrid,mass);

  MdagMLinearOperator<WilsonFermionR,LatticeFermion> HermOp(Dw);
  ConjugateGradient<LatticeFermion> CG(1.0e-8,10000);
  CG(HermOp,src,result);

  Grid_finalize();
}
