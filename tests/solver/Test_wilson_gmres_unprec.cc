/*************************************************************************************

Grid physics library, www.github.com/paboyle/Grid

Source file: ./tests/solver/Test_wilson_gmres_unprec.cc

Copyright (C) 2015-2018

Author: Daniel Richtmann <daniel.richtmann@ur.de>

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
#include <Grid/Grid.h>

//BJ: Settings variables
int bj_asynch_setting = 0;
int bj_max_iter_diff = 0;
int bj_restart_length = 0;
int bj_synchronous_restarts = 0;
int bj_me = -1;

//BJ: Working variables
std::vector<std::vector<std::vector<Grid::CommsRequest_t>>> bj_reqs;
int bj_asynch = 0;
int bj_iteration = 0;
int bj_startsend_calls = 0;
int bj_completesend_calls = 0;
int bj_old_comms = 0;

using namespace Grid;

int main (int argc, char ** argv) {
	
  Grid_init(&argc,&argv);

  //BJ: Read in parameters from file
  std::ifstream fin;
  fin.open("settings.txt");
  std::string param_name;
  int param_value;
  
  MPI_Comm_rank(MPI_COMM_WORLD, &bj_me);
  fin >> param_name >> param_value;
  bj_asynch_setting = param_value;
  if (bj_asynch_setting == 1) {bj_asynch = 1;}
  std::cout << "BJ settings: " << param_name << " " << param_value << "\n";
  fin >> param_name >> param_value;
  bj_max_iter_diff = param_value;
  std::cout << "BJ settings: " << param_name << " " << param_value << "\n";
  fin >> param_name >> param_value;
  bj_restart_length = param_value;
  std::cout << "BJ settings: " << param_name << " " << param_value << "\n";
  fin >> param_name >> param_value;
  int max_iterations = param_value;
  std::cout << "BJ settings: " << param_name << " " << param_value << "\n";
  fin >> param_name >> param_value;
  bj_synchronous_restarts = param_value;
  std::cout << "BJ settings: " << param_name << " " << param_value << "\n";
  fin >> param_name >> param_value;
  int bj_save_result = param_value;
  std::cout << "BJ settings: " << param_name << " " << param_value << "\n";

  fin.close();
  
  Coordinate latt_size   = GridDefaultLatt();
  Coordinate simd_layout = GridDefaultSimd(Nd,vComplex::Nsimd());
  Coordinate mpi_layout  = GridDefaultMpi();
  GridCartesian Grid(latt_size,simd_layout,mpi_layout);
  GridRedBlackCartesian RBGrid(&Grid);

  std::vector<int> seeds({1,2,3,4});
  GridParallelRNG pRNG(&Grid);
  pRNG.SeedFixedIntegers(seeds);

  LatticeFermion src(&Grid);
  random(pRNG,src);
  
  RealD nrm = norm2(src);
  
  LatticeFermion result(&Grid);
  result=Zero();
  
  LatticeGaugeField Umu(&Grid);
  SU<Nc>::HotConfiguration(pRNG,Umu);

  double volume = 1;
  for(int mu=0;mu<Nd;mu++){
    volume=volume*latt_size[mu];
  }

  RealD mass = 0.5;
  WilsonFermionR Dw(Umu,Grid,RBGrid,mass);

  MdagMLinearOperator<WilsonFermionR,LatticeFermion> HermOp(Dw);
  //NonHermitianLinearOperator<WilsonFermionR,LatticeFermion> HermOp(Dw); //same behavior as standard
  GeneralisedMinimalResidual<LatticeFermion> GMRES(1.0e-8, max_iterations, bj_restart_length);
  GMRES(HermOp,src,result);

  //How many comms that were out of synch
  printf("Old Cooms: %d\n", bj_old_comms);
  
  if (bj_save_result) {
	//Write out result
	std::string file1("./Propagator1");
	emptyUserRecord record;
	uint32_t nersc_csum;
	uint32_t scidac_csuma;
	uint32_t scidac_csumb;
	typedef SpinColourVectorD   FermionD;
	typedef vSpinColourVectorD vFermionD;

	BinarySimpleMunger<FermionD,FermionD> munge;
	std::string format = getFormatString<vFermionD>();
	  
	BinaryIO::writeLatticeObject<vFermionD,FermionD>(result, file1, munge, 0, format, nersc_csum, scidac_csuma, scidac_csumb);
	std::cout << GridLogMessage << " CG checksums "<<std::hex << scidac_csuma << " "<<scidac_csumb<<std::endl;
  }

  Grid_finalize();
  
}
