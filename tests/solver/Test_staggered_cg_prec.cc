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
#include <Grid/algorithms/iterative/BlockConjugateGradient.h>

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

int main (int argc, char ** argv)
{
  typedef typename ImprovedStaggeredFermionD::FermionField FermionField; 
  typename ImprovedStaggeredFermionD::ImplParams params; 

  Grid_init(&argc,&argv);

  //std::vector<Complex> boundary_phases(Nd,1.);
  //boundary_phases[Nd-1]=-1.;
  //params.boundary_phases = boundary_phases;

  Coordinate latt_size   = GridDefaultLatt();
  Coordinate simd_layout = GridDefaultSimd(Nd,vComplex::Nsimd());
  Coordinate mpi_layout  = GridDefaultMpi();
  GridCartesian            Grid(latt_size,simd_layout,mpi_layout);
  GridRedBlackCartesian    RBGrid(&Grid);

  std::vector<int> seeds({1,2,3,4});
  GridParallelRNG          pRNG(&Grid);  pRNG.SeedFixedIntegers(seeds);

  FermionField src(&Grid); //random(pRNG,src);
  src = Zero();
  ColourVector ColourKronecker;
  ColourKronecker = Zero();
  ColourKronecker()()(1) = 1.0;
  Coordinate site({0,0,0,0}); // Point source at origin
  pokeSite(ColourKronecker,src,site);
  RealD nrm = norm2(src);
  std::cout<<GridLogMessage << "src norm sq   =   "<< nrm <<std::endl;
  
  LatticeGaugeField Umu(&Grid);
  LatticeGaugeField UUUmu(&Grid);

  FieldMetaData header;
  std::string file("./l4444.fat.ildg");
  IldgReader _IldgReader;
  _IldgReader.open(file);
  _IldgReader.readConfiguration(Umu,header);
  _IldgReader.close();
  RealD plaq = ColourWilsonLoops::avgPlaquette(Umu);
  std::cout<<GridLogMessage<<" PLAQUETTE "<<plaq<<std::endl;

  std::string file_long("./l4444.long.ildg");
  _IldgReader.open(file_long);
  _IldgReader.readConfiguration(UUUmu,header);
  _IldgReader.close();
  plaq = ColourWilsonLoops::avgPlaquette(UUUmu);
  std::cout<<GridLogMessage<<" LONG PLAQUETTE "<<plaq<<std::endl;

  double volume=1;
  for(int mu=0;mu<Nd;mu++){
    volume=volume*latt_size[mu];
  }  
  
  RealD mass=0.1;
  //RealD c1=9.0/8.0;
  //RealD c2=-1.0/24.0;
  //RealD u0=1.0;
  RealD c1=1.0;
  RealD c2=1.0;
  RealD u0=1.0;
  ImprovedStaggeredFermionD Ds(Umu,UUUmu,Grid,RBGrid,mass,c1,c2,u0);

  FermionField res_o(&RBGrid); 
  FermionField src_o(&RBGrid); 
  pickCheckerboard(Even,src_o,src);
  res_o=Zero();

  SchurStaggeredOperator<ImprovedStaggeredFermionD,FermionField> HermOpEO(Ds);
  ConjugateGradient<FermionField> CG(1.0e-8,10000);
  double t1=usecond();
  CG(HermOpEO,src_o,res_o);
  double t2=usecond();

  // Schur solver: uses DeoDoe => volume * 1146
  double ncall=CG.IterationsToComplete;
  double flops=(16*(3*(6+8+8)) + 15*3*2)*volume*ncall; // == 66*16 +  == 1146

  std::cout<<GridLogMessage << "iters    =   "<< ncall <<std::endl;
  std::cout<<GridLogMessage << "usec    =   "<< (t2-t1)<<std::endl;
  std::cout<<GridLogMessage << "flops   =   "<< flops<<std::endl;
  std::cout<<GridLogMessage << "mflop/s =   "<< flops/(t2-t1)<<std::endl;


  FermionField tmp(&RBGrid);

  std::cout << "norm sq sol " << norm2(res_o) << "\n";
  HermOpEO.Mpc(res_o,tmp);
  std::cout << "check Mpc resid " << axpy_norm(tmp,-1.0,src_o,tmp)/norm2(src_o) << "\n";

  src=Zero();
  pokeSite(ColourKronecker,src,site);
  FermionField out(&Grid);
  //Ds.Dhop(src,out,0);
  Ds.M(src,out);
  nrm = norm2(out);
  std::cout<<GridLogMessage << "Dhop * src "<< std::endl;
  std::cout<<GridLogMessage << out << std::endl;
  
  Grid_finalize();
}
