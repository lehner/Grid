/* 
 * Test_fatLinks.cc                                                               
 * 
 * D. Clarke 
 * 
 * Test the various constructs used to make fat links. 
 * 
 */

#include <Grid/Grid.h>
#include <Grid/lattice/PaddedCell.h>
#include <Grid/stencil/GeneralLocalStencil.h>
using namespace Grid;

// This is to optimize the SIMD
template<class vobj> void gpermute(vobj & inout,int perm) {
    vobj tmp=inout;
    if (perm & 0x1) {permute(inout,tmp,0); tmp=inout;}
    if (perm & 0x2) {permute(inout,tmp,1); tmp=inout;}
    if (perm & 0x4) {permute(inout,tmp,2); tmp=inout;}
    if (perm & 0x8) {permute(inout,tmp,3); tmp=inout;}
}

// Make the logger work like Python print()
template<typename ... Args>
inline std::string sjoin(Args&&... args) noexcept {
    std::ostringstream msg;
    (msg << ... << args);
    return msg.str();
}
template <typename ... Args>
inline void Grid_log(Args&&... args) {
    std::string msg = sjoin(std::forward<Args>(args)...);
    std::cout << GridLogMessage << msg << std::endl;
}

struct fatParams: Serializable {
    GRID_SERIALIZABLE_CLASS_MEMBERS(
        fatParams,
        std::string, conf_in,
        std::string, conf_out);

    template <class ReaderClass>
    fatParams(Reader<ReaderClass>& Reader){
        read(Reader, "parameters", *this);
    }
};



int main (int argc, char **argv)
{
    Grid_init(&argc,&argv);

    Coordinate latt_size   = GridDefaultLatt();
    Coordinate simd_layout = GridDefaultSimd(Nd,vComplexD::Nsimd());
    Coordinate mpi_layout  = GridDefaultMpi();

    Grid_log("mpi = ",mpi_layout);
    Grid_log("simd = ",simd_layout);
    Grid_log("latt = ",latt_size);

    GridCartesian GRID(latt_size,simd_layout,mpi_layout);

    XmlReader Reader("fatParams.xml",false, "grid");
    fatParams param(Reader);  

    LatticeGaugeField Umu(&GRID);
    FieldMetaData header;
    NerscIO::readConfiguration(Umu, header, param.conf_in);

    // Create a padded cell of extra padding depth=1
    int depth = 1;
    PaddedCell Ghost(depth,&GRID);
    LatticeGaugeField Ughost = Ghost.Exchange(Umu);

    // Array for <tr U_mu_nu>(x)
    GridBase *GhostGrid = Ughost.Grid();
    LatticeComplex gplaq(GhostGrid); 

    // This is where the 3-link constructs will be stored
    LatticeGaugeField Ughost_3link(Ughost.Grid());

    // Create 3-link stencil
    std::vector<Coordinate> shifts;
    for(int mu=0;mu<Nd;mu++){
        for(int nu=mu+1;nu<Nd;nu++){
            Coordinate shift_0(Nd,0);
            Coordinate shift_mu(Nd,0); shift_mu[mu]=1;
            Coordinate shift_nu(Nd,0); shift_nu[nu]=1;
            // push_back creates an element at the end of shifts and
            // assigns the data in the argument to it.
            shifts.push_back(shift_0);
            shifts.push_back(shift_mu);
            shifts.push_back(shift_nu);
            shifts.push_back(shift_0);
        }
    }
    GeneralLocalStencil gStencil(GhostGrid,shifts);

    Ughost_3link=Zero();

    // Create the accessors, here U_v and U_3link_v 
    autoView(U_v      , Ughost      , CpuRead);
    autoView(U_3link_v, Ughost_3link, CpuWrite);

    // This is a loop over local sites. 
    for(int ss=0;ss<U_v.size();ss++){

        // This is the stencil index. It increases as we make our way through the spacetime sites,
        // plaquette orientations, and as we travel around a plaquette.
        int s=0;
    
        for(int mu=0;mu<Nd;mu++){
            for(int nu=mu+1;nu<Nd;nu++){

                auto SE0 = gStencil.GetEntry(s+0,ss);
                auto SE1 = gStencil.GetEntry(s+1,ss);
                auto SE2 = gStencil.GetEntry(s+2,ss);
                auto SE3 = gStencil.GetEntry(s+3,ss);
    
                // Each offset corresponds to a site around the plaquette.
                int o0 = SE0->_offset;
                int o1 = SE1->_offset;
                int o2 = SE2->_offset;
                int o3 = SE3->_offset;
                
                auto U0 = U_v[o0](mu);
                auto U1 = U_v[o1](nu);
                auto U2 = adj(U_v[o2](mu));
                auto U3 = adj(U_v[o3](nu));
    
                gpermute(U0,SE0->_permute);
                gpermute(U1,SE1->_permute);
                gpermute(U2,SE2->_permute);
                gpermute(U3,SE3->_permute);

                auto W = U1*U2*U3;

                // We add together contributions coming from each orientation.                
                U_3link_v[ss](mu) = U_3link_v[ss](mu) + W;

                s=s+4;
            }
        }
    }

    // Here is my understanding of this part: The padded cell has its own periodic BCs, so
    // if I take a step to the right at the right-most side of the cell, I end up on the
    // left-most side. This means that the plaquettes in the padding are wrong. Luckily
    // all we care about are the plaquettes in the cell, which we obtain from Extract.
    Umu = Ghost.Extract(Ughost_3link);

    NerscIO::writeConfiguration(Umu,param.conf_out,"HISQ");

    Grid_finalize();
}