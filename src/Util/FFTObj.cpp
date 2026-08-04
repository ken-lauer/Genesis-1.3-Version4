#include <iostream>
#include <complex>
#include <map>
#include <mpi.h>

#ifdef FFTW
#include <fftw3.h>
#endif

#include "FFTObj.h"

using namespace std;
#ifdef FFTW

namespace {

// FFTW is fast for sizes of the form 2^a 3^b 5^c 7^d 11^e 13^f with
// e+f <= 1; any larger prime factor forces a much slower generic algorithm
bool fftwFriendly(int n)
{
	for (int p : {2, 3, 5, 7}) {
		while (n % p == 0) {
			n /= p;
		}
	}
	for (int p : {11, 13}) {
		if (n % p == 0) {
			n /= p;
			break;
		}
	}
	return n == 1;
}

int nearestFriendly(int n, int step)
{
	while (n > 2 && !fftwFriendly(n)) {
		n += step;
	}
	return n;
}

} // namespace

FFTObj::FFTObj(int ngrid)
{
	MPI_Comm_rank(MPI_COMM_WORLD, &rank_);

	// this is severe error: print on *all* ranks
	if(ngrid<0) {
		cerr << "Error: enforcing ngrid>0" << endl;
		ngrid=1;
	}

	if ((rank_ == 0) && (ngrid > 7) && !fftwFriendly(ngrid)) {
		cout << "Warning: ngrid=" << ngrid << " has a large prime factor - FFT-based field diagnostics" << endl
		     << "         will be several times slower. Nearest efficient sizes: "
		     << nearestFriendly(ngrid - 2, -2) << ", " << nearestFriendly(ngrid + 2, 2) << "." << endl
		     << "         (exclude_fft_output=true disables the FFT diagnostics entirely)" << endl;
	}
	in_ = new complex<double>[ngrid * ngrid];
	out_ = new complex<double>[ngrid * ngrid];
	p_ = fftw_plan_dft_2d(ngrid, ngrid,
	    reinterpret_cast<fftw_complex *>(in_),
	    reinterpret_cast<fftw_complex *>(out_),
	    FFTW_FORWARD, FFTW_ESTIMATE /* FFTW_MEASURE */);
	ngrid_ = ngrid;
}

FFTObj::~FFTObj() {
	if(rank_==0) {
		cout << "~FFTObj" << endl;
	}

	fftw_destroy_plan(p_);
	delete [] in_;
	delete [] out_;
}
#endif
