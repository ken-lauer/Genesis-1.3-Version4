#include "FieldSolverFFT.h"
#include "Field.h"
#include "Beam.h"



FieldSolverFFT::~FieldSolverFFT() = default;

void FieldSolverFFT::advance(double delz, Field *field, Beam *beam, Undulator *und) {

    for (unsigned long ii = 0; ii < field->field.size(); ii++) {  // ii is index for the beam

        field->constructSource(crsource, beam, und, delz, ii);

        unsigned long i = (ii + field->first) % field->field.size();           // index for the field

        // get the FFT representation of the radiation field and the source term


        this->FFT(field->field[i]);
    }
}


void FieldSolverFFT::FFT(vector<complex<double> > &crfield)
{
    // Transform the field into Fourier space
    for (unsigned long ii = 0; ii < crfield.size(); ii++) {
        in[ii] = crfield[ii];
    }
#ifdef FFTW
        fftw_execute(p);
#endif

    if (doFilter_) {
        // The source term is filtered in Fourier space, so it needs its own
        // forward transform.
        for (unsigned long ii = 0; ii < crfield.size(); ii++) {
            uf[ii] = out[ii];
            in[ii] = crsource[ii];
        }
#ifdef FFTW
        fftw_execute(p);
#endif
        for (unsigned long ii = 0; ii < crfield.size(); ii++) {
            sf[ii] = out[ii] * sigmoid_[ii];
        }
        // do the actual propagation
        for (unsigned long ii = 0; ii < crfield.size(); ii++) {
            in[ii] = uf[ii] * exp(K2[ii] * delz_save) + 2. * sf[ii];
        }
    } else {
        // Without the filter the source term is not modified in Fourier space.
        // The transform is linear, so IFFT(FFT(crsource))/ngrid^2 == crsource
        // and the source can simply be added in real space after the back
        // transform. This saves one of the three 2D FFTs per slice and step.
        for (unsigned long ii = 0; ii < crfield.size(); ii++) {
            in[ii] = out[ii] * exp(K2[ii] * delz_save);
        }
    }
#ifdef FFTW
        fftw_execute(ip);
#endif

    double norm = 1./static_cast<double>(ngrid*ngrid);
    if (doFilter_) {
        for (unsigned long ii = 0; ii < crfield.size(); ii++) {
            crfield[ii] = out[ii] * norm;
        }
    } else {
        for (unsigned long ii = 0; ii < crfield.size(); ii++) {
            crfield[ii] = out[ii] * norm + 2. * crsource[ii];
        }
    }
}


void FieldSolverFFT::init(double delz,double dgrid, double xks, unsigned int ngrid_in) {

    delz_save = delz;
    if (!hasPlan) {
        ks = xks;
        ngrid = ngrid_in;
        dk = 4.*asin(1.)/(static_cast<double>(ngrid)*dgrid);
        in = new complex<double> [ngrid*ngrid];
        out= new complex<double> [ngrid*ngrid];
        uf.resize(ngrid*ngrid);
        sf.resize(ngrid*ngrid);
        K2.resize(ngrid*ngrid);
        sigmoid_.resize(ngrid*ngrid);

        double shift=-0.5*static_cast<double> (ngrid-1);
        for (int iy=0;iy<ngrid;iy++) {
            double dy=static_cast<double>(iy)+shift;
            double y = dy / static_cast<double>(ngrid) /yc ;
            for (int ix=0;ix<ngrid;ix++) {
                double dx=static_cast<double>(ix)+shift;
                double x = dx / static_cast<double>(ngrid) /xc ;
                int iiy=(iy+(ngrid+1)/2) % ngrid;
                int iix=(ix+(ngrid+1)/2) % ngrid;
                int ii=iiy*ngrid+iix;
                K2[ii] = complex<double>(0,-(dx*dx+dy*dy)*dk*dk/2./xks);
                double r = (sqrt(x * x + y * y) - 1) / sig;
                sigmoid_[ii] = 1. / (1 + exp(r));
            }
        }
        crsource.resize(ngrid* ngrid);


#ifdef FFTW
        p  = fftw_plan_dft_2d(ngrid,ngrid,reinterpret_cast<fftw_complex*>(in),reinterpret_cast<fftw_complex*>(out),FFTW_FORWARD,FFTW_MEASURE);
        ip  = fftw_plan_dft_2d(ngrid,ngrid,reinterpret_cast<fftw_complex*>(in),reinterpret_cast<fftw_complex*>(out),FFTW_BACKWARD,FFTW_MEASURE);
#endif
        hasPlan = true;
    }
}

void FieldSolverFFT::initSourceFilter(double xc_in, double yc_in, double sig_in,bool do_filter) {
    xc=xc_in;
    yc=yc_in;
    sig=sig_in;
    doFilter_ = do_filter;
    // check for unphysical input. xc and yc are used as divisors when the
    // sigmoid is tabulated, sig is its width.
    if ((sig <= 0) || (xc <= 0) || (yc <= 0)) {
        doFilter_ = false;
    }
};