! COLLIER 1.2.10 scalar masters for benchmarks/onia: the arbiter when TIRust
! and quad original IREGI disagree on an IR-finite integral (run.py).
! Reads lines
!   C0 p10 p21 p20 m02 m12 m22
!   D0 p10 p21 p32 p30 p20 p31 m02 m12 m22 m32
! (pij = (q_i - q_j)^2, squared masses) and prints the finite part in
! COLLIER's (= OneLOop's) normalization at mu = 3 GeV, followed by the
! single and double IR pole coefficients (COLLIER's DeltaIR parameters set to
! zero; only IR-finite integrals are compared by run.py).
program collier_scalar
  use collier
  implicit none
  character(512) :: line
  character(4) :: kind
  double precision :: v(10)
  double complex :: r, r1, r2
  integer :: ios
  call Init_cll(4, 0, '')
  call SetMuUV2_cll(9d0)
  call SetMuIR2_cll(9d0)
  write(*,'(a)') 'READY COLLIER_SCALAR_V1'
  do
    read(*,'(a)',iostat=ios) line
    if (ios /= 0) exit
    if (trim(line) == 'QUIT') exit
    read(line,*) kind
    if (trim(kind) == 'C0') then
      read(line,*) kind, v(1:6)
      call SetDeltaIR_cll(0d0, 0d0)
      call C0_cll(r, dcmplx(v(1)), dcmplx(v(2)), dcmplx(v(3)), dcmplx(v(4)), dcmplx(v(5)), dcmplx(v(6)))
      call SetDeltaIR_cll(1d0, 0d0)
      call C0_cll(r1, dcmplx(v(1)), dcmplx(v(2)), dcmplx(v(3)), dcmplx(v(4)), dcmplx(v(5)), dcmplx(v(6)))
      call SetDeltaIR_cll(0d0, 1d0)
      call C0_cll(r2, dcmplx(v(1)), dcmplx(v(2)), dcmplx(v(3)), dcmplx(v(4)), dcmplx(v(5)), dcmplx(v(6)))
    else
      read(line,*) kind, v(1:10)
      call SetDeltaIR_cll(0d0, 0d0)
      call D0_cll(r, dcmplx(v(1)), dcmplx(v(2)), dcmplx(v(3)), dcmplx(v(4)), dcmplx(v(5)), dcmplx(v(6)), &
                  dcmplx(v(7)), dcmplx(v(8)), dcmplx(v(9)), dcmplx(v(10)))
      call SetDeltaIR_cll(1d0, 0d0)
      call D0_cll(r1, dcmplx(v(1)), dcmplx(v(2)), dcmplx(v(3)), dcmplx(v(4)), dcmplx(v(5)), dcmplx(v(6)), &
                  dcmplx(v(7)), dcmplx(v(8)), dcmplx(v(9)), dcmplx(v(10)))
      call SetDeltaIR_cll(0d0, 1d0)
      call D0_cll(r2, dcmplx(v(1)), dcmplx(v(2)), dcmplx(v(3)), dcmplx(v(4)), dcmplx(v(5)), dcmplx(v(6)), &
                  dcmplx(v(7)), dcmplx(v(8)), dcmplx(v(9)), dcmplx(v(10)))
    end if
    write(*,'(a,6es26.17)') 'SCALAR ', r, r1 - r, r2 - r
    flush(6)
  end do
end program collier_scalar
