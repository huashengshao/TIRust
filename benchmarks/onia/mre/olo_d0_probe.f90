! IREGI's bundled OneLOop (avh_olo_forIREGI), scalar D0 called directly:
! reads p1 p2 p3 p4 p12 p23 m1 m2 m3 m4 (squared masses), prints rslt(0:2)
! (finite, 1/eps, 1/eps^2) in OneLOop's normalization at mu = 3.
program olo_d0_probe
  use avh_olo_forIREGI, only: olo_d0, olo_scale
  implicit none
  double precision :: v(10)
  double complex :: rslt(0:2)
  integer :: ios
  call olo_scale(3d0)
  do
    read(*,*,iostat=ios) v
    if (ios /= 0) exit
    call olo_d0(rslt, v(1), v(2), v(3), v(4), v(5), v(6), v(7), v(8), v(9), v(10))
    write(*,'(a,6es25.16)') 'D0 ', rslt(0), rslt(1), rslt(2)
  end do
end program olo_d0_probe
