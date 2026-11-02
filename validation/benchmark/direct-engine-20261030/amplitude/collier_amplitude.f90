! Optional external-reference benchmark. Production iregi-rust remains Rust-only.
! The integrals of an amplitude (pinches of one N-gon) on a batch of phase-space
! points through COLLIER 1.2.10's public tensor calls, with or without its global
! cache. One amplitude call evaluates every integral at every point.
!
! usage: collier-amplitude SPEC_TXT CACHE(0|1)
! SPEC_TXT: "nint npoints nprop", then per integral "k rank m_1 .. m_k"
! (1-based propagators of the N-gon), then per point nprop rows
! "q0 q1 q2 q3 m2" and one row "mu" (positive real squared masses).
! Each integral is called in its own zero-origin routing: momenta q_{m_t}-q_{m_1}.
! CACHE=1: InitCacheSystem_cll(1, nprop) once, InitEvent_cll(1) before each
! point (COLLIER's documented use: the cache learns the call sequence of the
! first events and later returns results by position). CACHE=0: no global
! cache; InitEvent_cll before every tensor call, as the campaign's batch driver.
! Settings as the stability study: COLI (mode 1), TenRed 6, requested accuracy
! 1e-10, critical 1e-3, muUV2 = muIR2 = mu^2, delta parameters zero.
! stdin: BENCH warmup iterations -> RESULT ns per amplitude call
!        FINAL -> per (integral, point) the sum of all tensor coefficients
!                 and of their absolute values, and accuracy-flag counts
!        QUIT
program collier_amplitude
  use iso_fortran_env, only: int64, output_unit
  use, intrinsic :: ieee_arithmetic, only: ieee_is_finite
  use COLLIER
  implicit none
  character(4096) :: line, path, arg
  character(16) :: command
  integer :: ios, nint, npoints, nprop, maxk, maxrank, i, t, p, warmup, iterations, it, use_cache
  integer :: err_flag, acc_flag, flagged, errors
  integer, allocatable :: k(:), rank(:), members(:,:)
  real(kind(1d0)), allocatable :: shifts(:,:,:), m2(:,:), mu(:)
  integer(int64) :: start, finish, rate
  complex(kind(1d0)), allocatable :: tensor(:,:,:,:), tensor_uv(:,:,:,:), mom(:,:), invariants(:), masses(:)
  real(kind(1d0)), allocatable :: tensor_error(:)
  complex(kind(1d0)) :: total
  real(kind(1d0)) :: magnitude

  call get_command_argument(1, path)
  call get_command_argument(2, arg)
  read(arg,*) use_cache
  open(10, file=trim(path), status='old', action='read')
  read(10,*) nint, npoints, nprop
  allocate(k(nint), rank(nint), members(nprop, nint))
  do i=1,nint
    read(10,'(a)') line
    read(line,*) k(i), rank(i)
    read(line,*) k(i), rank(i), members(1:k(i), i)
  end do
  allocate(shifts(0:3, nprop, npoints), m2(nprop, npoints), mu(npoints))
  do p=1,npoints
    do t=1,nprop
      read(10,*) shifts(:, t, p), m2(t, p)
    end do
    read(10,*) mu(p)
  end do
  close(10)
  if (any(m2<=0d0)) error stop 'positive real squared masses required'
  maxk=maxval(k)
  maxrank=maxval(rank)
  allocate(tensor(0:maxrank,0:maxrank,0:maxrank,0:maxrank), tensor_uv(0:maxrank,0:maxrank,0:maxrank,0:maxrank))
  allocate(mom(0:3, maxk), invariants(maxk*(maxk-1)/2), masses(maxk), tensor_error(0:maxrank))

  call Init_cll(max(nprop,2), max(maxrank,2), '', .false.)
  call SwitchOffFileOutput_cll
  call SetInfOutLev_cll(0)
  call SetErrOutLev_cll(0)
  call SwitchOffErrStop_cll
  call SetMode_cll(1)
  call SetTenRed_cll(6)
  call InitCacheSystem_cll(use_cache, nprop)
  call SwitchOffArgPerm_cll
  call SetReqAcc_cll(1d-10)
  call SetCritAcc_cll(1d-3)
  call SetMuUV2_cll(mu(1)*mu(1))
  call SetMuIR2_cll(mu(1)*mu(1))
  call SetDeltaUV_cll(0d0)
  call SetDeltaIR_cll(0d0,0d0)
  if (any(mu/=mu(1))) error stop 'one mu for all points'
  call system_clock(count_rate=rate)
  write(output_unit,'(a)') 'READY COLLIER_AMPLITUDE_V1'
  flush(output_unit)
  do
    read(*,'(a)',iostat=ios) line
    if (ios/=0) exit
    if (len_trim(line)==0) cycle
    read(line,*) command
    select case (trim(command))
    case ('QUIT')
      exit
    case ('BENCH')
      read(line,*) command, warmup, iterations
      do it=1,warmup
        call amplitude(.false.)
      end do
      call system_clock(start)
      do it=1,iterations
        call amplitude(.false.)
      end do
      call system_clock(finish)
      write(output_unit,'(a,es25.17e3,a,i0,a,i0,a)') 'RESULT {"ns_per_call":', &
        real(finish-start,kind(1d0))*1d9/real(rate,kind(1d0))/iterations, ',"cache":', use_cache, &
        ',"iterations":', iterations, '}'
      flush(output_unit)
    case ('FINAL')
      flagged=0
      errors=0
      write(output_unit,'(a,i0,a)',advance='no') 'FINAL {"cache":', use_cache, ',"sums":['
      call amplitude(.true.)
      write(output_unit,'(a,i0,a,i0,a)') '],"accuracy_flagged":', flagged, ',"error_flagged":', errors, '}'
      flush(output_unit)
    case default
      error stop 'invalid request'
    end select
  end do

contains
  real(kind(1d0)) function square(v)
    real(kind(1d0)), intent(in) :: v(0:3)
    square=v(0)*v(0)-v(1)*v(1)-v(2)*v(2)-v(3)*v(3)
  end function square

  subroutine amplitude(report)
    logical, intent(in) :: report
    integer :: ip, ii, n, tt, distance, start_index, end_index, series_length, position, a0, a1, a2, a3
    logical :: first
    first=.true.
    do ip=1,npoints
      if (use_cache==1) call InitEvent_cll(1)
      do ii=1,nint
        n=k(ii)
        do tt=1,n-1
          mom(:,tt)=cmplx(shifts(:,members(tt+1,ii),ip)-shifts(:,members(1,ii),ip),0d0,kind(1d0))
        end do
        position=0
        do distance=1,n/2
          series_length=n
          if (2*distance==n) series_length=n/2
          do start_index=0,series_length-1
            end_index=mod(start_index+distance,n)
            position=position+1
            invariants(position)=cmplx(square(shifts(:,members(end_index+1,ii),ip) &
              -shifts(:,members(start_index+1,ii),ip)),0d0,kind(1d0))
          end do
        end do
        do tt=1,n
          masses(tt)=cmplx(m2(members(tt,ii),ip),0d0,kind(1d0))
        end do
        if (use_cache==0) call InitEvent_cll
        select case(n)
        case(2)
          call Bten_cll(tensor,tensor_uv,mom(:,1:1),invariants(1:1),masses(1:2),rank(ii),tensor_error)
        case(3)
          call Cten_cll(tensor,tensor_uv,mom(:,1:2),invariants(1:3),masses(1:3),rank(ii),tensor_error)
        case(4)
          call Dten_cll(tensor,tensor_uv,mom(:,1:3),invariants(1:6),masses(1:4),rank(ii),tensor_error)
        case default
          call TNten_cll(tensor,tensor_uv,mom(:,1:n-1),invariants(1:n*(n-1)/2),masses(1:n),n,rank(ii),tensor_error)
        end select
        if (report) then
          call GetAccFlag_cll(acc_flag)
          call GetErrFlag_cll(err_flag)
          if (acc_flag<0) flagged=flagged+1
          if (err_flag<0) errors=errors+1
          total=(0d0,0d0)
          magnitude=0d0
          do a0=0,rank(ii)
            do a1=0,rank(ii)-a0
              do a2=0,rank(ii)-a0-a1
                do a3=0,rank(ii)-a0-a1-a2
                  total=total+tensor(a0,a1,a2,a3)+tensor_uv(a0,a1,a2,a3)
                  magnitude=magnitude+abs(tensor(a0,a1,a2,a3))+abs(tensor_uv(a0,a1,a2,a3))
                end do
              end do
            end do
          end do
          if (.not.first) write(output_unit,'(a)',advance='no') ','
          first=.false.
          write(output_unit,'(a,es25.17e3,a,es25.17e3,a,es25.17e3,a)',advance='no') &
            '[',real(total),',',aimag(total),',',magnitude,']'
        end if
      end do
    end do
  end subroutine amplitude
end program collier_amplitude
