! Optional external-reference benchmark. Production iregi-rust remains Rust-only.
! Opening-table external reference: ordinary unit-power N=1..6, rank<=6.
! Source derivative of opening-collier-20261026; public options are wire inputs.
! Init capacities protect upstream low-N/rank-zero setup; requests stay exact.
! Native diagnostics and finite scans are outside the timer; gathering stays inside.
! Per-block snapshots and UV/IR pole probes remain outside the timer.
! Zero-origin real shifts and positive real squared masses. A positive
! Feynman-polynomial bound is required. CHECK varies all pole parameters.
! Public tensor calls, invariant setup, normalization and gathering are timed.
program collier_opening_double_reference
  use iso_fortran_env, only: int64, output_unit
  use, intrinsic :: ieee_arithmetic, only: ieee_is_finite
  use COLLIER
  implicit none
  character(8192) :: line
  character(32) :: command
  character(128) :: request_id
  character(32) :: active_interface
  integer :: ios, mode, n, rank, batch, warmup, repeats, iterations, init_rank_capacity, init_topology_capacity
  integer :: i, j, lane, repeat, iteration, count, r, nt, nx, ny, nz, pos
  integer :: err_flag, acc_flag, bad_poles, tenred, argperm
  integer, allocatable :: last_err_flags(:),last_acc_flags(:),worst_err_flags(:),worst_acc_flags(:)
  integer(int64) :: start, finish, rate
  real(kind(1d0)) :: mu, pi, a, uv_error, norm, requested_accuracy, critical_accuracy
  real(kind(1d0)), allocatable :: shifts(:,:,:), times(:), tensor_error(:), pole_errors(:,:)
  real(kind(1d0)), allocatable :: maximum_rank_errors(:,:)
  complex(kind(1d0)), allocatable :: masses(:,:), result(:,:,:), mom(:,:), invariants(:)
  complex(kind(1d0)), allocatable :: tensor(:,:,:,:), tensor_uv(:,:,:,:)
  complex(kind(1d0)), allocatable :: baseline(:), baseline_uv(:)
  logical, allocatable :: stable(:)
  logical :: all_stable, check_poles
  complex(kind(1d0)), allocatable :: block_result(:,:,:,:)
  integer, allocatable :: block_err(:,:),block_acc(:,:)
  real(kind(1d0)), allocatable :: block_rank_errors(:,:,:)
  logical, allocatable :: block_stable(:,:)
  complex(kind(1d0)) :: ipi2

  pi=acos(-1d0)
  a=0.577215664901532860606512090082402431d0+log(pi)
  ipi2=cmplx(0d0,pi*pi,kind(1d0))
  call system_clock(count_rate=rate)
  if (rate<=0) error stop 'Monotonic wall clock unavailable'
  write(output_unit,'(a)') 'READY COLLIER_OPENING_DOUBLE_V1'
  flush(output_unit)
  do
    read(*,'(a)',iostat=ios) line
    if (ios<0) exit
    if (ios/=0) error stop 'Cannot read request'
    if (len_trim(line)==0) cycle
    if (trim(line)=='QUIT') exit
    read(line,*,iostat=ios) command,request_id,mode,n,rank,mu,batch,warmup,repeats,iterations, &
      requested_accuracy,critical_accuracy,tenred,argperm
    if (ios/=0.or.(trim(command)/='BENCH'.and.trim(command)/='CHECK')) error stop 'Invalid header'
    if (verify(trim(request_id),'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789_-.')/=0) &
      error stop 'Invalid request ID'
    if (n<1.or.n>6.or.rank<0.or.rank>6.or.mode<1.or.mode>2) error stop 'Unsupported topology'
    if (mode==2.and.n==5.and.rank>5) error stop 'DD supports pentagons only through rank 5'
    if (tenred<6.or.tenred>7.or.argperm<0.or.argperm>1) error stop 'Invalid public configuration'
    if (.not.ieee_is_finite(requested_accuracy).or..not.ieee_is_finite(critical_accuracy)) &
      error stop 'Nonfinite requested accuracy'
    if (requested_accuracy<=0d0.or.critical_accuracy<requested_accuracy.or.critical_accuracy>=1d0) &
      error stop 'Invalid requested accuracy'
    if (batch<1.or.batch>4096.or.warmup<0.or.repeats<1.or.iterations<1.or.mu<=0d0) &
      error stop 'Invalid run dimensions'
    if (.not.ieee_is_finite(mu).or..not.ieee_is_finite(mu*mu)) error stop 'Invalid scale'
    if (mu*mu<=0d0) error stop 'Squared scale underflow'
    count=(rank+1)*(rank+2)*(rank+3)*(rank+4)/24
    allocate(shifts(n,0:3,batch),masses(n,batch),result(0:count-1,4,batch),stable(batch),times(repeats))
    allocate(mom(0:3,n-1),invariants(n*(n-1)/2),tensor_error(0:rank))
    allocate(tensor(0:rank,0:rank,0:rank,0:rank),tensor_uv(0:rank,0:rank,0:rank,0:rank))
    allocate(baseline(0:count-1),baseline_uv(0:count-1),pole_errors(3,batch))
    allocate(last_err_flags(batch),last_acc_flags(batch),worst_err_flags(batch),worst_acc_flags(batch))
    allocate(maximum_rank_errors(0:rank,batch))
    allocate(block_result(0:count-1,4,batch,repeats),block_err(batch,repeats),block_acc(batch,repeats))
    allocate(block_stable(batch,repeats),block_rank_errors(0:rank,batch,repeats))
    last_err_flags=0
    last_acc_flags=0
    worst_err_flags=0
    worst_acc_flags=0
    maximum_rank_errors=0d0
    do lane=1,batch
      do i=1,n
        read(*,*,iostat=ios) shifts(i,:,lane),masses(i,lane)
        if (ios/=0) error stop 'Invalid propagator row'
      end do
      if (any(shifts(1,:,lane)/=0d0)) error stop 'Only zero-origin routing is supported'
      if (any(.not.ieee_is_finite(shifts(:,:,lane)))) error stop 'Nonfinite shift'
      if (any(.not.ieee_is_finite(real(masses(:,lane))))) error stop 'Nonfinite mass'
      if (any(.not.ieee_is_finite(aimag(masses(:,lane))))) error stop 'Nonfinite mass'
      if (any(real(masses(:,lane))<=0d0).or.any(aimag(masses(:,lane))/=0d0)) &
        error stop 'This slice requires positive real squared masses'
      call require_finite_mass_domain(lane)
    end do
    ! Init_cll enables local-only memoization, reset by each root tensor call.
    ! No global event cache is created. Both references reuse initialization.
    ! The generic API reads BinomTable(2,n), while Init_cll sizes this
    ! table to init_rank_capacity+max(n-2,4).  Reserve that index without
    ! changing any requested tensor rank or requested output allocation.
    ! Upstream SetIndCombisEq requires Nmax>=2; SetAddIndTab writes index1 even for rank0.
    init_topology_capacity=max(n,2)
    init_rank_capacity=max(rank,1)
    if (n>=5) init_rank_capacity=max(init_rank_capacity,n-max(n-2,4))
    call Init_cll(init_topology_capacity,init_rank_capacity,'',.false.)
    call SwitchOffFileOutput_cll
    call SetInfOutLev_cll(0)
    call SetErrOutLev_cll(0)
    call SwitchOffErrStop_cll
    call SetMode_cll(mode)
    call SetTenRed_cll(tenred)
    call InitCacheSystem_cll(0,n)
    call SwitchOffArgPerm_cll
    ! Public argument permutation disables cache-system use, including local caching.
    if (argperm==1) call SwitchOnArgPerm_cll
    call SetReqAcc_cll(requested_accuracy)
    call SetCritAcc_cll(critical_accuracy)
    call SetMuUV2_cll(mu*mu)
    call SetMuIR2_cll(mu*mu)
    call SetDeltaUV_cll(0d0)
    call SetDeltaIR_cll(0d0,0d0)
    check_poles=trim(command)=='CHECK'
    all_stable=.true.
    stable=.true.
    bad_poles=0
    pole_errors=0d0
    if (check_poles) call verify_poles()
    do iteration=1,warmup
      call evaluate_batch()
    end do
    do repeat=1,repeats
      call system_clock(start)
      do iteration=1,iterations
        call evaluate_batch()
      end do
      call system_clock(finish)
      times(repeat)=real(finish-start,kind(1d0))*1d9/real(rate,kind(1d0))/iterations
      ! All output/status validation is deliberately outside the kernel timer.
      do lane=1,batch
        call collect_diagnostics(lane)
        stable(lane)=stable(lane).and.all(ieee_is_finite(real(result(:,:,lane)))).and. &
                                    all(ieee_is_finite(aimag(result(:,:,lane))))
        all_stable=all_stable.and.stable(lane)
      end do
      block_result(:,:,:,repeat)=result
      block_err(:,repeat)=last_err_flags
      block_acc(:,repeat)=last_acc_flags
      block_stable(:,repeat)=stable
      block_rank_errors(:,:,repeat)=maximum_rank_errors
    end do
    write(output_unit,'(a)',advance='no') 'RESULT {"id":"'//trim(request_id)//'","stable":'
    if (all_stable.and.bad_poles==0) then
      write(output_unit,'(a)',advance='no') 'true'
    else
      write(output_unit,'(a)',advance='no') 'false'
    end if
    write(output_unit,'(a,i0,a,i0,a,i0,a,i0,a)',advance='no') &
      ',"collier_mode":',mode,',"clock_ticks_per_second":',rate, &
      ',"initialization_rank_capacity":',init_rank_capacity, &
      ',"initialization_topology_capacity":',init_topology_capacity, &
      ',"interface":"'//trim(active_interface)//'"'
    write(output_unit,'(a,i0,a,i0,a,es25.17e3,a,es25.17e3,a)',advance='no') &
      ',"direct_tensor_threshold":',tenred,',"argument_permutation":',argperm, &
      ',"requested_accuracy":',requested_accuracy,',"critical_accuracy":',critical_accuracy,',"total_ns":['
    do i=1,repeats
      if (i>1) write(output_unit,'(a)',advance='no') ','
      write(output_unit,'(es25.17e3)',advance='no') times(i)
    end do
    write(output_unit,'(a)',advance='no') '],"blocks":['
    do repeat=1,repeats
      if (repeat>1) write(output_unit,'(a)',advance='no') ','
      call write_block(repeat)
    end do
    write(output_unit,'(a)',advance='no') '],"pole_check_errors":'
    if (check_poles) then
      write(output_unit,'(a)',advance='no') '['
      do lane=1,batch
        if (lane>1) write(output_unit,'(a)',advance='no') ','
        write(output_unit,'(a,es25.17e3,a,es25.17e3,a,es25.17e3,a)',advance='no') &
          '[',pole_errors(1,lane),',',pole_errors(2,lane),',',pole_errors(3,lane),']'
      end do
      write(output_unit,'(a)',advance='no') ']'
    else
      write(output_unit,'(a)',advance='no') 'null'
    end if
    write(output_unit,'(a)',advance='no') ',"lanes":['
    do lane=1,batch
      if (lane>1) write(output_unit,'(a)',advance='no') ','
      if (.not.stable(lane)) then
        write(output_unit,'(a)',advance='no') '{"status":"unstable","moments":null'
        call write_diagnostics(lane)
        write(output_unit,'(a)',advance='no') ',"diagnostic_moments":'
        call write_moments(lane)
        write(output_unit,'(a)',advance='no') '}'
        cycle
      end if
      ! Successful output layout and values retain the original contract.
      write(output_unit,'(a)',advance='no') '{"status":"success","moments":['
      do i=0,count-1
        if (i>0) write(output_unit,'(a)',advance='no') ','
        write(output_unit,'(a)',advance='no') '['
        do j=1,4
          if (j>1) write(output_unit,'(a)',advance='no') ','
          write(output_unit,'(a,es25.17e3,a,es25.17e3,a)',advance='no') &
            '[',real(result(i,j,lane)),',',aimag(result(i,j,lane)),']'
        end do
        write(output_unit,'(a)',advance='no') ']'
      end do
      write(output_unit,'(a)',advance='no') ']'
      call write_diagnostics(lane)
      write(output_unit,'(a)',advance='no') '}'
    end do
    write(output_unit,'(a)') ']}'
    flush(output_unit)
    deallocate(shifts,masses,result,stable,times,mom,invariants,tensor_error,tensor,tensor_uv)
    deallocate(baseline,baseline_uv,pole_errors)
    deallocate(last_err_flags,last_acc_flags,worst_err_flags,worst_acc_flags,maximum_rank_errors)
    deallocate(block_result,block_err,block_acc,block_stable,block_rank_errors)
  end do
contains
  real(kind(1d0)) function square(v)
    real(kind(1d0)), intent(in) :: v(0:3)
    square=v(0)*v(0)-v(1)*v(1)-v(2)*v(2)-v(3)*v(3)
  end function square

  logical function finite_complex(value)
    complex(kind(1d0)), intent(in) :: value
    finite_complex=ieee_is_finite(real(value)).and.ieee_is_finite(aimag(value))
  end function finite_complex

  subroutine write_block(b)
    integer, intent(in) :: b
    integer :: k,m,s
    write(output_unit,'(a)',advance='no') '{"lanes":['
    do k=1,batch
      if (k>1) write(output_unit,'(a)',advance='no') ','
      write(output_unit,'(a)',advance='no') '{"cumulative_stable":'
      if (block_stable(k,b)) then
        write(output_unit,'(a)',advance='no') 'true'
      else
        write(output_unit,'(a)',advance='no') 'false'
      end if
      write(output_unit,'(a,i0,a,i0,a)',advance='no') &
        ',"last_err_flag":',block_err(k,b),',"last_acc_flag":',block_acc(k,b), &
        ',"cumulative_maximum_rank_error_estimates":['
      do s=0,rank
        if (s>0) write(output_unit,'(a)',advance='no') ','
        write(output_unit,'(es25.17e3)',advance='no') block_rank_errors(s,k,b)
      end do
      write(output_unit,'(a)',advance='no') '],"moments":['
      do m=0,count-1
        if (m>0) write(output_unit,'(a)',advance='no') ','
        write(output_unit,'(a)',advance='no') '['
        do s=1,4
          if (s>1) write(output_unit,'(a)',advance='no') ','
          if (finite_complex(block_result(m,s,k,b))) then
            write(output_unit,'(a,es25.17e3,a,es25.17e3,a)',advance='no') &
              '[',real(block_result(m,s,k,b)),',',aimag(block_result(m,s,k,b)),']'
          else
            write(output_unit,'(a)',advance='no') 'null'
          end if
        end do
        write(output_unit,'(a)',advance='no') ']'
      end do
      write(output_unit,'(a)',advance='no') ']}'
    end do
    write(output_unit,'(a)',advance='no') ']}'
  end subroutine write_block

  subroutine write_diagnostics(k)
    integer, intent(in) :: k
    integer :: s
    write(output_unit,'(a,i0,a,i0,a,i0,a,i0,a)',advance='no') &
      ',"last_err_flag":',last_err_flags(k),',"last_acc_flag":',last_acc_flags(k), &
      ',"worst_err_flag":',worst_err_flags(k),',"worst_acc_flag":',worst_acc_flags(k), &
      ',"maximum_rank_error_estimates":['
    do s=0,rank
      if (s>0) write(output_unit,'(a)',advance='no') ','
      write(output_unit,'(es25.17e3)',advance='no') maximum_rank_errors(s,k)
    end do
    write(output_unit,'(a)',advance='no') ']'
  end subroutine write_diagnostics

  subroutine write_moments(k)
    integer, intent(in) :: k
    integer :: a1,a2
    write(output_unit,'(a)',advance='no') '['
    do a1=0,count-1
      if (a1>0) write(output_unit,'(a)',advance='no') ','
      write(output_unit,'(a)',advance='no') '['
      do a2=1,4
        if (a2>1) write(output_unit,'(a)',advance='no') ','
        if (finite_complex(result(a1,a2,k))) then
          write(output_unit,'(a,es25.17e3,a,es25.17e3,a)',advance='no') &
            '[',real(result(a1,a2,k)),',',aimag(result(a1,a2,k)),']'
        else
          write(output_unit,'(a)',advance='no') 'null'
        end if
      end do
      write(output_unit,'(a)',advance='no') ']'
    end do
    write(output_unit,'(a)',advance='no') ']'
  end subroutine write_moments

  subroutine require_finite_mass_domain(k)
    integer, intent(in) :: k
    integer :: left,right
    real(kind(1d0)) :: max_positive,interval,bound
    ! Sufficient no-threshold/IR-finite bound for positive real squared masses:
    ! F(x)>=min(m_i^2)-max(0,s_ij)/2, since sum_{i<j} x_i*x_j<=1/2.
    ! This fixed-cohort driver checks the conservative numerical bound directly;
    ! the runner also keeps separate UV/IR delta-parameter checks outside timing.
    max_positive=0d0
    do left=1,n-1
      do right=left+1,n
        interval=square(shifts(right,:,k)-shifts(left,:,k))
        if (.not.ieee_is_finite(interval)) error stop 'Derived invariant overflowed'
        max_positive=max(max_positive,interval)
      end do
    end do
    bound=minval(real(masses(:,k)))-max_positive/2d0
    if (.not.ieee_is_finite(bound).or.bound<=0d0) error stop 'Positive Feynman-polynomial bound required'
  end subroutine require_finite_mass_domain

  subroutine prepare_lane(k)
    integer, intent(in) :: k
    integer :: t,distance,start_index,end_index,series_length,position
    do t=1,n-1
      mom(:,t)=cmplx(shifts(t+1,:,k),0d0,kind(1d0))
    end do
    position=0
    ! Manual invariant series: cyclic distances; final even-N series has N/2 entries.
    do distance=1,n/2
      series_length=n
      if (2*distance==n) series_length=n/2
      do start_index=0,series_length-1
        end_index=mod(start_index+distance,n)
        position=position+1
        invariants(position)=cmplx(square(shifts(end_index+1,:,k)-shifts(start_index+1,:,k)),0d0,kind(1d0))
      end do
    end do
    if (position/=size(invariants)) error stop 'Incorrect invariant series length'
    ! Immutable input pairs are checked for finite invariants before timing.
  end subroutine prepare_lane

  subroutine tensor_call(k)
    integer, intent(in) :: k
    call InitEvent_cll
    select case(n)
    case(1)
      active_interface='Aten_cll'
      call Aten_cll(tensor,tensor_uv,masses(:,k),rank,tensor_error)
    case(2)
      active_interface='Bten_cll'
      call Bten_cll(tensor,tensor_uv,mom,invariants,masses(:,k),rank,tensor_error)
    case(3)
      active_interface='Cten_cll'
      call Cten_cll(tensor,tensor_uv,mom,invariants,masses(:,k),rank,tensor_error)
    case(4)
      active_interface='Dten_cll'
      call Dten_cll(tensor,tensor_uv,mom,invariants,masses(:,k),rank,tensor_error)
    case default
      active_interface='TNten_cll'
      call TNten_cll(tensor,tensor_uv,mom,invariants,masses(:,k),n,rank,tensor_error)
    end select
  end subroutine tensor_call

  subroutine collect_diagnostics(k)
    integer, intent(in) :: k
    call GetErrFlag_cll(err_flag)
    call GetAccFlag_cll(acc_flag)
    last_err_flags(k)=err_flag
    last_acc_flags(k)=acc_flag
    worst_err_flags(k)=min(worst_err_flags(k),err_flag)
    worst_acc_flags(k)=min(worst_acc_flags(k),acc_flag)
    do i=0,rank
      if (ieee_is_finite(tensor_error(i))) then
        maximum_rank_errors(i,k)=max(maximum_rank_errors(i,k),abs(tensor_error(i)))
      else
        maximum_rank_errors(i,k)=huge(1d0)
      end if
    end do
    stable(k)=stable(k).and.err_flag==0.and.acc_flag==0.and.all(ieee_is_finite(tensor_error))
    do r=0,rank
      do nt=r,0,-1
        do nx=r-nt,0,-1
          do ny=r-nt-nx,0,-1
            nz=r-nt-nx-ny
            stable(k)=stable(k).and.finite_complex(tensor(nt,nx,ny,nz)).and. &
                                  finite_complex(tensor_uv(nt,nx,ny,nz))
          end do
        end do
      end do
    end do
    all_stable=all_stable.and.stable(k)
  end subroutine collect_diagnostics

  subroutine evaluate_batch()
    integer :: k
    do k=1,batch
      call prepare_lane(k)
      call tensor_call(k)
      pos=0
      do r=0,rank
        do nt=r,0,-1
          do nx=r-nt,0,-1
            do ny=r-nt-nx,0,-1
              nz=r-nt-nx-ny
              result(pos,1,k)=ipi2*tensor_uv(nt,nx,ny,nz)
              result(pos,2:3,k)=cmplx(0d0,0d0,kind(1d0))
              result(pos,4,k)=ipi2*(tensor(nt,nx,ny,nz)-a*tensor_uv(nt,nx,ny,nz))
              pos=pos+1
            end do
          end do
        end do
      end do
    end do
  end subroutine evaluate_batch

  subroutine verify_poles()
    integer :: k, variation
    do k=1,batch
      call prepare_lane(k)
      call tensor_call(k)
      call collect_diagnostics(k)
      pos=0
      do r=0,rank
        do nt=r,0,-1
          do nx=r-nt,0,-1
            do ny=r-nt-nx,0,-1
              nz=r-nt-nx-ny
              baseline(pos)=tensor(nt,nx,ny,nz)
              baseline_uv(pos)=tensor_uv(nt,nx,ny,nz)
              pos=pos+1
            end do
          end do
        end do
      end do
      do variation=1,3
        call SetDeltaUV_cll(0d0)
        call SetDeltaIR_cll(0d0,0d0)
        if (variation==1) call SetDeltaUV_cll(1d0)
        if (variation==2) call SetDeltaIR_cll(1d0,0d0)
        if (variation==3) call SetDeltaIR_cll(0d0,1d0)
        call tensor_call(k)
        call collect_diagnostics(k)
        pos=0
        do r=0,rank
          do nt=r,0,-1
            do nx=r-nt,0,-1
              do ny=r-nt-nx,0,-1
                nz=r-nt-nx-ny
                if (finite_complex(baseline(pos)).and.finite_complex(baseline_uv(pos)).and. &
                    finite_complex(tensor(nt,nx,ny,nz))) then
                  norm=1d0+abs(baseline(pos))+abs(baseline_uv(pos))
                  uv_error=abs(tensor(nt,nx,ny,nz)-baseline(pos))
                  if (variation==1) uv_error=abs(tensor(nt,nx,ny,nz)-baseline(pos)-baseline_uv(pos))
                  if (ieee_is_finite(norm).and.ieee_is_finite(uv_error)) then
                    pole_errors(variation,k)=max(pole_errors(variation,k),uv_error/norm)
                  else
                    pole_errors(variation,k)=huge(1d0)
                  end if
                else
                  pole_errors(variation,k)=huge(1d0)
                end if
                pos=pos+1
              end do
            end do
          end do
        end do
      end do
      if (any(pole_errors(:,k)>1d-12)) then
        bad_poles=bad_poles+1
        stable(k)=.false.
        all_stable=.false.
      end if
      call SetDeltaUV_cll(0d0)
      call SetDeltaIR_cll(0d0,0d0)
    end do
  end subroutine verify_poles
end program collier_opening_double_reference
