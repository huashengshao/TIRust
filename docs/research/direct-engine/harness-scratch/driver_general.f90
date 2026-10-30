! Scratch prototype: original-IREGI timing driver with generalized powers / even idim.
! Protocol (one header + n rows per lane):
!   BENCH id api mode n rank idim mu batch warmup repeats iterations
!   p0 p1 p2 p3 (mre,mim) power
! api: tensor -> tensor_integral_reduce (requires powers=1, idim=0)
!      general -> general_ti_reduce (real masses, first shift zero, even idim)
! Output: RESULT {"id","stable","clock_ticks_per_second","total_ns":[...],"lanes":[{"status","moments"}]}
program iregi_benchmark_general
  use iso_fortran_env, only: int64, output_unit
  use, intrinsic :: ieee_arithmetic, only: ieee_is_finite
  use global
  use ti_reduce, only: tensor_integral_reduce
  use gti_reduce, only: general_ti_reduce
  use avh_olo_forIREGI, only: olo_scale
  implicit none
  character(8192) :: line
  character(32) :: command, api
  character(128) :: request_id
  integer :: ios, mode, n, rank, idim, batch, warmup, repeats, iterations
  integer :: i, j, k, lane, repeat, iteration, count
  integer(int64) :: start, finish, rate
  real(kind(1d0)) :: mu
  real(kind(1d0)), allocatable :: shifts(:,:,:), masses_real(:,:), times(:)
  integer, allocatable :: powers(:,:)
  complex(kind(1d0)), allocatable :: masses(:,:), result(:,:,:)
  logical, allocatable :: stable(:)
  logical :: all_stable, general

  nullify(ibp_save,pave_save,shiftpaveden_save,ibp_save2,pave_save2,shiftpaveden_save2)
  nullify(cibp_save,cpave_save,cshiftpaveden_save)
  nullify(xyzmatrices_save,rsmatrices_save,cxyzmatrices_save,crsmatrices_save)
  print_banner=.true.
  call olo_scale(1d0)
  call system_clock(count_rate=rate)
  write(output_unit,'(a)') 'READY IREGI_BENCHMARK_GENERAL_V1'
  flush(output_unit)
  do
    read(*,'(a)',iostat=ios) line
    if (ios<0) exit
    if (ios/=0) error stop 'Cannot read request'
    if (len_trim(line)==0) cycle
    if (trim(line)=='QUIT') exit
    read(line,*,iostat=ios) command,request_id,api,mode,n,rank,idim,mu,batch,warmup,repeats,iterations
    if (ios/=0.or.trim(command)/='BENCH') error stop 'Invalid BENCH header'
    if (n<1.or.n>7.or.rank<0.or.rank>7.or.mode<0.or.mode>2.or.mod(idim,2)/=0) error stop 'Unsupported'
    general=trim(api)=='general'
    count=(rank+1)*(rank+2)*(rank+3)*(rank+4)/24
    allocate(shifts(n,0:3,batch),masses_real(n,batch),masses(n,batch),powers(n,batch))
    allocate(result(0:count-1,4,batch),stable(batch),times(repeats))
    do lane=1,batch
      do i=1,n
        read(*,*,iostat=ios) shifts(i,:,lane),masses(i,lane),powers(i,lane)
        if (ios/=0) error stop 'Invalid propagator row'
      end do
      masses_real(:,lane)=real(masses(:,lane))
      if (any(aimag(masses(:,lane))/=0d0)) error stop 'real masses only in this prototype'
      if (.not.general.and.(any(powers(:,lane)/=1).or.idim/=0)) error stop 'tensor api requires unit powers/idim=0'
      if (general.and.any(shifts(1,:,lane)/=0d0)) error stop 'general api requires zero first shift'
    end do
    call INITIREGI(.true.,2,1d-12)
    ONSHELL_IREGI=.false.
    ML5_CONVENTION=.true.
    MU_R_IREGI=mu
    all_stable=.true.
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
    end do
    write(output_unit,'(a)',advance='no') 'RESULT {"id":"'//trim(request_id)//'","stable":'
    if (all_stable) then
      write(output_unit,'(a)',advance='no') 'true'
    else
      write(output_unit,'(a)',advance='no') 'false'
    end if
    write(output_unit,'(a,i0,a)',advance='no') ',"clock_ticks_per_second":',rate,',"total_ns":['
    do i=1,repeats
      if (i>1) write(output_unit,'(a)',advance='no') ','
      write(output_unit,'(es25.17e3)',advance='no') times(i)
    end do
    write(output_unit,'(a)',advance='no') '],"lanes":['
    do lane=1,batch
      if (lane>1) write(output_unit,'(a)',advance='no') ','
      if (.not.stable(lane)) then
        write(output_unit,'(a)',advance='no') '{"status":"unstable","moments":null}'
        cycle
      end if
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
      write(output_unit,'(a)',advance='no') ']}'
    end do
    write(output_unit,'(a)') ']}'
    flush(output_unit)
    deallocate(shifts,masses_real,masses,powers,result,stable,times)
  end do
contains
  subroutine evaluate_batch()
    do k=1,batch
      STABLE_IREGI=.true.
      nmass=0
      massref=0d0
      do j=1,n
        if (abs(masses_real(j,k))>1d-12) then
          nmass=nmass+1
          massref(nmass)=masses_real(j,k)
        end if
      end do
      if (general) then
        call general_ti_reduce(mode,n,idim,powers(:,k),rank,count,shifts(:,:,k),masses_real(:,k),mu, &
                               result(:,:,k),stable(k))
      else
        call tensor_integral_reduce(mode,n,rank,count,shifts(:,:,k),masses_real(:,k),mu, &
                                    result(:,:,k),stable(k),1)
      end if
      stable(k)=stable(k).and.all(ieee_is_finite(real(result(:,:,k)))).and. &
                             all(ieee_is_finite(aimag(result(:,:,k))))
      all_stable=all_stable.and.stable(k)
      call IREGI_FREE_PS
    end do
  end subroutine evaluate_batch
end program iregi_benchmark_general
