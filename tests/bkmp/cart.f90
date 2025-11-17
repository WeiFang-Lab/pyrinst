! This code transforms 9 cartesian coordinates of H + H2 system into 3 distances

!full 3d "distance-coordinates" employed here:
!         A     B        
!
!                        C
! d(1)    |-----|
! d(2)          |--------|
! d(3)    |--------------|

!Input Format:
! x = (A1,B1,C1)
! x = (A2,B2,C2)
! x = (A3,B3,C3)

!Output format:
! d(3) = (dAB, dBC, dCA) 

double precision function cartpot(x)
implicit none
double precision, intent(in) :: x(3,3)
double precision, dimension(3) :: r1, r2, r3, r, dVtot
integer :: i
do i = 1, 3
	r1(i) = x(i,1)-x(i,2)
	r2(i) = x(i,2)-x(i,3)
	r3(i) = x(i,3)-x(i,1)
enddo
r(1) = sqrt(sum(r1**2)) 
r(2) = sqrt(sum(r2**2)) 
r(3) = sqrt(sum(r3**2)) 
call bkmp2(r, cartpot, dVtot, 0)
end function cartpot

subroutine cartboth(x, E, g)
! calculates Energy E and gradient g
implicit none
double precision, intent(in) :: x(3,3)
double precision, intent(out) :: E, g(3,3)
double precision, dimension(3) :: r1, r2, r3, r, dVtot
integer :: i
do i = 1, 3
	r1(i) = x(i,1)-x(i,2)
	r2(i) = x(i,2)-x(i,3)
	r3(i) = x(i,3)-x(i,1)
enddo
r(1) = sqrt(sum(r1**2)) 
r(2) = sqrt(sum(r2**2)) 
r(3) = sqrt(sum(r3**2)) 
call bkmp2(r, E, dVtot, 1)
r1 = r1*(dVtot(1)/r(1))
r2 = r2*(dVtot(2)/r(2))
r3 = r3*(dVtot(3)/r(3))
g(:,1) = r1 - r3
g(:,2) = r2 - r1
g(:,3) = r3 - r2
end subroutine cartboth
