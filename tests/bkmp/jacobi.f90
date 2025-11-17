!amu = 1822.89
!massH = 1.007825*amu
!mass = [2./3, 1./2]*massH

! r(1) = rAB
! r(2) = rBC
! r(3) = rAC

!            A     B        C
! x(1) = R =    |-----------|
! x(2) = r = |-----|

double precision function jacpot(x)
! doesn't calculate derivatives
implicit none
double precision, intent(in) :: x(2)
double precision :: r(3), dVtot(3)
r(1) = x(2)
r(2) = x(1) - x(2)/2
r(3) = r(1) + r(2)
call bkmp2(r, jacpot, dVtot, 0)
end function jacpot

subroutine jacboth(x, E, g)
! returns energy E and gradient g
implicit none
double precision, intent(in) :: x(2)
double precision, intent(out) :: E, g(2)
double precision :: r(3), dVtot(3)
r(1) = x(2)
r(2) = x(1) - x(2)/2
r(3) = r(1) + r(2)
call bkmp2(r, E, dVtot, 1)
g(1) = dVtot(2) + dVtot(3)
g(2) = dVtot(1) - 0.5d0*(dVtot(2) - dVtot(3))
end subroutine jacboth

double precision function h2pot(r)
implicit none
double precision, intent(in) :: r
double precision :: E(3)
call vH2opt95(r, E, 0)
h2pot = E(1)
end function h2pot

double precision function h2force(r)
implicit none
double precision, intent(in) :: r
double precision :: E(3)
call vH2opt95(r, E, 1)
h2force = E(2)
end function h2force

double precision function h2hess(r)
implicit none
double precision, intent(in) :: r
double precision :: E(3)
call vH2opt95(r, E, 2)
h2hess = E(3)
end function h2hess
