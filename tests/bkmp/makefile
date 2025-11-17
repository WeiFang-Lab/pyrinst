bkmp.so: bkmp.pyf jacobi.f90 cart.f90 bkmp2send.f
	f2py -c jacobi.f90 cart.f90 bkmp2send.f -m bkmp

bkmp.pyf: jacobi.f90 cart.f90
	f2py -h bkmp.pyf jacobi.f90 cart.f90

install: bkmp.so
	install bkmp.so ~/projects/lib/f2py

clean:
	rm -f *.so *.o *.pyf
