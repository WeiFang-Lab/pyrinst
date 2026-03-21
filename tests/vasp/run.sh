export TMPDIR='.'
export RUNCMD="mpirun -n 8 vasp_std"

export Temp=150
export Nbeads=14

echo "optimizing minima"
optimize.py --mode min guess_min.xyz -o optimized_min.xyz --phase solid -g 2e-2 --maxstep 0.2 --opt lBFGS -P vasp --fix fixed.xyz --cell 4.9354472160  0.0000000000  0.0000000000  -2.4677236080  4.2742226681  0.0000000000  0.0000000000  0.0000000000  10.0000000000 -F INCAR -A KPOINTS POTCAR --runcmd $RUNCMD --working-dir beads > optimized_min.out

echo "minima optimization completed!"
echo ""
echo "the optimized energy is:"
grep V\ = optimized_min.out
echo "* Result for reference:"
echo "*   V = -75.92810"

echo "optimizing classical TS"
optimize.py --mode TS guess_TS.xyz -o optimized_TS.xyz --phase solid -T $Temp -g 2e-2 --maxstep 0.2 --maxiter 30 -P vasp --fix fixed.xyz --cell 4.9354472160  0.0000000000  0.0000000000  -2.4677236080  4.2742226681  0.0000000000  0.0000000000  0.0000000000  10.0000000000 -F INCAR -A KPOINTS POTCAR --runcmd $RUNCMD --working-dir beads > optimized_TS.out

echo "TS optimization completed!"
echo ""
echo "the energy of TS is:"
grep V\ = optimized_TS.out
echo "the barrier height and TST rate:"
tail -n 3 optimized_TS.out
echo "* Result for reference:"
echo "*   V = -75.12086"
echo "*   barrier = 0.0296656 Eh = 0.807241 eV = 77.8869 kJ/mol = 18.6154 kcal/mol"
echo "*   kEyring = 1.151903734670534e-25 = 1.1314798767345396e-11 / s"
echo "*   log10(kEyring / s^-1) = -10.946353165527954"

echo "instanton optimization"
optimize.py --mode inst optimized_TS.pkl -o optimized_inst_T${Temp}N${Nbeads}.xyz --phase solid -T $Temp -N $Nbeads -s 0.3 -g 2e-2 --maxstep 0.2 --maxiter 30 -P vasp --fix fixed.xyz --cell 4.9354472160  0.0000000000  0.0000000000  -2.4677236080  4.2742226681  0.0000000000  0.0000000000  0.0000000000  10.0000000000 -F INCAR -A KPOINTS POTCAR --runcmd $RUNCMD --working-dir beads > optimized_inst_T${Temp}N${Nbeads}.out

echo "instanton optimization completed!"
echo ""
echo "the instanton rate:"
tail -n 3 optimized_inst_T${Temp}N${Nbeads}.out
echo "* Result for reference:"
echo "*   S/hbar - beta*Vr = 47.7729"
echo "*   kinst = 2.804806210007433e-19 = 2.7550755234517796e-05 / s"
echo "*   log10(kinst / s^-1) = -4.559866491560363"
