# For smaller problems run both AmgX and cuDSS solvers
for m in {8..11}; do python test_reference_solvers.py 2 1 $m ; done
for m in {6..10}; do python test_reference_solvers.py 2 2 $m ; done
for m in {6..9}; do python test_reference_solvers.py 2 3 $m ; done
for m in {5..9}; do python test_reference_solvers.py 2 4 $m ; done
for m in {5..8}; do python test_reference_solvers.py 2 5 $m ; done

for m in {3..6}; do python test_reference_solvers.py 3 1 $m ; done
for m in {2..5}; do python test_reference_solvers.py 3 2 $m ; done
for m in {1..4}; do python test_reference_solvers.py 3 3 $m ; done
for m in {1..4}; do python test_reference_solvers.py 3 4 $m ; done
for m in {1..3}; do python test_reference_solvers.py 3 5 $m ; done


# For the largest ones run only AmgX first
python test_reference_solvers.py 2 1 12 amgx,amgx64
python test_reference_solvers.py 2 2 11 amgx,amgx64
python test_reference_solvers.py 2 3 10 amgx,amgx64
python test_reference_solvers.py 2 4 10 amgx,amgx64
python test_reference_solvers.py 2 5 9 amgx,amgx64

python test_reference_solvers.py 3 1 7 amgx,amgx64
python test_reference_solvers.py 3 2 6 amgx,amgx64
python test_reference_solvers.py 3 3 5 amgx,amgx64
python test_reference_solvers.py 3 4 5 amgx,amgx64
python test_reference_solvers.py 3 5 4 amgx,amgx64


# Then try cuDSS to make sure it cannot handle the largest ones. We need
# to enforce a timeout here since cuDSS may take a very long time trying to
# produce a factorization.
TIMELIMIT=1200  # 20 minutes
timeout $TIMELIMIT python test_reference_solvers.py 2 1 12 cudss
timeout $TIMELIMIT python test_reference_solvers.py 2 2 11 cudss
timeout $TIMELIMIT python test_reference_solvers.py 2 3 10 cudss
timeout $TIMELIMIT python test_reference_solvers.py 2 4 10 cudss
timeout $TIMELIMIT python test_reference_solvers.py 2 5 9 cudss

timeout $TIMELIMIT python test_reference_solvers.py 3 1 7 cudss
timeout $TIMELIMIT python test_reference_solvers.py 3 2 6 cudss
timeout $TIMELIMIT python test_reference_solvers.py 3 3 5 cudss
timeout $TIMELIMIT python test_reference_solvers.py 3 4 5 cudss
timeout $TIMELIMIT python test_reference_solvers.py 3 5 4 cudss
