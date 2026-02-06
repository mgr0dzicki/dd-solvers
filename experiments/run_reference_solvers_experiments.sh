# # For smaller problems run both AmgX and cuDSS solvers
# for m in {8..11}; do python test_reference_solvers.py 2 1 $m amgx,cudss ; done
# for m in {6..9}; do python test_reference_solvers.py 2 3 $m amgx,cudss ; done
# for m in {5..8}; do python test_reference_solvers.py 2 5 $m amgx,cudss ; done

# for m in {3..6}; do python test_reference_solvers.py 3 1 $m amgx,cudss ; done
# for m in {2..5}; do python test_reference_solvers.py 3 2 $m amgx,cudss ; done
# for m in {1..4}; do python test_reference_solvers.py 3 3 $m amgx,cudss ; done
# for m in {1..3}; do python test_reference_solvers.py 3 4 $m amgx,cudss ; done

# # For the largest ones run only AmgX first
# python test_reference_solvers.py 2 1 12 amgx
# python test_reference_solvers.py 2 3 10 amgx
# python test_reference_solvers.py 2 5 9 amgx

# python test_reference_solvers.py 3 1 7 amgx
# python test_reference_solvers.py 3 2 6 amgx
# python test_reference_solvers.py 3 3 5 amgx
# python test_reference_solvers.py 3 4 4 amgx

# # Then try cuDSS to make sure it cannot handle the largest ones. We need
# # to enforce a timeout here since cuDSS may take a very long time trying to
# # produce a factorization.
# TIMELIMIT=1200  # 20 minutes
# timeout $TIMELIMIT python test_reference_solvers.py 2 1 12 cudss
# timeout $TIMELIMIT python test_reference_solvers.py 2 3 10 cudss
# timeout $TIMELIMIT python test_reference_solvers.py 2 5 9 cudss

# timeout $TIMELIMIT python test_reference_solvers.py 3 1 7 cudss
# timeout $TIMELIMIT python test_reference_solvers.py 3 2 6 cudss
# timeout $TIMELIMIT python test_reference_solvers.py 3 3 5 cudss
# timeout $TIMELIMIT python test_reference_solvers.py 3 4 4 cudss

for m in {6..10}; do python test_reference_solvers.py 2 2 $m amgx,cudss ; done
for m in {5..9}; do python test_reference_solvers.py 2 4 $m amgx,cudss ; done
# python test_reference_solvers.py 3 4 4 amgx,cudss dopisz wyzej, ale juz jest policzone
for m in {1..3}; do python test_reference_solvers.py 3 5 $m amgx,cudss ; done

python test_reference_solvers.py 2 2 11 amgx
python test_reference_solvers.py 2 4 10 amgx

python test_reference_solvers.py 3 4 5 amgx
python test_reference_solvers.py 3 5 4 amgx

TIMELIMIT=1200  # 20 minutes
timeout $TIMELIMIT python test_reference_solvers.py 2 2 11 cudss
timeout $TIMELIMIT python test_reference_solvers.py 2 4 10 cudss

timeout $TIMELIMIT python test_reference_solvers.py 3 4 5 cudss
timeout $TIMELIMIT python test_reference_solvers.py 3 5 4 cudss

