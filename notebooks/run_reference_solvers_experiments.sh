for m in {8..12}; do python test_reference_solvers.py 2 1 $m ; done
for m in {6..10}; do python test_reference_solvers.py 2 3 $m ; done
for m in {5..9}; do python test_reference_solvers.py 2 5 $m ; done

for m in {3..7}; do python test_reference_solvers.py 3 1 $m ; done
for m in {2..6}; do python test_reference_solvers.py 3 2 $m ; done
for m in {1..5}; do python test_reference_solvers.py 3 3 $m ; done
for m in {1..4}; do python test_reference_solvers.py 3 4 $m ; done
