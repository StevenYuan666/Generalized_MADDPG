#!/bin/bash
for i in 1 2 3 4 5
do
	python main.py --scenario-name=simple_test --run-index=i --load-meta=0
done

for i in 1 2 3 4 5
do
        python main.py --scenario-name=simple_test --run-index=i --load-meta=1
done

