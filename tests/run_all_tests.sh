if [ $# -eq 0 ]
    echo "Specify config file to run tests!"
fi

# Attempt to load the config file:
source $1

# Now run the tests:
if [ $# -gt 1 ]
then
    # Assume we gave a list of tests to run:
    for i in "${@:2}"
    do
        echo -n "Running test on ${i}   "
        cd ${i}
        source ${i}/run_test.sh
        cd ..
    done
else
    for i in test_*
    do
        echo -n "Running test on ${i}   "
        cd ${i}
        source ${i}/run_test.sh
        cd ..
    done
fi

echo "Tests seem OK"
