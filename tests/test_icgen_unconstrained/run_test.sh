# Test the generation of initial conditions

# Generate test ICs:
${python_exec} ${simulationDir}/generate_genetIC_ics.py ic_test --wn_file --Ob ${Ob} --Om ${Om} --Ol ${Ol} --ns ${ns} --hubble ${hubble} --zin ${zin} --renormalise_noise True --inverse_fourier False --flip False --reverse True --generate_reversed True --genetic_dir ${genetic_dir} --Nres ${resolution} --sample 1
if [[ $? -ne 0 && "$1" != *error* ]]
then
    echo "--> TEST ERRORED"
    exit 1
fi

# Use built-in genetIC tool to compare with reference:
${python_exec} ${genetic_dir}/../tools/compare.py ic_test.gadget2 ../data_for_tests/reference_ics_unconstrained.gadget2
if [ $? -ne 0 ]
then 
    echo
    echo "--> TEST FAILED"
    exit 1
else
    echo
fi

