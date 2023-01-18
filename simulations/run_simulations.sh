# Run the simulations, generating initial conditions if necessary.
# Note - this script needs to be run using source run_simulations.sh from
# another script that has already setup the relevant environment variables.

for j in "${arr[@]}"
do
    # GADGET run:
    # Run GADGET2:
    if [ ! -d "sample${j}" ]; then
        mkdir "sample${j}"
    fi
    if [ ! -d "sample${j}/${subfolders[1]}" ]; then
        mkdir "sample${j}/${subfolders[1]}"
    fi
    if [ ! -d "sample${j}/${subfolders[0]}" ]; then
        mkdir "sample${j}/${subfolders[0]}"
    fi
# Generate reversed and forward initial conditions:
    if [ ${generateICs} == "true" ] || [ ! -f sample${j}/ic/gadget_ic_${resolution}_for.gadget2 ] || [ ! -f sample${j}/ic/gadget_ic_${resolution}_rev.gadget2 ]; then
        if [ ${mode} == "constrained" ]; then
            ${python_exec} ${simulationDir}/generate_genetIC_ics.py sample${j}/ic/gadget_ic_${resolution} --wn_file sample${j}/mcmc_${j}.h5 --Ob ${Ob} --Om ${Om} --Ol ${Ol} --ns ${ns} --hubble ${hubble} --zin ${zin} --renormalise_noise True --inverse_fourier False --flip False --reverse True --generate_reversed True --genetic_dir ${genetic_dir} --Nres ${resolution} --sample ${j} --s8 ${sigma8} --baseRes ${base_res}
        elif [ ${mode} == "unconstrained" ]; then
            ${python_exec} ${simulationDir}/generate_genetIC_ics.py sample${j}/ic/gadget_ic_${resolution} --Ob ${Ob} --Om ${Om} --Ol ${Ol} --ns ${ns} --hubble ${hubble} --zin ${zin} --renormalise_noise True --inverse_fourier False --flip False --reverse True --generate_reversed True --genetic_dir ${genetic_dir} --Nres ${resolution} --sample ${j} --s8 ${sigma8} --baseRes ${base_res}
        else
            echo "Variable mode is not recognised!"
            exit 1
        fi
    fi
    if [ ! -f sample${j}/snapshots.txt ]; then
        echo 1.0 >> sample${j}/snapshots.txt
    fi
    cd sample${j}
    if [ ${runForward} == "true" ]; then
        if [ ! -f ${subfolders[0]}/snapshot_001 ] || [ ${rerun} == "true" ]; then
            # Run the forward simulation:
            echo "Running forward simulation for sample ${j}." >> timings_newchain.txt
            if [ -f restart.0 ]; then
                echo "Restarting forward simulation for sample ${j}." >> timings_newchain.txt
                SECONDS=0
                ${mpiexec} -n ${NTASKS} ${gadget_exec} "${simulationDir}/parameters_full_forward_${resolution}.params" 1
                duration=$SECONDS
            else
                SECONDS=0
                ${mpiexec} -n ${NTASKS} ${gadget_exec} "${simulationDir}/parameters_full_forward_${resolution}.params"
                duration=$SECONDS
            fi
            echo "sample${j}, GADGET, forward: $duration s" >> timings_newchain.txt
            # Save the files and tidy up:
            mv ${snapname} "${subfolders[0]}/"
            rm restart*
        else
            echo "Existing forward run found for sample ${j}. Skipping." >> timings_newchain.txt
        fi
    fi
    if [ ${runReverse} == "true" ]; then
        if [ ! -f ${subfolders[1]}/${snapname} ] || [ ${rerun} == "true" ]; then
            # Reversed simulation:
            echo "Running reverse simulation for sample ${j}." >> timings_newchain.txt
            if [ -f restart.0 ]; then
                echo "Restarting reverse simulation for sample ${j}." >> timings_newchain.txt
                SECONDS=0
                ${mpiexec} -n ${NTASKS} ${gadget_exec} "${simulationDir}/parameters_full_reverse_${resolution}.params" 1
                duration=$SECONDS
            else
                SECONDS=0
                ${mpiexec} -n ${NTASKS} ${gadget_exec} "${simulationDir}/parameters_full_reverse_${resolution}.params"
                duration=$SECONDS
            fi
            echo "sample${j}, GADGET, reversed: $duration s" >> timings_newchain.txt
            mv ${snapname} "${subfolders[1]}/"
            rm restart*
        else
            echo "Existing reverse run found for sample ${j}. Skipping." >> timings_newchain.txt
        fi
    fi
    cd ..
done
