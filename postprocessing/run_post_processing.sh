# Run the post-processing routines for all simulations
# Note - this script needs to be run using source run_simulations.sh from
# another script that has already setup the relevant environment variables.

for j in "${arr[@]}"
do
    # Halo catalogues:
    cd sample${j}
    for k in "${subfolders[@]}"
    do
        cd ${k}
        if [ ${getHalos} == "true" ]; then
            # Check for pre-existing halo catalogues, and remove them.
            if [ -f ${snapname}.0000.z0.000.AHF_halos ]; then
                rm -rf ${snapname}.0*
            fi
            if [ -f *.AHF_halos ] || [ -f *.AHF_halos_mpiorder ]; then
                rm -rf ${snapname}.z0*
            fi
            # Generate halo catalogue using MPI:
            ${mpiexec} -n ${NTASKS} ${ahf_exec} ${postProcessingDir}/AHF.forward
            # Combine halo files for each CPU and re-order them in
            # descending order of mass:
            cat *.z0.000.AHF_halos > ${snapname}.z0.000.AHF_halos
            cat *.z0.000.AHF_profiles > ${snapname}.z0.000.AHF_profiles
            cat *.z0.000.AHF_particles > ${snapname}.z0.000.AHF_particles
            cat *.z0.000.AHF_substructure > ${snapname}.z0.000.AHF_substructure
            rm ${snapname}.0*
            # Re-order halos by mass:
            ${python_exec} ${postProcessingDir}/order_halo_catalogue.py ${snapname}
            # Clean up files:
            mv ${snapname}.z0.000.AHF_halos ${snapname}.z0.000.AHF_halos_mpiorder
            mv ${snapname}.z0.000.AHF_profiles ${snapname}.z0.000.AHF_profiles_mpiorder
            mv ${snapname}.z0.000.AHF_particles ${snapname}.z0.000.AHF_particles_mpiorder
            mv ${snapname}.z0.000.AHF_substructure ${snapname}.z0.000.AHF_substructure_mpiorder
            mv ${snapname}.z0.000.AHF_halos_ordered ${snapname}.z0.000.AHF_halos
            mv ${snapname}.z0.000.AHF_profiles_ordered ${snapname}.z0.000.AHF_profiles
            mv ${snapname}.z0.000.AHF_particles_ordered ${snapname}.z0.000.AHF_particles
            mv ${snapname}.z0.000.AHF_substructure_ordered ${snapname}.z0.000.AHF_substructure
            if [ -f ${snapname}.z0.000.AHF_fpos ]; then
                rm ${snapname}.z0.000.AHF_fpos # Remove any files left by pynbody.
            fi
            # Remove mpiorder files, since these actually interfere with 
            # pynbody's import routine, and we don't need them anyway:
            rm -rf *_mpiorder
        fi
        if [ ${getDen} == "true" ]; then
            # Compute the density field:
            ${dtfe_exec} "${snapname}" "${snapname}" --grid ${dtfe_res} --periodic --field density_a
        fi
        if [ ${getPS} == "true" ]; then
            # Compute power spectrum for the simulation:
            ${python_exec} ${postProcessingDir}/get_sim_ps.py ${snapname}
        fi
        cd ..
    done
    # Voronoi Density field:
    if [ ${getVor} == "true" ]; then
        cd "${subfolders[0]}"
        ${python_exec} ${postProcessingDir}/run_voz.py 0 ${snapname} ${voz_div} ${voz_ncores} ${voz_buf} ${voz_dir}
        ${mpiexec} -n ${vorCPUS} ${voz_dir}/voz1b1_mpi ${snapname}.pos ${voz_frac} ${boxsize} vor ${voz_div} # Main voronoi finder in the sub-boxes
        ${voz_dir}/voztie 4 vor # Combine subboxes to get snapshot voronoi volumes
        mv volvor.dat ${snapname}.vols # Rename so that the pipeline can find it.
        rm -rf part.* # cleanup unneeded particle files.
        cd ..
    fi
    if [ ${runPipeline} == "true" ]; then
        # Run processing pipeline:
        ${python_exec} ${postProcessingDir}/process_snapshot.py "${subfolders[0]}"/${snapname} "${subfolders[1]}"/${snapname}
    fi
    cd ..
done
