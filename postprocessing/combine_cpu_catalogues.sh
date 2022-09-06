#!/usr/bin/env bash

# Set default values of variables, in case these aren't defined:
if [ -z "${python_exec}" ]; then
    python_exec=python3
fi
if [ -z "${postProcessingDir}" ]; then
    DIR="$( dirname -- "${BASH_SOURCE[0]}"; )";   # Get the directory name
    postProcessingDir="$( realpath "$DIR"; )";  # Resolve full path
fi
if [ -z "${snapname}" ]; then
    snapname=snapshot_001
fi


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
