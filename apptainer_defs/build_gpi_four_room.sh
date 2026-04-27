#!/bin/bash
mkdir /tmp/ksandor
mv .apptainer_cache /tmp/ksandor/.apptainer_cache
ln -s /tmp/ksandor/.apptainer_cache .apptainer_cache

apptainer build gpi_pd_four_room.sif apptainer_defs/gpi_pd_four_room.def

rm .apptainer_cache
mv /tmp/ksandor/.apptainer_cache .apptainer_cache
rmdir /tmp/ksandor