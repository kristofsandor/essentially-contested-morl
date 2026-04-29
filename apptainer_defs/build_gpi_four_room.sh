#!/bin/bash
mkdir /tmp/ksandor
mv .apptainer_cache /tmp/ksandor/.apptainer_cache
mv .tmp /tmp/ksandor/.tmp
ln -s /tmp/ksandor/.apptainer_cache .apptainer_cache
ln -s /tmp/ksandor/.tmp .tmp

apptainer build gpi_pd_four_room.sif apptainer_defs/gpi_pd_four_room.def

rm .apptainer_cache
rm .tmp
mv /tmp/ksandor/.apptainer_cache .apptainer_cache
mv /tmp/ksandor/.tmp .tmp
rmdir /tmp/ksandor