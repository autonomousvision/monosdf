
#!/bin/bash
mkdir data
cd data

# DTU
wget https://s3.eu-central-1.amazonaws.com/avg-projects/monosdf/data/DTU.tar
tar -xf DTU.tar
rm -rf DTU.tar

# scannet
wget https://s3.eu-central-1.amazonaws.com/avg-projects/monosdf/data/scannet.tar
tar -xf scannet.tar
rm -rf scannet.tar

# Replica
wget https://s3.eu-central-1.amazonaws.com/avg-projects/monosdf/data/Replica.tar
tar -xf Replica.tar
rm -rf Replica.tar

# tnt_advanced
wget https://s3.eu-central-1.amazonaws.com/avg-projects/monosdf/data/tnt_advanced.tar
tar -xf tnt_advanced.tar
rm -rf tnt_advanced.tar


