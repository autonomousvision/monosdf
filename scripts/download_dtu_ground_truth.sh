cd dtu_eval

wget http://roboimagedata2.compute.dtu.dk/data/MVS/SampleSet.zip

unzip SampleSet.zip

mv SampleSet/MVS\ Data Offical_DTU_Dataset
rm -r SampleSet.zip
rm -r SampleSet

cd Offical_DTU_Dataset

wget http://roboimagedata2.compute.dtu.dk/data/MVS/Points.zip

rm -r Points
unzip Points.zip
rm -r Points.zip