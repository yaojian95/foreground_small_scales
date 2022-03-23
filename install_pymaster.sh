module load cgpu
tf_version="gpu-2.1.0-py37"
module load tensorflow/$tf_version
module load gsl/2.5
module load cray-fftw

export FFTW_DIR=/opt/cray/pe/fftw/3.3.8.4/haswell
export CFITSIO_DIR="/usr/common/software/cfitsio/3.47"
export LDFLAGS+=" -L$GSL_DIR/lib -L$CFITSIO_DIR/lib -L/$FFTW_DIR/lib"
export CPPFLAGS+=" -I$GSL_DIR/include -I$CFITSIO_DIR/include -I$FFTW_DIR/include"
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$GSL_DIR/lib:$CFITSIO_DIR/lib:$FFTW_DIR/lib
export CC=cc
export CRAYPE_LINK_TYPE=dynamic
export XTPE_LINK_TYPE=dynamic

LDSHARED="cc -shared" CC=cc python -m pip install pymaster --user

pip install reproject --user
pip install healpy --user

python -m ipykernel install --user --name my_tf --display-name My_TF

cd $HOME/.local/share/jupyter/kernels/my_tf
file=kernel.json

sed -i '3d' $file

sed -i '3i \  \"{resource_dir}/kernel-helper.sh",' $file
sed -i '4i \  \"python",' $file

file="kernel-helper.sh"
echo "#!/bin/bash" >> $file
echo "module load cgpu" >> $file
echo "module load tensorflow/$tf_version" >> $file
echo "export LD_LIBRARY_PATH=\$LD_LIBRARY_PATH:/global/common/sw/cray/cnl7/haswell/gsl/2.5/intel/19.0.3.199/7twqxxq/lib:/usr/common/software/cfitsio/3.47/lib:/opt/cray/pe/fftw/3.3.8.4/haswell/lib:/opt/intel/compilers_and_libraries_2020.2.254/linux/compiler/lib/intel64" >> $file
echo "exec \"\$@\"" >> $file

chmod u+x kernel-helper.sh