#!/usr/bin/env bash
set -ex

export CMAKE_LIBRARY_PATH=$PREFIX/lib:$PREFIX/include:$CMAKE_LIBRARY_PATH
export CMAKE_PREFIX_PATH=$PREFIX
export TH_BINARY_BUILD=1 # links CPU BLAS libraries thrice in a row (was needed for some MKL static linkage)
export PYTORCH_BUILD_VERSION=$PKG_VERSION
export PYTORCH_BUILD_NUMBER=$PKG_BUILDNUM
export USE_LLVM="/opt/llvm_no_cxx11_abi"
export LLVM_DIR="$USE_LLVM/lib/cmake/llvm"

# set OPENSSL_ROOT_DIR=/opt/openssl if it exists
if [[ -e /opt/openssl ]]; then
    export OPENSSL_ROOT_DIR=/opt/openssl
    export CMAKE_INCLUDE_PATH="/opt/openssl/include":$CMAKE_INCLUDE_PATH
fi

# Why do we disable Ninja when ninja is included in the meta.yaml? Well, using
# ninja in the conda builds leads to a system python2.7 library being called
# which leads to ascii decode errors when building third_party/onnx. Is the
# ninja n this conda env being picked up? We still need ninja in the meta.yaml
# for cpp_tests I believe though. TODO figure out what's going on here and fix
# it. It would be nice to use ninja in the builds of the conda binaries as well
export USE_NINJA=OFF
export INSTALL_TEST=0 # dont install test binaries into site-packages

# MacOS build is simple, and will not be for CUDA
if [[ "$OSTYPE" == "darwin"* ]]; then
    export USE_LLVM=$CMAKE_PREFIX_PATH
    export LLVM_DIR=$USE_LLVM/lib/cmake/llvm
    MACOSX_DEPLOYMENT_TARGET=10.9 \
        CXX=clang++ \
        CC=clang \
        python setup.py install
    exit 0
fi


if [[ -z "$USE_CUDA" || "$USE_CUDA" == 1 ]]; then
    build_with_cuda=1
fi
if [[ -n "$build_with_cuda" ]]; then
    export TORCH_NVCC_FLAGS="-Xfatbin -compress-all"
    export TORCH_CUDA_ARCH_LIST="3.7+PTX;5.0"
    if [[ $CUDA_VERSION == 8.0* ]]; then
        export TORCH_CUDA_ARCH_LIST="$TORCH_CUDA_ARCH_LIST;6.0;6.1"
    elif [[ $CUDA_VERSION == 9.0* ]]; then
        export TORCH_CUDA_ARCH_LIST="$TORCH_CUDA_ARCH_LIST;6.0;7.0"
    elif [[ $CUDA_VERSION == 9.2* ]]; then
        export TORCH_CUDA_ARCH_LIST="$TORCH_CUDA_ARCH_LIST;6.0;6.1;7.0"
    elif [[ $CUDA_VERSION == 10* ]]; then
        export TORCH_CUDA_ARCH_LIST="$TORCH_CUDA_ARCH_LIST;6.0;6.1;7.0;7.5"
    elif [[ $CUDA_VERSION == 11.0* ]]; then
        export TORCH_CUDA_ARCH_LIST="$TORCH_CUDA_ARCH_LIST;6.0;6.1;7.0;7.5;8.0"
    elif [[ $CUDA_VERSION == 11.1* ]]; then
        export TORCH_CUDA_ARCH_LIST="$TORCH_CUDA_ARCH_LIST;6.0;6.1;7.0;7.5;8.0;8.6"
    elif [[ $CUDA_VERSION == 11.2* ]]; then
        export TORCH_CUDA_ARCH_LIST="$TORCH_CUDA_ARCH_LIST;6.0;6.1;7.0;7.5;8.0;8.6"
    elif [[ $CUDA_VERSION == 11.3* ]]; then
        export TORCH_CUDA_ARCH_LIST="$TORCH_CUDA_ARCH_LIST;6.0;6.1;7.0;7.5;8.0;8.6"
    fi
    export NCCL_ROOT_DIR=/usr/local/cuda
    export USE_STATIC_CUDNN=1 # links cudnn statically (driven by tools/setup_helpers/cudnn.py)
    export USE_STATIC_NCCL=1  # links nccl statically (driven by tools/setup_helpers/nccl.py, some of the NCCL cmake files such as FindNCCL.cmake and gloo/FindNCCL.cmake)

    # not needed if using conda's cudatoolkit package. Uncomment to statically link a new CUDA version that's not available in conda yet
    # export ATEN_STATIC_CUDA=1 # links ATen / libcaffe2_gpu.so with static CUDA libs, also sets up special cufft linkage
    # export USE_CUDA_STATIC_LINK=1 # links libcaffe2_gpu.so with static CUDA libs. Likely both these flags can be de-duplicated
fi

fname_with_sha256() {
    HASH=$(sha256sum $1 | cut -c1-8)
    DIRNAME=$(dirname $1)
    BASENAME=$(basename $1)
    if [[ $BASENAME == "libnvrtc-builtins.so" ]]; then
	echo $1
    else
	INITNAME=$(echo $BASENAME | cut -f1 -d".")
	ENDNAME=$(echo $BASENAME | cut -f 2- -d".")
	echo "$DIRNAME/$INITNAME-$HASH.$ENDNAME"
    fi
}

DEPS_LIST=()
# not needed if using conda's cudatoolkit package. Uncomment to statically link a new CUDA version that's not available in conda yet
# if [[ -n "$build_with_cuda" ]]; then
#     cuda_majmin="$(echo $CUDA_VERSION | cut -f1,2 -d'.')"
#     DEPS_LIST+=("/usr/local/cuda/lib64/libcudart.so.$cuda_majmin")
#     DEPS_LIST+=("/usr/local/cuda/lib64/libnvToolsExt.so.1")
#     DEPS_LIST+=("/usr/local/cuda/lib64/libnvrtc.so.$cuda_majmin")
#     DEPS_LIST+=("/usr/local/cuda/lib64/libnvrtc-builtins.so")
# fi


# install
python setup.py install

# copy over needed dependent .so files over and tag them with their hash
patched=()
for filepath in "${DEPS_LIST[@]}"; do
    filename=$(basename $filepath)
    destpath=$SP_DIR/torch/lib/$filename
    cp $filepath $destpath

    patchedpath=$(fname_with_sha256 $destpath)
    patchedname=$(basename $patchedpath)
    if [[ "$destpath" != "$patchedpath" ]]; then
        mv $destpath $patchedpath
    fi

    patched+=("$patchedname")
    echo "Copied $filepath to $patchedpath"
done

# run patchelf to fix the so names to the hashed names
for ((i=0;i<${#DEPS_LIST[@]};++i)); do
    find $SP_DIR/torch -name '*.so*' | while read sofile; do
        origname="$(basename ${DEPS_LIST[i]})"
        patchedname=${patched[i]}
        set +e
        patchelf --print-needed $sofile | grep $origname 2>&1 >/dev/null
        ERRCODE=$?
        set -e
        if [ "$ERRCODE" -eq "0" ]; then
    	      echo "patching $sofile entry $origname to $patchedname"
    	      patchelf --replace-needed $origname $patchedname $sofile
        fi
    done
done

# set RPATH of _C.so and similar to $ORIGIN, $ORIGIN/lib and conda/lib
find $SP_DIR/torch -name "*.so*" -maxdepth 1 -type f | while read sofile; do
    echo "Setting rpath of $sofile to " '$ORIGIN:$ORIGIN/lib:$ORIGIN/../../..'
    patchelf --set-rpath '$ORIGIN:$ORIGIN/lib:$ORIGIN/../../..' --force-rpath \
             $sofile
    patchelf --print-rpath $sofile
done

# set RPATH of lib/ files to $ORIGIN and conda/lib
find $SP_DIR/torch/lib -name "*.so*" -maxdepth 1 -type f | while read sofile; do
    echo "Setting rpath of $sofile to " '$ORIGIN:$ORIGIN/../../../..'
    patchelf --set-rpath '$ORIGIN:$ORIGIN/../../../..' --force-rpath $sofile
    patchelf --print-rpath $sofile
done
