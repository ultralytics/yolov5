@echo off

set TH_BINARY_BUILD=1
set PYTORCH_BUILD_VERSION=%PKG_VERSION%
set PYTORCH_BUILD_NUMBER=%PKG_BUILDNUM%

set INSTALL_TEST=0

if "%USE_CUDA%" == "0" (
    set build_with_cuda=
) else (
    set build_with_cuda=1
    set desired_cuda=%CUDA_VERSION%
    :: Set up nodot version for use with magma
    set desired_cuda_nodot=%CUDA_VERSION:.=%
)

if "%build_with_cuda%" == "" goto cuda_flags_end

set CUDA_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v%desired_cuda%
set CUDA_BIN_PATH=%CUDA_PATH%\bin
set TORCH_NVCC_FLAGS=-Xfatbin -compress-all
set TORCH_CUDA_ARCH_LIST=3.7+PTX;5.0
if "%desired_cuda%" == "8.0" set TORCH_CUDA_ARCH_LIST=%TORCH_CUDA_ARCH_LIST%;6.0;6.1
if "%desired_cuda%" == "9.0" set TORCH_CUDA_ARCH_LIST=%TORCH_CUDA_ARCH_LIST%;6.0;7.0
if "%desired_cuda%" == "9.2" set TORCH_CUDA_ARCH_LIST=%TORCH_CUDA_ARCH_LIST%;6.0;6.1;7.0
if "%desired_cuda%" == "10.0" set TORCH_CUDA_ARCH_LIST=%TORCH_CUDA_ARCH_LIST%;6.0;6.1;7.0;7.5
if "%desired_cuda%" == "10.1" set TORCH_CUDA_ARCH_LIST=%TORCH_CUDA_ARCH_LIST%;6.0;6.1;7.0;7.5
if "%desired_cuda%" == "10.2" set TORCH_CUDA_ARCH_LIST=%TORCH_CUDA_ARCH_LIST%;6.0;6.1;7.0;7.5
if "%desired_cuda%" == "11.0" set TORCH_CUDA_ARCH_LIST=%TORCH_CUDA_ARCH_LIST%;6.0;6.1;7.0;7.5;8.0
if "%desired_cuda%" == "11.1" set TORCH_CUDA_ARCH_LIST=%TORCH_CUDA_ARCH_LIST%;6.0;6.1;7.0;7.5;8.0;8.6
if "%desired_cuda%" == "11.2" set TORCH_CUDA_ARCH_LIST=%TORCH_CUDA_ARCH_LIST%;6.0;6.1;7.0;7.5;8.0;8.6
if "%desired_cuda%" == "11.3" (
    set TORCH_CUDA_ARCH_LIST=%TORCH_CUDA_ARCH_LIST%;6.0;6.1;7.0;7.5;8.0;8.6
    set TORCH_NVCC_FLAGS=-Xfatbin -compress-all --threads 2
)

:cuda_flags_end

set DISTUTILS_USE_SDK=1

curl https://s3.amazonaws.com/ossci-windows/mkl_2020.2.254.7z -k -O
7z x -aoa mkl_2020.2.254.7z -omkl
set CMAKE_INCLUDE_PATH=%SRC_DIR%\mkl\include
set LIB=%SRC_DIR%\mkl\lib;%LIB%

set libuv_ROOT=%PREFIX%\Library
echo libuv_ROOT=%libuv_ROOT%

IF "%USE_SCCACHE%" == "1" (
    mkdir %SRC_DIR%\tmp_bin
    curl -k https://s3.amazonaws.com/ossci-windows/sccache.exe --output %SRC_DIR%\tmp_bin\sccache.exe
    curl -k https://s3.amazonaws.com/ossci-windows/sccache-cl.exe --output %SRC_DIR%\tmp_bin\sccache-cl.exe
    set "PATH=%SRC_DIR%\tmp_bin;%PATH%"
    set SCCACHE_IDLE_TIMEOUT=1500
)

IF "%build_with_cuda%" == "" goto cuda_end

set MAGMA_VERSION=2.5.4
if "%desired_cuda%" == "9.2" set MAGMA_VERSION=2.5.2
if "%desired_cuda%" == "10.0" set MAGMA_VERSION=2.5.2

curl https://s3.amazonaws.com/ossci-windows/magma_%MAGMA_VERSION%_cuda%desired_cuda_nodot%_release.7z -k -O
7z x -aoa magma_%MAGMA_VERSION%_cuda%desired_cuda_nodot%_release.7z -omagma_cuda%desired_cuda_nodot%_release
set MAGMA_HOME=%cd%\magma_cuda%desired_cuda_nodot%_release

set "PATH=%CUDA_BIN_PATH%;%PATH%"

if "%desired_cuda_nodot%" == "80" (
    :: Only if you use Ninja with CUDA 8
    set "CUDAHOSTCXX=%VS140COMNTOOLS%\..\..\VC\bin\amd64\cl.exe"
)

:: randomtemp is used to resolve the intermittent build error related to CUDA.
:: code: https://github.com/peterjc123/randomtemp-rust
:: issue: https://github.com/pytorch/pytorch/issues/25393
::
:: CMake requires a single command as CUDA_NVCC_EXECUTABLE, so we push the wrappers
:: randomtemp.exe and sccache.exe into a batch file which CMake invokes.
curl -kL https://github.com/peterjc123/randomtemp-rust/releases/download/v0.4/randomtemp.exe --output %SRC_DIR%\tmp_bin\randomtemp.exe
echo @"%SRC_DIR%\tmp_bin\randomtemp.exe" "%SRC_DIR%\tmp_bin\sccache.exe" "%CUDA_PATH%\bin\nvcc.exe" %%* > "%SRC_DIR%/tmp_bin/nvcc.bat"
cat %SRC_DIR%/tmp_bin/nvcc.bat
set CUDA_NVCC_EXECUTABLE=%SRC_DIR%/tmp_bin/nvcc.bat
:: CMake doesn't accept back-slashes in the path
for /F "usebackq delims=" %%n in (`cygpath -m "%CUDA_PATH%\bin\nvcc.exe"`) do set CMAKE_CUDA_COMPILER=%%n
set CMAKE_CUDA_COMPILER_LAUNCHER=%SRC_DIR%\tmp_bin\randomtemp.exe;%SRC_DIR%\tmp_bin\sccache.exe

:cuda_end

set CMAKE_GENERATOR=Ninja

IF NOT "%USE_SCCACHE%" == "1" goto sccache_end

set SCCACHE_IDLE_TIMEOUT=0

sccache --stop-server
sccache --start-server
sccache --zero-stats

set CC=sccache-cl
set CXX=sccache-cl

:sccache_end

python setup.py install
if errorlevel 1 exit /b 1

IF "%USE_SCCACHE%" == "1" (
    sccache --show-stats
    taskkill /im sccache.exe /f /t || ver > nul
    taskkill /im nvcc.exe /f /t || ver > nul
)

if NOT "%build_with_cuda%" == "" (
    copy "%CUDA_BIN_PATH%\cudnn*64_*.dll*" %SP_DIR%\torch\lib
)

exit /b 0
