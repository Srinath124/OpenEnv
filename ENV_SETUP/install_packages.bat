:: Try activating the Conda environment
echo Activating the Conda environment 'EDU'...

:: Check if CONDA_PREFIX is set, which would indicate conda is initialized
IF "%CONDA_PREFIX%"=="" (
    :: If not set, find the correct path to conda's activation script
    IF EXIST "%UserProfile%\miniconda3\condabin\conda.bat" (
        call "%UserProfile%\miniconda3\condabin\conda.bat" activate EDU
    ) ELSE IF EXIST "%UserProfile%\Anaconda3\condabin\conda.bat" (
        call "%UserProfile%\Anaconda3\condabin\conda.bat" activate EDU
    ) ELSE (
        echo "Could not find the Conda activation script. Ensure that Conda is installed and properly initialized."
        exit /b 1
    )
) ELSE (
    call conda activate EDU
)


:: Install the required packages
echo Installing PyTorch, Torchvision, CUDA 11.8, and project dependencies for DeepLabV3-MobileNetV3...
conda install -c pytorch -c nvidia -c conda-forge pytorch torchvision pytorch-cuda=11.8 -y && pip install opencv-contrib-python tqdm matplotlib pillow


echo Environment setup complete. You can now run your code in the 'EDU' environment.
pause

