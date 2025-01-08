# Fully Connected Neural Network Training

This is a short example on how to use train fully connected network using libtorch library. Row dataset locates on ```data``` directory. 

## Build
Make sure you have libtorch library installed. You can install it [from here](https://pytorch.org/cppdocs/installing.html).

Clone the repo
```shell
git clone https://github.com/meme-nal/FullyConnectedNNTraining.git
cd FullyConnectedNNTraining
```

Build the executables. Note that you need set your CMAKE_PREFIX_PATH in ```CMakeLists.txt```
```shell
mkdir build
cd build
cmake ..
make
```

Run it
```shell
chmod +x run.sh
./run.sh
```
