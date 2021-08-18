# Stochastic Convolutional Sparse Coding

This is the source code for the paper entitled "Stochastic Convolutional Sparse Coding". 

### [Project Page](https://vccimaging.org/Publications/Xiong2019StochasticCSC/) | [Paper](https://vccimaging.org/Publications/Xiong2019StochasticCSC/Xiong2019StochasticCSC.pdf)

Author:
JINHUI XIONG, PETER RICHTARIK, WOLFGANG HEIDRICH

The algorithms were tested on MATLAB R2016a.

Please run main.m in SBCSC (batch mode) or SOCSC (online mode) on toy examples.

On a Core i7 machine, SBCSC takes around 140 seconds and the comparable method takes 300 seconds for 12 iterations. SOCSC takes 70 seconds and the comparable mthod takes around 440 seconds to process all 10 images.

In the file folder "filters", we include the over-complete dictionary reported in the manuscript.

## Citation
```
@inproceedings{xiong2019stochastic,
  title={Stochastic Convolutional Sparse Coding},
  author={Xiong, Jinhui and Richtarik, Peter and Heidrich, Wolfgang},
  year={2019},
  booktitle={International Symposium on Vision, Modeling, and Visualization (VMV)}
}
```
### Contact
Please contact Jinhui Xiong <jinhui.xiong@kaust.edu.sa> if you have any question or comment.
