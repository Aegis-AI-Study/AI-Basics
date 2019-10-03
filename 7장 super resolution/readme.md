# Super Resolution

이번에는 이미지의 해상도를 높히는 초해상(super resolution) 알고리즘에 대해 알아보자. 초해상을 위한 가장 기본이 되는 알고리즘인 SRCNN(Super Resolution CNN)에 대해 얘기하고자 한다. 

## SRCNN 구조

SRCNN구조는 그리 복잡하지 않다. SRCNN은 3개의 층으로 구성되어 있으며, 각각 Patch Extraction, Non-linear mapping, Reconstruction 레이어다. 

![SRCNN](./img/SRCNN.png)

첫번째 레이어인 Patch Extraction을 하기 전에, 저화질 이미지를 고화질 이미지의 크기에 맞게 upscaling한다. 그후, Patch Extraction레이어는 다음을 실행한다.

![layer1](https://latex.codecogs.com/gif.latex?%5Clarge%20F_1%28Y%29%3Dmax%28W_1*Y&plus;B1%29)

이때, W는 convolution kernel, Y는 upscaling된 저화질의 이미지, B는 레이어의 bias벡터를 의미한다. W의 크기는 cXf1Xf1Xn1으로 f1은 커널 크기, n1은 필터의 개수, c는 이미지의 채널 개수를 말한다. 이 과정으로 인해 이미지의 패치가 feature domain으로 표현된다.

Non-linear mapping은 1X1 크기의 커널을 이용한 convolution을 통해 얻어진다. 

![layer2](https://latex.codecogs.com/gif.latex?%5Clarge%20F_2%28Y%29%3Dmax%28W_2*F_1%28Y%29&plus;B2%29)
W2:n1X1X1Xn2
B2:n2

1X1 convolution은 네트워크의 비선형성을 증가시켜 정확도가 올라간다고 알려져있다. 여기서는 저해상도의 벡터를 고해상도의 벡터로 사상시키는데 사용되었다. 저해상도의 특징맵을 고해상도의 특징맵으로 사상하는 과정이다.

마지막으로 다시 한번 convolution을 통해 reconstruction을 한다. 

![layer3](https://latex.codecogs.com/gif.latex?%5Clarge%20F_3%28Y%29%3Dmax%28W_3*F_2%28Y%29&plus;B3%29)
W3:n2Xf3Xf3Xc
B3:c

최종적으로 고해상도의 특징맵을 다시 image domain으로 변환하는 과정이다.

## Loss

SRCNN은 가장 흔한 손실함수인 MSE(Mean Squared Error)를 사용한다.

![MSE](https://latex.codecogs.com/gif.latex?%5Clarge%20L%28%5Ctheta%29%3D%5Cfrac%7B1%7D%7Bn%7D%5Csum_%7Bi%3D1%7D%5E%7Bn%7D%5Cleft%20%5C%7C%20F%28Y_i%3B%5Ctheta%29-X_i%20%5Cright%20%5C%7C%5E%7B2%7D)

## Evaluation

SRCNN은 지금까지 사용했던 정확도(accuracy)를 이용한 평가가 아닌, RSNR(Peak Signal-to-Noise Ratio)을 사용한다. PSNR은 두 영상에 대한 차이를 정량적으로 나타낸 수치이다.

![PSNR](https://latex.codecogs.com/gif.latex?%5Clarge%20PSNR%3D10%5Ccdot%20log_%7B10%7D%28%5Cfrac%7BMAX_i%5E2%7D%7BMSE%7D%29)

