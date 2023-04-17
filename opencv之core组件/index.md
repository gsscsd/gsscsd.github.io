# OpenCV之core组件进阶


> 初级图像混合
>
> 分离颜色通道、多通道图像混合
>
> 改变图像对比度和亮度

<!--more-->

### 初级图像混合

```C++
#include <iostream>
#include <opencv2/opencv.hpp>
using namespace std;
using namespace cv;

int main()
{
  Mat srcImage = imread("src.jpg");
  Mat logoImage = imread("logo.jpg");
  
  if(!srcImage.data)
  {
    printf("Oh, no, logoImage is error");
    return false;
  }
  if(!logoImage.data)
  {
    printf("Oh,no, srcImage is error");
    return false;
  }
  
  Mat imageROI;
  imageROI = srcImage(Rect(500, 250, logoImage.cols, logoImage.rows));
  
  addWeighted(imageROI, 1.0, logoImage, 0.5, 0.0, imageROI);
  
  imshow("初级图像混合", srcImage);

  return 0;
}
```

### 分离颜色通道、多通道图像混合

```C++

#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
using namespace std;
using namespace cv;


bool MutiChannelBlending();

int main()
{
	system("close5E");

	if (MutiChannelBlending())
	{
		cout << endl << "嗯嗯，好了，得出你想要的混合图像了";
	}

	waitKey(0);

	return 0;
}

//多通道图像混合的实现函数
bool MutiChannelBlending()
{
	//定义相关变量
	Mat srcImage;
	Mat logoImage;
	vector<Mat> channels;
	Mat imageBlueChannel;

	//多通道图像混合——蓝色分量部分

	//读入图片
	logoImage = imread("apple.jpg", 0);
	srcImage = imread("4.jpg");

	if (!logoImage.data)
	{
		printf("Oh, no, logoImage失败了吧");
		return false;
	}
	if (!srcImage.data)
	{
		printf("Oh, no srcImage失败了吧");
		return false;
	}

	//把一个3通道图像转换成3个单通道图像
	split(srcImage, channels);
	//将原图的绿色通道的引用返回给imageBlueChannel，注意是引用，相当于两者等价，
	//修改其中一个另一个跟着变
	imageBlueChannel = channels.at(0);
	//将原图的蓝色通道的（500,250）坐标处右下方的一块区域和logo图进行加权操作，
	//将得到的混合结果存到imageBlueChannel中 
	addWeighted(imageBlueChannel(Rect(500, 250, logoImage.cols, logoImage.rows)), 1.0,
		logoImage, 0.5, 0, imageBlueChannel(Rect(500, 250, logoImage.cols, logoImage.rows)));

	//将三个单通道重新合成一个三通道
	merge(channels, srcImage);

	//显示效果图
	namedWindow("原图+logo蓝色通道");
	imshow("原图+logo蓝色通道", srcImage);


	//多通道图像混合——绿色分量部分

	Mat imageGreenChannel;

	srcImage = imread("src.jpg");
	logoImage = imread("logo.jpg", 0);

	if (!srcImage.data)
	{
		printf("Oh, no, srcImage失败了吧");
		return false;
	}
	if (!logoImage.data)
	{
		printf("Oh，no，logoImage失败了吧");
		return false;
	}

	split(srcImage, channels);
	imageGreenChannel = channels.at(1);
	addWeighted(imageGreenChannel(Rect(500, 250, logoImage.cols, logoImage.rows)), 1.0,
		logoImage, 0.5, 0.0, imageGreenChannel(Rect(500, 250, logoImage.cols, logoImage.rows)));
	merge(channels, srcImage);
	namedWindow("原图+logo绿色通道");
	imshow("原图+logo绿色通道", srcImage);



	//多通道图像混合——红色分量部分

	Mat  imageRedChannel;

	logoImage = imread("logo.jpg", 0);
	srcImage = imread("src.jpg");

	if (!logoImage.data)
	{
		printf("Oh, no, logoImage失败了吧");
		return false;
	}
	if (!srcImage.data)
	{
		printf("Oh, no, srcImage失败了吧");
		return false;
	}

	split(srcImage, channels);
	imageBlueChannel = channels.at(2);
	addWeighted(imageRedChannel(Rect(500, 250, logoImage.cols, logoImage.rows)), 1.0,
		logoImage, 0.5, 0.0, imageRedChannel(Rect(500, 250, logoImage.cols, logoImage.rows)));
	merge(channels, srcImage);
	namedWindow("原图+logo红色通道");
	imshow("原图+logo红色通道", srcImage);

}
```

### 改变图像对比度和亮度

```C++
#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>


Mat srcImage;
Mat dstImage;

Mat B_position; //亮度
Mat C_Position; //对比度


int main()
{
  srcImage = imread("srcImage.jpg");
  if(!srcImage.data)
  {
    printf("Oh, no, srcImage is error");
    return -1;
  }
  
  dstImage = Mat::zeros(srcImage.size(), srcImage.type());
  
  //设置对比度、亮度的初值
  C_Position = 80;
  B_position = 80;
  
  //创建窗口
  namedWindow("效果图", 1);
  
  createTrackbar("滑动条亮度", "效果图",  &B_position, 200, ConstractAndBright);
  createTrackbar("滑动条对比度", "效果图",  &C_Position, 300, ConstractAndBright);
  
  ConstractAndBright(B_position, 0);
  ConstractAndBright(C_Position, 0);
  
  while(waitKey(1) != 'q'){}
  
  
  return 0;
}

//
//描述：改变图像对比度和亮度的回调函数
//
static void ConstractAndBright(int , void *)
{

  namedWindow("原图", 1);
   
  for(int y = 0; y < srcImage.rows; y++)
  {
    for(int x = 0; x < srcImage.cols; x++)
	{
	  for(int c = 0; c < 3; c++)
	  {
	    dstImage.at<Vec3b>(y, x)[c] = saturate_cast<uchar>(
		                               (C_Position * 0.01)*srcImage.at<Vec3b>(y,x)[c] + 
		                                B_position)
	  }
	}
  }
  
  imshow("原图", srcImage);
  imshow("效果图", dstImage);
} 
```


