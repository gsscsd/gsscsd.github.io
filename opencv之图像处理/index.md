# OpenCV之图像处理


> 线性滤波：方框滤波、均值滤波、高斯滤波
>
> 非线性滤波：中值滤波、双边滤波
>
> 形态学滤波： 腐蚀、膨胀、开运算、闭运算、形态学梯度、顶帽、黑帽
>
> 漫水填充
>
> 图像金字塔和图像尺寸缩放

<!--more-->

### 线性滤波：

#### 	方框滤波、均值滤波、高斯滤波

```C++
//
//  main.cpp
//  BoxFilter
//
//  Created by 吕晴 on 2018/9/19.
//  Copyright  2018年 吕晴. All rights reserved.
//

#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
using namespace std;
using namespace cv;


//int main()
//{
//    Mat img = imread("/Users/lvqing/Project/Project_OpenCV_C++/image/7.jpg");
//
//    //创建窗口
//    namedWindow("均值滤波【原图】");
//    namedWindow("均值滤波【效果图】");
//
//    imshow("均值滤波【原图】", img);
//
//    //进行均值滤波操作
////    Mat out;
////    boxFilter(img, out, -1, Size(5, 5));
//
//
//    //进行高斯滤波
//    Mat out;
//    GaussianBlur(img, out, Size(0, 0), 0, 0);
//
//    imshow("均值滤波【效果图】", out);
//
//    waitKey(0);
//}


Mat srcImage, dstImage1, dstImage2, dstImage3;
int nBoxFilter = 3;
int nBlur = 3;
int nGaussianBlur = 3;

//轨迹条的回调函数
static void On_BoxFilter(int , void *); //方框滤波
static void On_Blur(int , void *); //均值滤波
static void On_GaussianBlur(int , void *); //高斯滤波

int main()
{
    srcImage = imread("/Users/lvqing/Project/Project_OpenCV_C++/image/7.jpg");
    if(!srcImage.data)
    {
        printf("Oh, no, srcImage is error");
        return -1;
    }
    
    dstImage1 = srcImage.clone();
    dstImage2 = srcImage.clone();
    dstImage3 = srcImage.clone();
    
    namedWindow("原图", 1);
    imshow("原图", srcImage);
    
    namedWindow("方框滤波", 1);
    createTrackbar("内核", "方框滤波", &nBoxFilter, 40, On_BoxFilter);
    On_BoxFilter(nBoxFilter, 0);
    
    namedWindow("均值滤波", 1);
    createTrackbar("内核", "均值滤波", &nBlur, 40, On_Blur);
    On_Blur(nBlur, 0);
    
    namedWindow("高斯滤波", 1);
    createTrackbar("内核", "高斯滤波", &nGaussianBlur, 40, On_GaussianBlur);
    On_GaussianBlur(nGaussianBlur, 0);
    
    
    
    while(char(waitKey(1)) != 'q'){}
    
    return 0;
}

static void On_BoxFilter(int , void *)
{
    boxFilter(srcImage, dstImage1, -1, Size( nBoxFilter + 1, nBoxFilter + 1));
    imshow("方框滤波", dstImage1);
}

static void On_Blur(int , void *)
{
    blur(srcImage, dstImage2, Size( nBlur + 1, nBlur + 1), Point(-1, -1));
    imshow("均值滤波", dstImage2);
}

static void On_GaussianBlur(int , void *)
{
    GaussianBlur(srcImage, dstImage3, Size( nGaussianBlur * 2 + 1, nGaussianBlur * 2 + 1), 0, 0);
    imshow("高斯滤波", dstImage3);
}


```

### 非线性滤波：

#### 	中值滤波、双边滤波

```C++
#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

using namespace std;
using namespace cv;

int main()
{
	Mat srcimg = imread("4.jpg");

	if (!srcimg.data)
	{
		printf("Oh, no, srcimg is error");
		return -1;
	}

	//namedWindow("MedianBlur SrcImage");
	namedWindow("BilateralFilter SrcImage");

	//imshow("MedianBlur", srcimg);
	imshow("BilateralFilter", srcimg);


	//namedWindow("MedianBlur out");
	namedWindow("BilateralFilter out");

	Mat out;
	//medianBlur(srcimg, out, 7);
	bilateralFilter(srcimg, out,25, 25*2, 25/2);

	//imshow("MedianBlur out", out);
	imshow("BilateralFilter out", out);

	waitKey(0);
}
```

### 线性、非线性总结

```C++
#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
using namespace std;
using namespace cv;


Mat srcImage, dstImage1, dstImage2, dstImage3, dstImage4, dstImage5;
int n_boxFilter = 6; //方框滤波
int n_blur = 10;    //均值滤波
int n_gaussianBlur = 10;    //高斯滤波
int n_medianBlur = 6;      //中值滤波
int n_bilateralFilter = 10; //双边滤波

//回调函数声明
static void BoxFilter(int, void *);
static void Blur(int, void *);
static void GaussianBlur(int, void *);
static void MedianBlur(int, void *);
static void BilateralFilter(int, void *);


int main()
{
	srcImage = imread("4.jpg");
	if (!srcImage.data)
	{
		printf("Oh, no, SrcImage is error");
		return -1;
	}

	dstImage1 = srcImage.clone();
	dstImage2 = srcImage.clone();
	dstImage3 = srcImage.clone();
	dstImage4 = srcImage.clone();
	dstImage5 = srcImage.clone();


	namedWindow("SrcImage");
	imshow("SrcImage", srcImage);

	namedWindow("BoxFilter");
	createTrackbar("box", "BoxFilter", &n_boxFilter, 40, BoxFilter);
	BoxFilter(n_boxFilter, 0);

	namedWindow("Blur");
	createTrackbar("blur", "Blur", &n_blur, 40, Blur);
	Blur(n_blur, 0);

	namedWindow("GaussianBlur");
	createTrackbar("gaussian", "GaussianBlur", &n_gaussianBlur, 40, GaussianBlur);
	GaussianBlur(n_gaussianBlur, 0);

	namedWindow("MedianBlur");
	createTrackbar("median", "MedianBlur", &n_medianBlur, 50, MedianBlur);
	MedianBlur(n_medianBlur, 0);

	namedWindow("BilateralFilter");
	createTrackbar("bilateral", "BilateralFilter", &n_bilateralFilter, 50, BilateralFilter);
	BilateralFilter(n_bilateralFilter, 0);


	while (char(waitKey(1) != '9')) {}


	return 0;
}


static void BoxFilter(int, void *)
{
	boxFilter(srcImage, dstImage1, -1, Size(n_boxFilter + 1, n_boxFilter + 1));
	imshow("BoxFilter", dstImage1);
}
static void Blur(int, void *)
{
	blur(srcImage, dstImage2, Size(n_blur + 1, n_blur + 1), Point(-1, -1));
	imshow("Blur", dstImage2);
}
static void GaussianBlur(int, void *)
{
	GaussianBlur(srcImage, dstImage3, Size(n_gaussianBlur * 2 + 1, n_gaussianBlur * 2 + 1), 0, 0);
	imshow("n_gaussianBlur", 0);
}
static void MedianBlur(int, void *)
{
	medianBlur(srcImage, dstImage4, n_medianBlur * 2 + 1);
	imshow("n_medianBlur", 0);
}
static void BilateralFilter(int, void *)
{
	bilateralFilter(srcImage, dstImage5, n_bilateralFilter, n_bilateralFilter * 2, n_bilateralFilter / 2);
	imshow("n_bilateralFilter", 0);
}
```

### 形态学滤波：

####      膨胀与腐蚀

```C++
#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
using namespace std;
using namespace cv;

int main()
{
	Mat srcImg = imread("4.jpg");
	if (!srcImg.data)
	{
		printf("Oh, no, srcImg is error");
		return -1;
	}

	namedWindow("srcImg");

	namedWindow("dilate");
	namedWindow("erode");

	imshow("srcImg", srcImg);

	Mat out1;
	Mat out2;
	Mat element = getStructuringElement(MORPH_RECT, Size(15, 15));
	//膨胀操作
	dilate(srcImg, out1, element);
    //腐蚀操作
	erode(srcImg, out2, element);

	imshow("dilate", out1);
	imshow("erode", out2);

	waitKey(0);

	return 0;
}

```

```C++
//使用滚动条进行膨胀腐蚀操作
#include <iostream>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/opencv.hpp>
using namespace std;
using namespace cv;

Mat srcImg, dstImg;
int g_nStructElementSize = 3;
int g_nTrackbarNumber = 1;

static void on_event(int, void *);
static void on_chengdu(int, void *);
void process();

int main()
{
	srcImg = imread("4.jpg");
	if (!srcImg.data)
	{
		printf("Oh. no. srcImg is error");
		return -1;
	}

	namedWindow("SrcImg");
	namedWindow("dande");


	imshow("srcImg", srcImg);

	createTrackbar("dande", "dande", &g_nTrackbarNumber, 1, on_event);
	createTrackbar("chengdu", "dande", &g_nStructElementSize, 5, on_chengdu);

	while (char(waitKey(1) != 'q')) {}

	return 0;
}

static void on_event(int, void *)
{
	process();

}

static void on_chengdu(int, void *)
{
	process();
}

void process()
{
	Mat element = getStructuringElement(MORPH_RECT,
		Size(2 * g_nStructElementSize + 1, 2 * g_nStructElementSize + 1),
		Point(g_nStructElementSize, g_nStructElementSize));
	if (g_nTrackbarNumber == 0)
	{
		dilate(srcImg, dstImg, element);
	}
	else
	{
		erode(srcImg, dstImg, element);
	}

	imshow("dande", dstImg);
}


```

####  开运算、闭运算、形态学梯度、顶帽、黑帽

```C++
#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
using namespace std;
using namespace cv;

int main()
{
	Mat srcImg = imread("4.jpg");
	if (!srcImg.data)
	{
		printf("Oh, no, srcImg is error");
		return -1;
	}

	namedWindow("srcImg");
	namedWindow("Open");
	namedWindow("Close");
	namedWindow("Gradient");
	namedWindow("TopHat");
	namedWindow("BlackHat");

	imshow("srcImg", srcImg);

	Mat element = getStructuringElement(MORPH_RECT, Size(15, 15));

	Mat out1, out2, out3, out4, out5;
	//先腐蚀后膨胀
	morphologyEx(srcImg, out1, MORPH_OPEN, element);
	//先膨胀后腐蚀
	morphologyEx(srcImg, out2, MORPH_CLOSE, element);
	//膨胀图 - 腐蚀图
	morphologyEx(srcImg, out3, MORPH_GRADIENT, element);
	//原图 - 开运算图
	morphologyEx(srcImg, out4, MORPH_TOPHAT, element);
	//闭运算 - 原图
	morphologyEx(srcImg, out5, MORPH_BLACKHAT, element);

	imshow("Open", out1);
	imshow("Close", out2);
	imshow("Gradient", out3);
	imshow("Tophat", out4);
	imshow("Blackhat", out5);

	waitKey(0);

	return 0;
}
```

### 漫水填充

```C++
#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
using namespace std;
using namespace cv;

Mat srcImage;

int main()
{
     srcImage = imread("4.jpg");
     if(!srcImage.data)
     {
       printf("Oh, no, srcImage is error");
       return -1;
     }

     namedWindow("srcImage");
     namedWindow("floodFill");
     imshow("srcImage", srcImage);

     Rect ccomp;
     floodFill(srcImage, Point(50, 300), Scalar(155, 255, 55), &ccomp, Scalar(20, 20, 20), Scalar(20, 20, 20));

     imshow("floodFill", srcImage);

     waitKey(0);


 	return 0;
}
```

```C++
//漫水填充
#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
using namespace std;
using namespace cv;

//定义原始图、目标图、灰度图、掩膜图
Mat srcImage, dstImage;
Mat grayImage, maskImage;
//漫水填充的模式
int fillMode = 1;
//负差最大值、正差最大值
int n_LowDifference = 20, n_UpDifference = 20;
//表示floodFill函数标识符低八位的连通值
int Connectivity = 4;
//是否为彩色图的标识符布尔值
bool IsColor = true;
//是否显示掩膜窗口的布尔值
bool UseMask = false;
//新的重新绘制的像素值
int NewMaskVal = 255;


static void Show()
{
	//输出一些帮助信息
	printf("\n\n\n欢迎来到漫水填充示例程序\n\n");
	printf("\n\n\t按键操作说明：\n\n"
		"\t\t鼠标点击图中区域- 进行漫水填充操作\n"
		"\t\t键盘按键【ESC】 - 退出程序\n"
		"\t\t键盘按键【1】- 切换彩色图/灰度图模式\n"
		"\t\t键盘按键【2】- 显示/隐藏掩膜窗口\n"
		"\t\t键盘按键【3】- 恢复原始图像\n"
		"\t\t键盘按键【4】- 使用空范围的漫水填充\n"
		"\t\t键盘按键【5】- 使用渐变、固定范围的漫水填充\n"
		"\t\t键盘按键【6】- 使用渐变、浮动范围的漫水填充\n"
		"\t\t键盘按键【7】- 操作标志符的低八度使用4位的连接模式\n"
		"\t\t键盘按键【8】- 操作标志符的低八度使用8位的连接模式\n"
		"\n\n\t\t\t\t\t\t\t\t\tby 晴宝\n\n\n");
}

//鼠标onMouse回调函数
static void onMouse(int event, int x, int y, int, void *)
{
	//若鼠标左键没有按下，便返回
	if (event != CV_EVENT_LBUTTONDOWN)
		return;

	//调用floodFill函数之前的参数准备部分
	Point seed = Point(x, y);
	int LowDifference = fillMode == 0 ? 0 : n_LowDifference;
	int UpDifference = fillMode == 0 ? 0 : n_UpDifference;
	//标志符的0～7位为Connectivity, 8~15位为NewMaskVal左移8位的值, 16～23CV_FLOODFILL_FIXED_RANGE或者0
	int flags = Connectivity + (NewMaskVal << 8) + (fillMode == 1 ? CV_FLOODFILL_FIXED_RANGE : 0);
	//随机生成bgr值
	int b = (unsigned)theRNG() & 255; //随机返回一个0～255之间的值                 
	int g = (unsigned)theRNG() & 255;
	int r = (unsigned)theRNG() & 255;
	Rect ccomp; //定义重绘区域的最小边界矩形区域

	//在重绘区域像素的新值， 若是彩色图模式，取Scalar(b, g, r)，若是灰色图模式，取Scalar(r * 0.229 + g * 0.587 + b * 0.114)
	Scalar newVal = IsColor ? Scalar(b, g, r) : Scalar(r * 0.229 + g * 0.587 + b * 0.114);
	//目标图的赋值
	Mat dst = IsColor ? dstImage : grayImage;
	int area;

	//正式调用floodFill函数
	if (UseMask)
	{
		threshold(maskImage, maskImage, 1, 128, CV_THRESH_BINARY);
		area = floodFill(dstImage, maskImage, seed, newVal, &ccomp, Scalar(LowDifference, LowDifference, LowDifference),
			Scalar(UpDifference, UpDifference, UpDifference), flags);
		imshow("mask", maskImage);
	}
	else
	{
		area = floodFill(dstImage, seed, newVal, &ccomp, Scalar(LowDifference, LowDifference, LowDifference), 
			Scalar(UpDifference, UpDifference, UpDifference), flags);
	}

	imshow("floodFill", dstImage);
	cout << area << "  个像素被重绘\n";
}

int main()
{
	system("color 2F");

	srcImage = imread("4.jpg", 1);
	if (!srcImage.data)
	{
		printf("Oh, no, srcImage is error");
		return -1;
	}

	//显示帮助文档
	Show();

	//拷贝原图到目标图
	srcImage.copyTo(dstImage);
	//转换三通道的image道灰度图
	cvtColor(srcImage, grayImage, COLOR_BGR2GRAY);
	//利用image的尺寸来初始化掩膜mask
	maskImage.create(srcImage.rows + 2, srcImage.cols + 2, CV_8UC1);

	namedWindow("floodFill", CV_WINDOW_AUTOSIZE);

	//创建Trackbar
	createTrackbar("minus", "floodFill", &n_LowDifference, 255, 0);
	createTrackbar("positive", "floodFill", &n_UpDifference, 255, 0);

	//鼠标回调函数
	setMouseCallback("floodFill", onMouse, 0);

	//循环轮询按键
	while (1)
	{
		//先显示效果图
		imshow("floodFill", IsColor ? dstImage : grayImage);

		//获取键盘按键
		int c = waitKey(0);
		//判断ESC是否按下，若按下便退出
		if ((c & 255) == 27)
		{
			cout << "程序退出。。。。。。。。。。\n";
			break;
		}

		//根据按键的不同，进行各种操作
		switch ((char)c)
		{
			//如果键盘“1”被按下， 效果图在灰度图，彩色图之间互换
		case '1':
			//若原来为彩色，并转化为灰度图，并且将掩膜mask所有元素设置为0
			if (IsColor)
			{
				cout << "键盘“1”被按下，切换彩色/灰度模式，当前操作为将【彩色模式】切换为【灰度模式】\n";
					cvtColor(srcImage, grayImage, COLOR_BGR2GRAY);
				maskImage = Scalar::all(0);
				//将标志符设为false， 表示当前图像不为彩色，而是灰度
				IsColor = false;
			}
			else///若原来为灰度图，便将原来的彩图image0再次拷贝给image，并且将掩膜mask所有元素设置为0
			{
				srcImage.copyTo(dstImage);
				maskImage = Scalar::all(0);
				//将标志符设为false， 表示当前图像为彩色
				IsColor = true;
			}
			break;
			//如果键盘“2”被按下， 显示/隐藏掩膜窗口
		case '2':
			if (UseMask)
			{
				destroyWindow("mask");
				UseMask = false;
			}
			else
			{
				namedWindow("mask", 0);
				maskImage = Scalar::all(0);
				imshow("mask", maskImage);
				UseMask = true;
			}
			break;
			//如果键盘“3”被按下，恢复原始图像 
		case '3':
			srcImage.copyTo(dstImage);
			cvtColor(dstImage, grayImage, COLOR_BGR2GRAY);
			maskImage = Scalar::all(0);
			break;
			//如果键盘“4”被按下，使用空范围的漫水填充 
		case '4':
			fillMode = 0;
			break;
			//如果键盘“5”被按下，使用渐变、固定范围的漫水填充 
		case '5':
			fillMode = 1;
			break;
			//如果键盘“6”被按下，使用渐变、浮动范围的漫水填充
		case '6':
			fillMode = 3;
			break;
			//如果键盘“7”被按下，操作标志符的低八度使用4位的连接模式
		case '7':
			fillMode = 4;
			break;
			//如果键盘“8”被按下，操作标志符的低八度使用8位的连接模式
		case '8':
			fillMode = 8;
			break;
		}
	}

	return 0;
}
```

### 图像金字塔和图像尺寸缩放

####    图像尺寸缩放

```C++
#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
using namespace std;
using namespace cv;

int main()
{
	Mat srcImg = imread("4.jpg");
	if (!srcImg.data)
	{
		printf("Oh, no, srcImg is error");
		return -1;
	}

	Mat dstImg1, dstImg2, tmpImg, dstImg3, dstImg4;
	imshow("srcImg", srcImg);

	tmpImg = srcImg; //将原图给临时变量

    //进行尺寸调整操作
	resize(tmpImg, dstImg1, Size(tmpImg.cols / 2, tmpImg.rows / 2), 0, 0, 3);
	resize(tmpImg, dstImg2, Size(tmpImg.cols * 2, tmpImg.rows * 2), 0, 0, 3);

	//进行向上取样操作
	pyrUp(tmpImg, dstImg3, Size(tmpImg.cols * 2, tmpImg.rows * 2));

	//进行向下取样操作
	pyrDown(tmpImg, dstImg4, Size(tmpImg.cols / 2, tmpImg.rows / 2));

	imshow("dstImg1", dstImg1);
	imshow("dstImg2", dstImg2);
	imshow("dstImg3", dstImg3);
	imshow("dstImg4", dstImg4);

	waitKey(0);
	return 0;
}
```

```C++
#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
using namespace std;
using namespace cv;

//定义全局变量
Mat srcImage, tmpImage, dstImage;
//全局函数声明
void Show();

int main()
{
	Show();

	srcImage = imread("4.jpg");
	if(!srcImage.data)
	{
		printf("Oh, no , srcImage is error");
		return -1;
	}

	namedWindow("srcImage");
	namedWindow("dstImage");
	imshow("srcImage", srcImage);

	//参数赋值
	tmpImage = srcImage;
	dstImage = tmpImage;

	int key = 0;

	//轮询获取按键信息
	while (1)
	{
		key = waitKey(9); //读取键值到key变量中

		//根据key变量的值，进行不同的操作
		switch (key)
		{
		case 27://按下ESC
			return 0;
			break;
		case 'q':
			return 0;
			break;
		case 'a':
			pyrUp(tmpImage, dstImage, Size(tmpImage.cols * 2, tmpImage.rows * 2));
			break;
		case 'd':
			pyrDown(tmpImage, dstImage, Size(tmpImage.cols / 2, tmpImage.rows / 2));
			break;
		case 'w':
			resize(tmpImage, dstImage, Size(tmpImage.cols * 2, tmpImage.rows * 2), 0, 0);
			break;
		case 'S':
			resize(tmpImage, dstImage, Size(tmpImage.cols / 2, tmpImage.rows / 2), 0, 0);
			break;
		case '3':
			pyrUp(tmpImage, dstImage, Size(tmpImage.cols * 2, tmpImage.rows * 2));
			break;
		case '4':
			pyrDown(tmpImage, dstImage, Size(tmpImage.cols / 2, tmpImage.rows / 2));
			break;
		case '1':
			resize(tmpImage, dstImage, Size(tmpImage.cols * 2, tmpImage.rows * 2), 0, 0);
			break;
		case '2':
			resize(tmpImage, dstImage, Size(tmpImage.cols / 2, tmpImage.rows / 2), 0, 0);
			break;
		default:
			break;
		}

		imshow("dstImage", dstImage);

		tmpImage = dstImage;
	}


	return 0;
}
void Show()
{
	printf("\n\n\n\n欢迎来到图像尺寸缩放程序~\n\n");
	printf("\n\n\n\n按键说明：\n\n\n"
	       "\t\t\t键盘按键【ESC】或者【Q】 -  退出程序"
	       "\t\t\t键盘按键【1】或者【W】 -  进行基于【resize】函数的图片放大"
	       "\t\t\t键盘按键【2】或者【S】 -  进行基于【resize】函数的图片缩小"
	       "\t\t\t键盘按键【3】或者【A】 -  进行基于【pyrUp】函数的图片放大"
	       "\t\t\t键盘按键【4】或者【D】 -  进行基于【pyrDown】函数的图片缩小"
	       "\n\n\n\n\n\n\n\n\n\n\t\t\t\t   by 晴宝\n\n\n");
}
```


