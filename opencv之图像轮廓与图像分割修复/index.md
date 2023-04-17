# OpenCV之图像轮廓与图像分割修复


> 使用多边形将轮廓包围:  矩形、椭圆、十字型结构

<!--more-->

### 使用多边形将轮廓包围

####     矩形、椭圆、十字型结构

```C++
#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
using namespace std;
using namespace cv;

Mat srcImg, dstImg;
//构建形状
int elementshape = MORPH_RECT;
//接收TrackBar位置参数
int ocvalue = 1;
int tbvalue = 1;
int devalue = 1;
int maxvalue = 10;

//回调函数
void OpenClose(int , void *);
void DilateErode(int , void *);
void TopBlackHat(int , void *);
//显示帮助信息
static void Show();


int main()
{
	Show();

	srcImg = imread("4.jpg");
	if (!srcImg.data)
	{
		printf("Oh, no, srcImg is error");
		return -1;
	}
	namedWindow("srcImg");
	namedWindow("OpenClose");
	namedWindow("DilateErode");
	namedWindow("TopBlackHat");

	imshow("srcImg", srcImg);


	createTrackbar("OpenC", "OpenClose", &ocvalue, maxvalue * 2 + 1, OpenClose);
	createTrackbar("TopB", "TopBlackHat", &tbvalue, maxvalue * 2 + 1, TopBlackHat);
	createTrackbar("DilateE", "DilateErode", &devalue, maxvalue * 2 + 1, DilateErode);

	//轮询获取按键信息
	while (1)
	{
		int c;

		//执行回调函数
		OpenClose(ocvalue, 0);
		TopBlackHat(tbvalue, 0);
		DilateErode(devalue, 0);

		//获取按键
		c = waitKey(0);

		//按下键盘按键Q或者ESC，程序退出
		if ((char)c == 'q' || (char)c == 27)
			break;
		//按下键盘按键1， 使用椭圆（Elliptic）结构元素MORPH_ELLIPSE
		if ((char)c == 49)
			elementshape = MORPH_ELLIPSE;
		//按下键盘按键2， 使用矩形（Rectangle）结构元素MORPH_RECT
		else if ((char)c == 50)
			elementshape = MORPH_RECT;
		//按下键盘按键3， 使用十字形（Cross-Shaped）结构元素MORPH_CROSS
		else if ((char)c == 51)
			elementshape = MORPH_CROSS;
		//按下键盘按键space， 在矩形，椭圆，十字形中循环
		else if ((char)c == ' ')
			elementshape = (elementshape + 1) % 3;
	}

	return 0;
}

void OpenClose(int, void *)
{
	//偏移量的定义
	int offset = ocvalue - maxvalue;
	int absolute_offset = offset > 0 ? offset : -offset;

	Mat element = getStructuringElement(elementshape, 
		         Size(absolute_offset * 2 +1 , absolute_offset * 2 + 1), 
		         Point(absolute_offset, absolute_offset));

	//进行操作
	if(offset < 0)
		morphologyEx(srcImg, dstImg, MORPH_OPEN, element);
	else
		morphologyEx(srcImg, dstImg, MORPH_CLOSE, element);

	imshow("OpenClose", dstImg);
	
}

void TopBlackHat(int, void *)
{
	//偏移量的定义
	int offset = tbvalue - maxvalue;
	int absolute_offset = offset > 0 ? offset : -offset;

	Mat element = getStructuringElement(elementshape,
		Size(absolute_offset * 2 + 1, absolute_offset * 2 + 1),
		Point(absolute_offset, absolute_offset));

	//进行操作
	if (offset < 0)
		morphologyEx(srcImg, dstImg, MORPH_TOPHAT, element);
	else
		morphologyEx(srcImg, dstImg, MORPH_BLACKHAT, element);

	imshow("TopBlackHat", dstImg);
}

void DilateErode(int, void *)
{
	//偏移量的定义
	int offset = devalue - maxvalue;
	int absolute_offset = offset > 0 ? offset : -offset;

	Mat element = getStructuringElement(elementshape,
		Size(absolute_offset * 2 + 1, absolute_offset * 2 + 1),
		Point(absolute_offset, absolute_offset));

	//进行操作
	if (offset < 0)
		erode(srcImg, dstImg, element);
	else
		dilate(srcImg, dstImg, element);

	imshow("DilateErode", dstImg);
}

static void Show()
{
	printf("\n\n\n请调整滚动条观察图像效果\n\n");
	printf( "\n\n\t按键说明操作：\n\n"
		    "\t\t键盘按键【ESC】或者【Q】- 退出程序\n"
		    "\t\t键盘按键【1】 - 使用椭圆(Elliptic)结构元素\n"
	        "\t\t键盘按键【2】 - 使用矩形(Rectangle)结构元素\n"
	        "\t\t键盘按键【3】 - 使用十字型(Cross-shaped)结构元素\n"
	        "\t\t键盘按键【SPACE】 - 在矩形、椭圆、十字型结构元素循环\n"
	        "\n\n\n\n\t\t\t\t\t\t\t  by晴宝"
	        );
}
```


