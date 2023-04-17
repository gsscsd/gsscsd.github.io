# OpenCV之图像变换


> 边缘检测：canny算子、sobel算子、Laplace算子、Scharr滤波器
>
> 霍夫变换
>
> 重映射
>
> 仿射变换

<!--more-->

### 边缘检测

####     canny算子、sobel算子、Laplace算子、Scharr滤波器

```C++
#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
using namespace std;
using namespace cv;

int main()
{
	Mat srcImage = imread("4.jpg");
	if (!srcImage.data)
	{
		printf("Oh, no, srcImage is error");
		return -1;

	}

	namedWindow("srcImage");
	namedWindow("Canny1");
	namedWindow("Canny2");
	namedWindow("Sobel");
	namedWindow("Laplace");
	namedWindow("Scharr");
	imshow("srcImage", srcImage);


	//一.最简单的Canny用法
	Mat dstImage;
	dstImage = srcImage.clone();

	Canny(srcImage, dstImage, 3, 9, 3);
	imshow("Canny1", dstImage);

	//二.高阶的Canny用法， 转成灰度图，降噪，用Canny，最后将得到的边缘作为掩码，
	//拷贝原图到效果图上，得到彩色的边缘图
	Mat dst, edge, gray;
	//1.创建与src同类型和大小的矩阵dst
	dst.create(srcImage.size(), srcImage.type());
	//2.将原图转换为灰色图像
	cvtColor(srcImage, gray, CV_BGR2GRAY);
	//3.先用使用3*3内核降噪
	blur(gray, edge, Size(3, 3));
	//4.运行Canny算子
	Canny(edge, edge, 3, 9, 3);
	//5.将dst内的所有元素设为0
	dst = Scalar::all(0);
	//6.使用Canny算子输出的边缘图g_cannyDetectedEdges作为掩码，来将原图srcImage拷到目标图dst中
	srcImage.copyTo(dst, edge);

	imshow("Canny2", dst);


	//sobel算子
	Mat dst_x, dst_y;
	Mat s_dst_x, s_dst_y;
	Mat ddst;
	//1.求x方向的梯度
	Sobel(srcImage, dst_x, CV_16S, 1, 0, 3, 1, 1, BORDER_DEFAULT);
	convertScaleAbs(dst_x, s_dst_x);
	imshow("X_Sobel", s_dst_x);
	//2.求y方向的梯度
	Sobel(srcImage, dst_y, CV_16S, 0, 1, 3, 1, 1, BORDER_DEFAULT);
	convertScaleAbs(dst_y, s_dst_y);
	imshow("Y_Sobel", s_dst_y);
	//合并梯度
	addWeighted(s_dst_x, 0.5, s_dst_y, 0.5, 0, ddst);
	imshow("Sobel", ddst);

	//Laplace算子
	Mat l_gray, l_dst, l_abs_dst;
	//1.使用高斯滤波消除噪声
	GaussianBlur(srcImage, l_dst, Size(3, 3), 0, 0, BORDER_DEFAULT);
	//2.转为灰度图
	cvtColor(srcImage, l_gray, CV_RGB2GRAY);
	//3.使用Laplace函数
	Laplacian(l_gray, l_ds, CV_16S, 3, t1, 0, BORDER_DEFAULT);
    //4.计算绝对值，并将结果转换成8位
	convertScaleAbs(l_dst, l_abs_dst);
	imshow("Laplace", l_abs_dst);

    //Scharr滤波器
	Mat s_x, s_y;
	Mat s_abs_x, s_abs_y, s_dst;
	//1.求X方向梯度
	Scharr(srcImage, s_x, CV_16S, 1, 0, 1, 0, BORDER_DEFAULT);
	convertScaleAbs(s_x, s_abs_x);
	imshow("X_Scharr", s_abs_x);
	//2.求Y方向梯度
	Scharr(srcImage, s_y, CV_16S, 0, 1, 1, 0, BORDER_DEFAULT);
	convertScaleAbs(s_y, s_abs_y);
	imshow("Y_Scharr", s_abs_y);
	//3.合并梯度
	addWeighted(s_abs_x, 0.5, s_abs_y, 0.5, 0, s_dst);
	imshow("Scharr", s_dst);

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

Mat srcImage, cImage, sImage, scImage, grayImage;
//Canny相关变量
Mat cannyImg;
int cannyvalue = 1;
//Sobel相关参数
Mat sobel_x, sobel_y;
Mat sobel_abs_x, sobel_abs_y;
int sobelvalue = 1;
//Scharr相关参数
Mat scharr_x, scharr_y;
Mat scharr_abs_x, scharr_abs_y;

//回调函数
void Show();
void on_Canny(int , void *);
void on_Sobel(int, void *);
void Scharr();

int main()
{
	Show();

	srcImage = imread("4.jpg");
	if (!srcImage.data)
	{
		printf("Oh, no, srcImage is error");
		return -1;
	}

	namedWindow("srcImage");
	namedWindow("Canny");
	namedWindow("Sobel");
	namedWindow("Scharr");
	imshow("srcImage", srcImage);

	cImage.create(srcImage.size(), srcImage.type());

	cvtColor(srcImage, grayImage, CV_BGR2GRAY);

	createTrackbar("canny", "Canny", &cannyvalue, 120, on_Canny);
	createTrackbar("sobel", "Sobel", &sobelvalue, 3, on_Sobel);
	on_Canny(cannyvalue, 0);
	on_Sobel(sobelvalue, 0);

	Scharr();

	while (char(waitKey(1)) != 'q') {}

	return 0;
}
void Show()
{
	printf("By 晴宝");
}

void on_Canny(int, void *)
{
	//降噪
	blur(grayImage, cannyImg, Size(3, 3));
	//canny算子
	Canny(cannyImg, cannyImg, cannyvalue, cannyvalue * 3, 3);
	//将dstImage所有元素设为0
	cImage = Scalar::all(0);
	srcImage.copyTo(cImage, cannyImg);
	
	imshow("Canny", cImage);

}
void on_Sobel(int, void *)
{
	//x方向
	Sobel(srcImage, sobel_x, CV_16S, 1, 0, (2 * sobelvalue + 1), 1, 0, BORDER_DEFAULT);
	convertScaleAbs(sobel_x, sobel_abs_x);
	//y方向
	Sobel(srcImage, sobel_y, CV_16S, 0, 1, (2 * sobelvalue + 1), 1, 0, BORDER_DEFAULT);
	convertScaleAbs(sobel_y, sobel_abs_y);
	//合并
	addWeighted(sobel_abs_x, 0.5, sobel_abs_y, 0.5, 0, sImage);

	imshow("Sobel", sImage);
}
void Scharr()
{
	Scharr(srcImage, scharr_x, CV_16S, 1, 0, 1, 0, BORDER_DEFAULT);
	convertScaleAbs(scharr_x, scharr_abs_x);

	Scharr(srcImage, scharr_y, CV_16S, 0, 1, 1, 0, BORDER_DEFAULT);
	convertScaleAbs(scharr_y, scharr_abs_y);

	addWeighted(scharr_abs_x, 0.5, scharr_abs_y, 0.5, 0, scImage);

	imshow("Scharr", scImage);

}
```

### 霍夫变换

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
	namedWindow("Canny");
	namedWindow("Lines");
	namedWindow("Linesp");
	namedWindow("Circle");
	imshow("srcImg", srcImg);

	Mat tmpImg, dstImg1, dstImg2, dstImg3;

	//1.进行边缘检测和转化为灰度图
	Canny(srcImg, tmpImg, 50, 200, 3);
	cvtColor(tmpImg, dstImg1, CV_GRAY2BGR);
	cvtColor(tmpImg, dstImg2, CV_GRAY2BGR);
	//1.转为灰度图，进行图像平滑
	cvtColor(srcImg, dstImg3, CV_BGR2GRAY);
	GaussianBlur(dstImg3, dstImg3, Size(9, 9), 2, 2);
	

	//2.进行霍夫线变换
	vector<Vec2f> lines;  //定义一个矢量结构lines用于存放得到的线段矢量集合
	HoughLines(tmpImg, lines, 1, CV_PI / 180, 150, 0, 0);

	vector<Vec4i> linesp;
	HoughLinesP(tmpImg, linesp, 1, CV_PI / 180, 80, 50, 10);

	//2.进行霍夫圆变换
	vector<Vec3f> circles;
	HoughCircles(dstImg3, circles, CV_HOUGH_GRADIENT, 1.5, 10, 200, 100, 0, 0);

	//3.依次在图中绘制每条线段
	for (size_t i = 0; i < lines.size(); i++)
	{
		float rho = lines[i][0], theta = lines[i][1];
		Point pt1, pt2;
		double a = cos(theta), b = sin(theta);
		double x0 = a * rho, y0 = b * rho;
		pt1.x = cvRound(x0 + 1000 * (-b));
		pt1.y = cvRound(y0 + 1000 * (a));
		pt2.x = cvRound(x0 - 1000 * (-b));
		pt2.y = cvRound(y0 - 1000 * (a));
		line(dstImg1, pt1, pt2, Scalar(55, 100, 195), 1, CV_AA);

	}

	for (size_t j = 0; j < linesp.size(); j++)
	{
		Vec4i l = linesp[j];
		line(dstImg2, Point(l[0], l[1]), Point(l[2], l[3]), Scalar(186, 88, 255), 1, CV_AA);

	}

	//3.依次在图中绘制图圆
	for (size_t z = 0; z < circles.size(); z++)
	{
		Point center(cvRound(circles[z][0]), cvRound(circles[z][1]));
		int radius = cvRound(circles[z][2]);
		//绘制圆心
		circle(srcImg, center, 3, Scalar(0, 255, 0), -1, 8, 0);
		//绘制圆轮廓
		circle(srcImg, center, radius, Scalar(155, 50, 255), 3, 8, 0);
	}


	imshow("Canny", tmpImg);
	imshow("Lines", dstImg1);
	imshow("Linesp", dstImg2);
	imshow("Circle", srcImg);

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

Mat srcImage, dstImage, tmpImage;
vector<Vec4i> lines;
//变量接收的TrackBar位置参数
int threadhold = 100;

//回调函数
static void on_HoughLinesp(int , void *);

void Show();

int main()
{
	Show();
	
	srcImage = imread("4.jpg");
	if (!srcImage.data)
	{
		printf("Oh, no, srcImage is error");
		return -1;
	}
	namedWindow("srcImage");
	namedWindow("HLP");
	imshow("srcImage", srcImage);

	createTrackbar("HoughLinesp", "HLP", &threadhold, 200, on_HoughLinesp);

	//进行边缘检测和转化为灰度图
	Canny(srcImage, tmpImage, 50, 200, 3);
	cvtColor(tmpImage, dstImage, CV_GRAY2BGR);
    //调用一次回调函数，调用一次HoughLinesp函数
	on_HoughLinesp(threadhold, 0);
	HoughLinesP(tmpImage, lines, 1, CV_PI/180, 80, 50, 10);

	imshow("HLP", dstImage);
	
	while (char(waitKey(1) != 'q') ) {}
	return 0;
}

static void on_HoughLinesp(int, void *)
{
	Mat mydstImage = dstImage.clone();
	Mat mytmpImage = tmpImage.clone();

	vector<Vec4i> mylines;
	HoughLinesP(mytmpImage, mylines, 1, CV_PI/180, threadhold + 1, 50, 10);

	for (size_t i = 0; i < mylines.size(); i++)
	{
		Vec4i l = mylines[i];
		line(mydstImage, Point(l[0],l[1]), Point(l[2],l[3]), Scalar(23, 180, 55), 1, CV_AA);
	}

	imshow("HLP", mydstImage);
}

void Show()
{
	printf("\n\n\n\t请调整滚动条观察图像效果\n\n");
	printf("\n\n\t\t\t\t\t\t\t\t\t\t\t by 晴宝");
}
```

### 重映射

```C++
#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
using namespace std;
using namespace cv;

int main()
{
  Mat srcImage, dstImage;
  Mat map_x, map_y;
  
  srcImage = imread("4.jpg");
  if(!srcImage.data)
  {
    printf("Oh,no, srcImage is error");
	return false;
  }
  
  imshow("srcImage", srcImage);
  
  //创建和原图一样的效果图，x重映射图， y重映射图
  dstImage.create(srcImage.size(), srcImage.type());
  map_x.create(srcImage.size(), CV_32FC1);
  map_y.create(srcImage.size(), CV_32FC1);
  
  //双层循环，遍历每一个像素点，改变map_x & map_y的值
  for(int j = 0; j < srcImage.rows; j++)
  {
     for(int i = 0; i < srcImage.cols; i++)
	 {
	    //改变map_x & map_y的值
	    map_x.at<float>(j, i) =  static_cast<float>(i);
		map_y.at<float>(j, i) = static_cast<float>(srcImage.rows - j);
	 }
  }
  
  //进行重映射操作
  remap(srcImage, dstImage, map_x, map_y, CV_INTER_LINEAR, BORDER_CONSTANT, Scalar(0, 0));
  
  imshow("dstImage", dstImage);
  
  waitKey(0);
  
  
  return 0;
}
```

```C++
//重映射操作
#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
using namespace std;
using namespace cv;

Mat srcImage, dstImage;
Mat map_x, map_y;

int update_map(int key);//更新按键按键
void Show();

int main()
{
	Show();

	srcImage = imread("4.jpg");
	if (!srcImage.data)
	{
		printf("Oh, no, srcImage is error");
		return false;
	}
	imshow("srcImage", srcImage);

	dstImage.create(srcImage.size(), srcImage.type());
	map_x.create(srcImage.size(), CV_32FC1);
	map_y.create(srcImage.size(), CV_32FC1);

	//轮询按键，更新map_x和map_y的值，进行重映射操作并显示效果图
	while (1)
	{
		//获取键盘按键
		int key = waitKey(0);

		//判断ESC是否按下，若按下便退出
		if ((key & 255) == 27)
		{
			cout << "程序退出。。。。。。\n";
			break;
		}

		//根据按下的键盘按键更新map_x & map_y的值，然后调用remap()进行重映射
		update_map(key);
		remap(srcImage, dstImage, map_x, map_y, CV_INTER_LINEAR, BORDER_CONSTANT, Scalar(0, 0, 0));

		imshow("dstImage", dstImage);
	}

	return 0;
}

void Show()
{
	printf("\n\n\t\t\t\t欢迎来到重映射示例程序~\n");
	printf("\t当前使用的OpenCV的版本为",CV_VERSION);
	printf("\n\t\t按键操作说明：\n\n"
	       "\t\t键盘按键【ESC】- 退出程序\n"
	       "\t\t键盘按键【1】 - 第一种映射方式\n"
	       "\t\t键盘按键【2】 - 第二种映射方式\n"
	       "\t\t键盘按键【3】 - 第三种映射方式\n"
	       "\t\t键盘按键【4】 - 第四种映射方式\n"
	       "\t\t\t\t\t\t\t\t\t\t\t\t\t by 晴宝\n");
}
int update_map(int key)
{
	//双层循环，遍历每一个像素点
	for (int j = 0; j < srcImage.rows; j++)
	{
		for (int i = 0; i < srcImage.cols; i++)
		{
			switch (key)
			{
			case '1':
				if (i > srcImage.cols * 0.25 && i < srcImage.cols * 0.75 &&
					j > srcImage.rows * 0.25 && j < srcImage.rows * 0.75)
				{
					map_x.at<float>(j, i) = static_cast<float>(2 * (i - srcImage.cols * 0.25) + 0.5);
					map_y.at<float>(j, i) = static_cast<float>(2 * (j - srcImage.rows * 0.25) + 0.5);
				}
				else
				{
					map_x.at<float>(j, i) = 0;
					map_y.at<float>(j, i) = 0;
				}
				break;
			case '2':
				map_x.at<float>(j, i) = static_cast<float>(i);
				map_y.at<float>(j, i) = static_cast<float>(srcImage.rows - j);
				break;
			case '3':
				map_x.at<float>(j, i) = static_cast<float>(srcImage.cols - i);
				map_y.at<float>(j, i) = static_cast<float>(j);
				break;
			case '4':
				map_x.at<float>(j, i) = static_cast<float>(srcImage.cols - i);
				map_y.at<float>(j, i) = static_cast<float>(srcImage.rows - j);
				break;
			}
		}
	}

	return 1;
}

```

### 仿射变换

```C++
#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
using namespace std;
using namespace cv;

void Show();

int main()
{
	Show();

	//1.参数准备
	//定义两组点，代表两个三角形
	Point2f srcTriangle[3];
	Point2f dstTriangle[3];

	//定义一些Mat变量
	Mat rotMat(2, 3, CV_32FC1);
	Mat warpMat(2, 3, CV_32FC1);
	Mat srcImage, dstImage_warp, dstImage_rotate;

	srcImage = imread("4.jpg");
	if (!srcImage.data)
	{
		printf("Oh, no, srcImage is error");
		return false;
	}
	imshow("srcImage", srcImage);

	//设置目标图像的大小和类型与源图像一致
	dstImage_warp = Mat::zeros(srcImage.rows, srcImage.cols, srcImage.type());

	//设置源图像和目标图像上的三组点以计算仿射变换
	srcTriangle[0] = Point2f(0, 0);
	srcTriangle[1] = Point2f(static_cast<float>(srcImage.cols - 1), 0);
	srcTriangle[2] = Point2f(static_cast<float>(srcImage.rows - 1));

	dstTriangle[0] = Point2f(static_cast<float>(srcImage.cols*0.0), static_cast<float>(srcImage.rows*0.33));
	dstTriangle[1] = Point2f(static_cast<float>(srcImage.cols*0.65), static_cast<float>(srcImage.rows*0.35));
	dstTriangle[2] = Point2f(static_cast<float>(srcImage.cols*0.15), static_cast<float>(srcImage.rows*0.6));

	//求得仿射变换
	warpMat = getAffineTransform(srcTriangle, dstTriangle);

	//对源图像应用刚刚求得的仿射变换
	warpAffine(srcImage, dstImage_warp, warpMat, dstImage_warp.size());

	//对图像进行缩放后再旋转
	//计算绕图像中点顺时针旋转50度缩放因子为0.6的旋转矩阵
	Point center = Point(dstImage_warp.cols / 2, dstImage_warp.rows / 2);
	double angle = -30.0;
	double scale = 0.8;

	//通过上面的旋转细节信息求得旋转矩阵
	rotMat = getRotationMatrix2D(center, angle, scale);

	//旋转已缩放后的图像
	warpAffine(dstImage_warp, dstImage_rotate, rotMat, dstImage_warp.size());

	imshow("dstImage_warp", dstImage_warp);
	imshow("dstImage_rotate", dstImage_rotate);


	waitKey(0);

	return 0;
}

void Show()
{
	printf("\n\n\n\t欢迎来到【仿射变换】示例程序~\n\n");
	printf("\n\n\t\t\t\t\t\t\t\t\tby 晴宝\n\n\n");
}

```


