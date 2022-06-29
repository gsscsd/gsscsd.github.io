# OpenCV之Practice


> 1. 计算两个矩阵的相似性
>
> 2. 画三角形、圆形检测出来
>
> 3. Haar特征与Viola-Joines人脸检测
>
> 4. 基于OpenCV的运动物体追踪
>
> 5. 基于Opencv的数字图像识别

<!--more-->

### 计算两个矩阵的相似性

```c++

//计算两个图的相似性
#include <iostream>
#include <opencv2/opencv.hpp>
using namespace std;
using namespace cv;

Mat srcImage, dstImage;

// 函数说明
// 这个函数是用来返回两个黑白图片的相似度的
// 输入是两个图片，输出是相似度
float isSimilar(Mat &src, Mat &dst)
{
	// 先定义两个图片的矩阵
	Mat src_copy, dst_copy;
	// resize(dst, src.row, src.col);
	// 讲两个图片缩放到src的尺寸上，并且用心的矩阵来储存
	resize(src, src_copy, src.size(), 0, 0, INTER_LINEAR);
	resize(dst, dst_copy, src.size(), 0, 0, INTER_LINEAR);


	float same = 0, diff = 0;
	// 两层for循环访问所有的元素点
	for (int i = 0; i < src_copy.rows; i++)
	{
		for (int j = 0; j < src_copy.cols; j++)
		{
			// 如果像素相同，那么same加1
			// 这里的代码明明是学习笔记上的，要好好看啊
			if (src_copy.at<uchar>(i, j) == dst_copy.at<uchar>(i, j))
				same += 1;
			// 如果像素不同，那么diff加1
			else
				diff += 1;
		}
	}
	// 返回相似度
	return same / (same + diff);
}

int main()
{
	// 在main函数里面读入两个图片
	srcImage = imread("8.png");
	dstImage = imread("9.png");
	// 对两个图片转换成灰度
	cvtColor(srcImage, srcImage, CV_BGR2GRAY);
	cvtColor(dstImage, dstImage, CV_BGR2GRAY);
	// 将两个图片转成黑白图，阈值化
	threshold(srcImage, srcImage, 100, 255, CV_THRESH_BINARY);
	threshold(dstImage, dstImage, 100, 255, CV_THRESH_BINARY);
	// 定义一个float
	float sim = 0;
	sim = isSimilar(srcImage, dstImage);

	cout << " the sim is " << sim << endl;
	system("pause");
	waitKey(0);
	return 0;
}

```



### 画三角形、圆形检测出来

> 任务说明：
> 首先创建一个512×512的3通道的白色的图像
> 然后在这个图像上画三角形，四角形，五角形，六角形，圆，以及椭圆
> 对这个图像，使用霍夫曼检查直线和圆

```c++
#include <iostream>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;


//绘制椭圆
void drawEllipse(Mat img, double angle)
{
	int thickness = 2;
	int lineType = 8;

	ellipse(img, Point(40, 40), Size(20, 40), angle, 0, 360, Scalar(255, 129, 0), thickness, lineType);
}

//绘制圆
void drawCircle(Mat img, Point center)
{
	int thickness = 2;
	int lineType = 8;

	circle(img, center, 30, Scalar(168, 109, 255), thickness, lineType);
}

//画线
void drawLine(Mat img, Point start, Point end)
{
	int thickness = 2;
	int lineType = 8;

	line(img, start, end, Scalar(0, 0, 0), thickness, lineType);
}

//绘制矩形
void drawRectangle(Mat img, Point pt1, Point pt2)
{
	int thickness = 2;
	int lineType = 8;

	rectangle(img, pt1, pt2, Scalar(0, 255, 0), thickness, lineType);
}

//绘制三角形
void drawLine_Triangle(Mat img)
{
	int thickness = 2;
	int lineType = 8;

	Point l1_start(200, 150), l1_end(50, 250);
	Point l2_start(50, 250), l2_end(400, 380);
	Point l3_start(400, 380), l3_end(200, 150);
	line(img, l1_start, l1_end, Scalar(0, 0, 0), thickness, lineType);
	line(img, l2_start, l2_end, Scalar(0, 0, 0), thickness, lineType);
	line(img, l3_start, l3_end, Scalar(0, 0, 0), thickness, lineType);
}

////绘制多边形
//void drawfillPloy(Mat img)
//{
//	int lineType = 8;
//
//	Point rookPoints[1][3];
//	rookPoints[0][0] = Point(30, 30);
//	rookPoints[0][1] = Point(60, 60);
//	rookPoints[0][2] = Point(90, 90);
//
//	Point *pts[1] = {rookPoints[0]}; //点序列的序列
//	int npts[] = {3};   //pts[i]中点的数目
//	int ncontours = 3;  //pts中的序列数
//
//	fillPoly(img, pts, npts, ncontours, Scalar(168, 200, 120), lineType);
//}

//使用霍夫线检测出线
//Mat check_line(Mat img)
//{
//	Mat dstImage;
//
//	cvtColor(img, dstImage, COLOR_BGR2GRAY);
//	adaptiveThreshold(dstImage, dstImage, 50, ADAPTIVE_THRESH_GAUSSIAN_C, THRESH_BINARY_INV, 10, 10);
//
//	vector<Vec2f> lines;
//	HoughLines(dstImage, lines, 1, CV_PI/180, 80, 0, 0);
//
//	vector<Vec2f> ::iterator itl = lines.begin();
//	while (itl != lines.end())
//	{
//		rectangle(img, Point((*itl)[0] + 5, (*itl)[1] + 5), Point((*itl)[2] + 5, (*itl)[3] + 5), 2, 8);
//		++itl;
//	}
//
//	imshow("check_lines", img);
//
//	return img;
//}

//使用霍夫线检测出线
Mat check_line(Mat& img)
{
	Mat dstImage;

	cvtColor(img, dstImage, COLOR_BGR2GRAY);

	Canny(dstImage, dstImage, 150, 255, 3);

	//使用霍夫曼检测
	//vector<Vec4f> lines;
	// 注意这里的第二个参数
	//HoughLines(dstImage, lines, 1, CV_PI/180.0, 80,50,10);

	// 以下修改为霍夫曼概率直线检测函数
	// 注意这里的参数
	vector<Vec4i> lines;
	// 注意一下，使用霍夫曼概率直线检测与霍夫曼直线检测的区别
	// 注意这里的第二个参数
	HoughLinesP(dstImage, lines, 1, CV_PI / 180.0, 80, 50, 10);

	Scalar color = Scalar(0, 0, 255);
	for (size_t i = 0; i < lines.size(); i++) {
		cout << " i is " << i << "  ";
		Vec4i hline = lines[i];
		cout << hline[0] << " ||| " << hline[1] << " ||| " << hline[2] << " |||| " << hline[3] << endl;
		line(img, Point(hline[0], hline[1]), Point(hline[2], hline[3]), color, 3, LINE_AA);
	}

	imshow("check_lines", img);

	return img;
}

//使用霍夫圆检测出圆形
Mat check_circle(Mat img)
{
	Mat dstImage;

	cvtColor(img, dstImage, COLOR_BGR2GRAY);
	GaussianBlur(dstImage, dstImage, Size(7, 7), 1.5);

	vector<Vec3f> circles;
	HoughCircles(dstImage, circles, CV_HOUGH_GRADIENT, 2, 30, 200, 100, 20, 80);

	/*vector<Vec3f> ::iterator itc = circles.begin();
	while (itc != circles.end())
	{
		circle(img, Point((*itc)[0], (*itc)[1]), (*itc)[2] + 20, Scalar(0, 0, 255), 6);
		++itc;
	}*/

	for (size_t i = 0; i < circles.size(); i++)
	{
		Point center(cvRound(circles[i][0]), cvRound(circles[i][1]));
		int r = cvRound(circles[i][2]);

		circle(img, center, r, Scalar(0, 0, 255), 3, 8, 0);
	}

	imshow("check_Circle", img);

	return img;
}

int main()
{
	Mat srcImage(512, 512, CV_8UC3 ,Scalar(255, 255, 255));

	//imshow("srcImage", srcImage);

	//绘制椭圆形
	drawEllipse(srcImage, 30);
	//绘制圆形
	Point center(300, 100);
	drawCircle(srcImage, center);
	//绘制线段
	Point start(100, 100);
	Point end(200, 200);
	drawLine(srcImage, start, end);
	//绘制矩形
	Point pt1(300, 300); //矩形的第一个顶点
	Point pt2(500, 500); //矩形的对角线顶点
	drawRectangle(srcImage, pt1, pt2);
	//绘制三角形
	drawLine_Triangle(srcImage);

	//putText(srcImage, "dasha is wangjiacun", Point(70, 70), FONT_HERSHEY_PLAIN, 2, 255, 2);

	imshow("out", srcImage);
	                                                                   
	check_line(srcImage);
	check_circle(srcImage);

	waitKey(0);
	return 0;
}
```



### Haar特征与Viola-Joines人脸检测

```c++
//OpenCV3/C++ Haar特征与Viola-Joines人脸检测
#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/objdetect.hpp>
using namespace std;
using namespace cv;

int main()
{
	Mat srcImage;
	CascadeClassifier face_cacade;

	VideoCapture capture(0);
	if (!face_cacade.load("D:\\Project_OpenCV\\Cmake\\install\\etc\\haarcascades\\haarcascade_frontalface_alt.xml"))
	{
		printf("can not load the file...\n");
		return -1;
	}

	while (1)
	{
		capture >> srcImage;
		vector<Rect> faces;
		Mat gray_image;
		cvtColor(srcImage, gray_image, CV_BGR2GRAY);
		//直方图均衡化
		equalizeHist(gray_image, gray_image);
		face_cacade.detectMultiScale(gray_image, faces, 1.1, 2, 0 | CV_HAAR_SCALE_IMAGE, Size(1, 1));
		//框选出脸部区域
		for (int i = 0; i < faces.size(); i++)
		{
			RNG rng(i);
			Scalar color = Scalar(rng.uniform(0, 255), rng.uniform(0, 255), 20);
			rectangle(srcImage, faces[static_cast<int>(i)], color, 2, 8, 0);
		}
		imshow("face", srcImage);
		waitKey(30);	//延时30
	}



	Mat srcImage;

	srcImage = imread("7.jpg");

	CascadeClassifier face_cacade;

	VideoCapture capture(0);
	if (!face_cacade.load("D:\\Project_OpenCV\\Cmake\\install\\etc\\haarcascades\\haarcascade_frontalface_alt.xml"))
	{
		printf("can not load the file...\n");
		return -1;
	}
	vector<Rect> faces;
	Mat gray_image;
	cvtColor(srcImage, gray_image, CV_BGR2GRAY);

	//直方图均衡化
	equalizeHist(gray_image, gray_image);
	face_cacade.detectMultiScale(gray_image, faces, 1.1, 2, 0 | CV_HAAR_SCALE_IMAGE, Size(1, 1));

	//框选出脸部区域
	for (int i = 0; i < faces.size(); i++)
	{
		RNG rng(i);
		Scalar color = Scalar(rng.uniform(0, 255), rng.uniform(0, 255), 20);
		rectangle(srcImage, faces[static_cast<int>(i)], color, 2, 8, 0);
	}

	imshow("face", srcImage);
	waitKey(0);


	return 0;
}
```



### 基于OpenCV的运动物体追踪

> 给一段视频，程序对视频中运动的物体进行追踪，并用框框起来

```c++
////1.运动物体检测----背景减法
//#include <iostream>
//#include <opencv2/opencv.hpp>
//using namespace std;
//using namespace cv;
//
////运动物体检测函数声明
//Mat MoveDetect(Mat backgroung, Mat frame);
//
//int main()
//{
//
//	VideoCapture video("move.avi");
//	if (!video.isOpened())
//	{
//		printf("Oh, no, video is error");
//		return -1;
//	}
//
//	int frameCount = video.get(CV_CAP_PROP_FRAME_COUNT); //获取帧数
//	double FPS = video.get(CV_CAP_PROP_FPS);//获取FPS
//	Mat frame; //存储帧
//	Mat background; //存储背景图像
//	Mat result; //存储结果图像
//
//	for (int i = 0; i < frameCount; i++)
//	{
//		video >> frame; //读帧进frame
//		imshow("frame", frame);
//		if (frame.empty()) //对帧进行异常检测
//		{
//			cout << "frame is empty" << endl;
//			break;
//		}
//		int framePosition = video.get(CV_CAP_PROP_POS_FRAMES); //获取帧位置(第几帧)
//		cout << "framePosition: " << framePosition << endl;
//		if (framePosition == 1) //将第一帧作为背景图像
//			background = frame.clone();
//		result = MoveDetect(background, frame); //调用MoveDect()进行运动物体检测，返回值存入result
//		imshow("result", result);
//		if (waitKey(1000.0 / FPS) == 27) //按原FPS显示
//		{
//			cout << "ESC退出！" << endl;
//			break;
//		}
//	}
//
//
//	return 0;
//}
//
//Mat MoveDetect(Mat background, Mat frame)
//{
//	Mat result = frame.clone();
//	//1.将background和frame转为灰度图
//	Mat gray1, gray2;
//	cvtColor(background, gray1, CV_BGR2GRAY);
//	cvtColor(frame, gray2, CV_BGR2GRAY);
//	//2.将background和frame做差
//	Mat diff;
//	absdiff(gray1, gray2, diff);
//	imshow("diff", diff);
//	//3. 对差值图diff_thresh进行阈值化处理
//	Mat diff_thresh;
//	threshold(diff, diff_thresh, 50, 255, CV_THRESH_BINARY);
//	imshow("diff_thresh", diff_thresh);
//	//4.腐蚀
//	Mat kernel_erode = getStructuringElement(MORPH_RECT, Size(3, 3));
//	Mat kernel_dilate = getStructuringElement(MORPH_RECT, Size(15, 15));
//	erode(diff_thresh, diff_thresh, kernel_erode);
//	imshow("erode", diff_thresh);
//	//5.膨胀
//	dilate(diff_thresh, diff_thresh, kernel_dilate);
//	imshow("dilate", diff_thresh);
//	//6.查找轮廓并绘制轮廓
//	vector<vector<Point>> contours;
//	findContours(diff_thresh, contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);
//	drawContours(result, contours, -1, Scalar(0, 0, 255), 2);//在result上绘制轮廓
//	//7.查找正外接矩形
//	vector<Rect> boundRect(contours.size());
//	for (int i = 0; i < contours.size(); i++)
//	{
//		boundRect[i] = boundingRect(contours[i]);
//		rectangle(result, boundRect[i], Scalar(0, 255, 0), 2); //在result上绘制正外接矩阵
//	}
//
//	return result; //返回result
//}

#include "opencv2/opencv.hpp"
#include<iostream>
using namespace std;
using namespace cv;

Mat MoveDetect(Mat frame1, Mat frame2)
{
	Mat result = frame2.clone();
	Mat gray1, gray2;
	cvtColor(frame1, gray1, CV_BGR2GRAY);
	cvtColor(frame2, gray2, CV_BGR2GRAY);

	Mat diff;
	absdiff(gray1, gray2, diff);
	imshow("absdiss", diff);
	threshold(diff, diff, 45, 255, CV_THRESH_BINARY);
	imshow("threshold", diff);

	Mat element = getStructuringElement(MORPH_RECT, Size(3, 3));
	Mat element2 = getStructuringElement(MORPH_RECT, Size(25, 25));
	erode(diff, diff, element);
	imshow("erode", diff);

	dilate(diff, diff, element2);
	imshow("dilate", diff);

	vector<vector<Point>> contours;
	vector<Vec4i> hierarcy;
	//画椭圆及中心
	findContours(diff, contours, hierarcy, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);
	cout << "num=" << contours.size() << endl;
	vector<RotatedRect> box(contours.size());
	for (int i = 0; i<contours.size(); i++)
	{
		box[i] = fitEllipse(Mat(contours[i]));
		ellipse(result, box[i], Scalar(0, 255, 0), 2, 8);
		circle(result, box[i].center, 3, Scalar(0, 0, 255), -1, 8);
	}
	return result;
}

void main()
{
	VideoCapture cap("move.avi");
	if (!cap.isOpened()) //检查打开是否成功
		return;
	Mat frame;
	Mat result;
	Mat background;
	int count = 0;
	while (1)
	{
		cap >> frame;
		if (frame.empty())
			break;
		else {
			count++;
			if (count == 1)
				background = frame.clone(); //提取第一帧为背景帧
			imshow("video", frame);
			result = MoveDetect(background, frame);
			imshow("result", result);
			if (waitKey(50) == 27)
				break;
		}
	}
	cap.release();
}


////2.运动物体检测----帧查法
//#include <iostream>
//#include <opencv2/opencv.hpp>
//using namespace std;
//using namespace cv;
//
//Mat MoveDetect(Mat background, Mat frame);
//
//int main()
//{
//	VideoCapture video("move.avi");
//	if (!video.isOpened())
//	{
//		printf("Oh, no, video is error");
//		return -1;
//	}
//
//	int frameCount = video.get(CV_CAP_PROP_FRAME_COUNT);
//	double FPS = video.get(CV_CAP_PROP_FPS);
//	Mat frame;
//	Mat background;
//	Mat result;
//
//	for(int i = 0; i < frameCount; i++)
//	{
//		video >> frame;
//		imshow("frame", frame);
//		if (frame.empty())
//		{
//			printf("frame is empty!");
//			break;
//		}
//
//		if (i == 0)
//		{
//			result = MoveDetect(frame, frame);
//		}
//		else
//		{
//			result = MoveDetect(background, frame);
//		}
//
//		imshow("result", result);
//		if (waitKey(1000.0 / FPS) == 27)
//			break;
//		background = frame.clone();
//	}
//
//	return 0;
//}
//
//Mat MoveDetect(Mat background, Mat frame)
//{
//	Mat result = frame.clone();
//	Mat gray1, gray2;
//	cvtColor(background, gray1, CV_BGR2GRAY);
//	cvtColor(frame, gray2, CV_BGR2GRAY);
//
//	Mat diff;
//	absdiff(gray1, gray2, diff);
//	imshow("diff", diff);
//
//	Mat diff_thresh;
//	threshold(diff, diff_thresh, 50, 255, CV_THRESH_BINARY);
//	imshow("threshold", diff_thresh);
//
//	Mat kernel_erode = getStructuringElement(MORPH_RECT, Size(3, 3));
//	Mat kernel_dilate = getStructuringElement(MORPH_RECT, Size(18, 18));
//
//	erode(diff_thresh, diff_thresh, kernel_erode);
//	imshow("erode", diff_thresh);
//	dilate(diff_thresh, diff_thresh, kernel_dilate);
//	imshow("dilate", diff_thresh);
//
//	vector<vector<Point>> contours;
//	findContours(diff_thresh, contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);
//	drawContours(result, contours, -1, Scalar(0, 0, 255), 2);
// 
//	vector<Rect> boundRect(contours.size());
//	for (int i = 0; i < contours.size(); i++)
//	{
//		boundRect[i] = boundingRect(contours[i]);
//		rectangle(result, boundRect[i], Scalar(0, 255, 0), 2);
//	}
//	return result;
//}
```



### 基于Opencv的数字图像识别

> 准备：
> 1.10个模板图像
> 2.相似度检测函数、轮廓检测函数
> 具体内容：
> 1.读入10个模板图像
> 2.对目标图像进行轮廓检测并剪切出识别的数字
> 3.对剪切出的数字与10个模板图像相似度计算挑选出最相似的

```c++
#include <iostream>
#include <opencv2/opencv.hpp>
#include <vector>
#include <algorithm>
using namespace std;
using namespace cv;

Mat srcImage, dstImage;
vector<Mat> vec;
vector<Mat> tutu;

//相似性函数
float isSimilar(Mat &src, Mat &dst)
{
	// 先定义两个图片的矩阵
	Mat src_copy, dst_copy;
	// resize(dst, src.row, src.col);
	// 讲两个图片缩放到src的尺寸上，并且用新的矩阵来储存
	resize(src, src_copy, src.size(), 0, 0, INTER_LINEAR);
	resize(dst, dst_copy, src.size(), 0, 0, INTER_LINEAR);


	float same = 0, diff = 0;
	// 两层for循环访问所有的元素点
	for (int i = 0; i < src_copy.rows; i++)
	{
		for (int j = 0; j < src_copy.cols; j++)
		{
			// 如果像素相同，那么same加1
			// 这里的代码明明是学习笔记上的，要好好看啊
			if (src_copy.at<uchar>(i, j) == dst_copy.at<uchar>(i, j))
				same += 1;
			// 如果像素不同，那么diff加1
			else
				diff += 1;
		}
	}
	// 返回相似度
	return same / (same + diff);
}
//轮廓检测函数
void isContours(Mat &src)
{
	Mat img_gray;
	resize(src, img_gray, src.size(), 0, 0, INTER_LINEAR);

	cvtColor(src, img_gray, CV_BGR2GRAY);

	// 注意在取阈值的时候，最后一个参数
	threshold(img_gray, dstImage, 100, 255, CV_THRESH_BINARY);

	vector<vector<Point> > contours;
	vector<Vec4i> hierarchy;
	// 注意这里的第四个参数，决定了检测的准确性
	findContours(dstImage, contours, hierarchy, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE, Point(0, 0));

	RNG rng(0);
	for (int i = 0; i < contours.size(); i++)
	{
		Scalar color = Scalar(rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255));
		Rect rect = boundingRect(contours[i]);//检测外轮廓
		Mat temp = src(rect);

		imwrite("shacun" + to_string(i) + ".png", temp);
		rectangle(src, rect, Scalar(0, 0, 255), 3);//对外轮廓加矩形框
	}
}
int main()
{
	//调用轮廓检测函数
	srcImage = imread("digits_4.jpg");
	if (!srcImage.data)
	{
		printf("Oh, no, srcImage is error");
		return 0;
	}
	// 轮廓检测并保存数据
	isContours(srcImage);


	//调用相似性函数
	for (int i = 0; i < 10; i++)
	{
		//模板10个图像
		Mat s_src = imread("tutu" + to_string(i) + ".png");
		vec.push_back(s_src);

		//切割的10个图像
		Mat tu = imread("shacun" + to_string(i) + ".png");
		tutu.push_back(tu);
	}
	for (int j = 0; j < 10; j++)
	{
		int val = -1;
		float maxSim = 0;
		for (int i = 0; i < 10; i++)
		{
			float sim = 0;
			sim = isSimilar(vec[i], tutu[j]);
			if (maxSim <= sim)
			{
				maxSim = sim;
				val = i;
			}
		}
		imshow("aim is ", tutu[j]);
		imshow("model is ", vec[val]);
		waitKey(0);
		cout << " the val is " << val << " the sim is  " << maxSim << endl;
	}
	

	system("pause");
	return 0;
}

```

```c++
#include <iostream>
#include <opencv2/opencv.hpp>
#include <vector>
#include <algorithm>
using namespace std;
using namespace cv;

Mat srcImage, dstImage;
vector<Mat> vec;


/**
代码如何理解
从main函数开始，首先调用init函数
我们用init函数来加载模板图像到全局的数组vec里面
然后调用isContours函数，使用isContours函数对目标图片检测
检测完成后，接着就调用predict函数来预测轮廓检测的数数字
在predict函数里面，我们调用isSimilar函数来检测出最相似的模板图片
然后，输出对应的数字
*/


// 函数声明
void init();
void isContours(Mat &src);
float isSimilar(Mat &src, Mat &dst);
void predict(Mat &dst);

// 初始化加载模板函数
void init()
{
	// 初始化函数，用来加载模板图片
	for (int i = 0; i < 10; i++)
	{
		//模板10个图像
		Mat s_src = imread("tutu" + to_string(i) + ".png");
		vec.push_back(s_src);
	}
	cout << "***************" << "init is OK." << "***************" << endl;

}

// 预测函数
void predict(Mat &dst)
{
	// 检测数字函数
	int val = -1;
	float maxSim = 0;
	for (int i = 0; i < 10; i++)
	{
		// cvtColor(vec[i], vec[i], CV_BGR2GRAY);
		// threshold(vec[j], vec[j], 100, 255, CV_THRESH_BINARY);

		float sim = 0;
		sim = isSimilar(vec[i], dst);
		if (maxSim <= sim)
		{
			maxSim = sim;
			val = i;
		}
	}
	imshow("aim is ", dst);
	imshow("model is ", vec[val]);
	waitKey(0);
	cout << " the val is " << val << " the sim is  " << maxSim << endl;

}


//轮廓检测函数
void isContours(Mat &src)
{
	Mat img_gray;
	resize(src, img_gray, src.size(), 0, 0, INTER_LINEAR);

	cvtColor(src, img_gray, CV_BGR2GRAY);

	// 注意在取阈值的时候，最后一个参数
	threshold(img_gray, dstImage, 100, 255, CV_THRESH_BINARY);

	// imshow("thre",dstImage);

	vector<vector<Point> > contours;
	vector<Vec4i> hierarchy;
	// 注意这里的第四个参数，决定了检测的准确性
	findContours(dstImage, contours, hierarchy, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE, Point(0, 0));

	RNG rng(0);
	for (int i = 0; i < contours.size(); i++)
	{
		Scalar color = Scalar(rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255));
		Rect rect = boundingRect(contours[i]);//检测外轮廓
	    // rectangle(src, rect, Scalar(0, 0, 255), 3);//对外轮廓加矩形框
		Mat dst = src(rect);
		// 预测函数
		predict(dst);
		// imwrite("../shacun/" + to_string(i) + ".png", temp);
	}
	// imshow("src",src);
	// waitKey(0);
}


//相似性函数
float isSimilar(Mat &src, Mat &dst)
{
	// 先定义两个图片的矩阵
	Mat src_copy, dst_copy;
	// resize(dst, src.row, src.col);
	// 讲两个图片缩放到src的尺寸上，并且用新的矩阵来储存
	resize(src, src_copy, src.size(), 0, 0, INTER_LINEAR);
	resize(dst, dst_copy, src.size(), 0, 0, INTER_LINEAR);

	float same = 0, diff = 0;
	// 两层for循环访问所有的元素点
	for (int i = 0; i < src_copy.rows; i++)
	{
		for (int j = 0; j < src_copy.cols; j++)
		{
			// 如果像素相同，那么same加1
			// 这里的代码明明是学习笔记上的，要好好看啊
			if (src_copy.at<uchar>(i, j) == dst_copy.at<uchar>(i, j))
				same += 1;
			// 如果像素不同，那么diff加1
			else
				diff += 1;
		}
	}
	// 返回相似度
	return same / (same + diff);
}


int main()
{
	// init初始化函数加载模板图片
	init();
	//调用轮廓检测函数
	srcImage = imread("digits_4.jpg");
	if (!srcImage.data)
	{
		printf("Oh, no, srcImage is error");
		return 0;
	}
	// 轮廓检测并保存数据
	isContours(srcImage);
	return 0;
}
```

```c++
////基于OpenCV的数字识别
//#include <iostream>
//#include <opencv2/opencv.hpp>
//#include <vector>
//using namespace std;
//using namespace cv;
//
//Mat srcImage, dstImage;
//Mat imgROI;
//
//int main()
//{
//	srcImage = imread("digits_4.jpg");
//	if (!srcImage.data)
//	{
//		printf("Oh, no, srcImage is error");
//		return false;
//	}
//
//	imshow("srcImage", srcImage);
//
//	cvtColor(srcImage, srcImage, CV_BGR2GRAY);
//
//	threshold(srcImage, dstImage, 100, 255, CV_THRESH_BINARY);
//
//	/*Mat kernel_erode = getStructuringElement(MORPH_RECT, Size(3, 3));
//	Mat kernel_dilate = getStructuringElement(MORPH_RECT, Size(15, 15));
//
//	erode(dstImage, dstImage, kernel_erode);
//	dilate(dstImage, dstImage, kernel_dilate);*/
//
//
//	vector<vector<Point> > contours;
//	vector<Vec4i> hierarchy;
//	findContours(dstImage, contours, RETR_TREE, CHAIN_APPROX_SIMPLE, Point(0, 0));
//
//	RNG rng(0);
//	for (int i = 0; i < contours.size(); i++)
//	{
//		Scalar color = Scalar(rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255));
//		drawContours(dstImage, contours, i, color, 2, 8, hierarchy, 0, Point(0, 0));
//		Rect rect = boundingRect(contours[i]);
//
//		//imgROI = dstImage(Rect(rect.x, rect.y, rect.width, rect.height ));
//
//		//imwrite("imgROI" + to_string(i) + ".png", imgROI);
//		cout << " the area is " << contourArea(contours[i]) << endl;
//		cout << "contours" << i << " height =" << rect.height << " width = " << rect.width << " rate = " << ((float)rect.width / rect.height) << endl;
//		Mat temp = srcImage(rect);
//		// imshow("tmp ",temp);
//		// cout << " temp shape is " << rect.x << " " << rect.y << " " << endl;
//
//		if(rect.height >= 60 && rect.width >= 30)
//		{
//			imwrite("digits" + to_string(i) + ".png", temp);
//		}
//		
//		rectangle(dstImage, rect, Scalar(0, 0, 255), 3);//对外轮廓加矩形框
//
//	}
//
//	imshow("dstImage", dstImage);
//	
//
//	waitKey(0);
//
//	return 0;
//}


#include <iostream>
#include <opencv2/opencv.hpp>
#include <vector>
using namespace std;
using namespace cv;

Mat srcImage, dstImage;
Mat dst;

int main()
{
	srcImage = imread("digits_3.png");
	Mat img_gray;
	cvtColor(srcImage, img_gray, CV_BGR2GRAY);

	if (!srcImage.data)
	{
		printf("Oh, no, srcImage is error");
		return 0;
	}

	namedWindow("gray", CV_WINDOW_AUTOSIZE);
	imshow("gray", img_gray);

	// 注意在取阈值的时候，最后一个参数
	threshold(img_gray, dstImage, 100, 255, CV_THRESH_BINARY);
	//threshold(img_gray, dstImage, 100, 255, CV_THRESH_BINARY_INV);



	namedWindow("Threshold", CV_WINDOW_AUTOSIZE);
	imshow("Threshold", dstImage);

	vector<vector<Point> > contours;
	vector<Vec4i> hierarchy;
	// 注意这里的第四个参数，决定了检测的准确性
	findContours(dstImage, contours, hierarchy, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE, Point(0, 0));

	// imshow("dstImage", dstImage);
	cout << srcImage.rows << " " << srcImage.cols;
	RNG rng(0);
	for (int i = 0; i < contours.size(); i++)
	{
		cout << " i is " << i << endl;
		Scalar color = Scalar(rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255));
		// drawContours(srcImage, contours, i, color, 2, 8, hierarchy, 0, Point(0,0));
		Rect rect = boundingRect(contours[i]);//检测外轮廓
		cout << " the area is " << contourArea(contours[i]) << endl;
		cout << "contours" << i << "height=" << rect.height << "width =" << rect.width << "rate =" << ((float)rect.width / rect.height) << endl;
		Mat temp = srcImage(rect);
		// imshow("tmp ",temp);
		// cout << " temp shape is " << rect.x << " " << rect.y << " " << endl;
		// if(rect.height >= 60 && rect.width >= 30)
		imwrite("tutu" + to_string(i) + ".png", temp);
		rectangle(srcImage, rect, Scalar(0, 0, 255), 3);//对外轮廓加矩形框
	}

	namedWindow("output", CV_WINDOW_AUTOSIZE);
	imshow("output", srcImage);

	waitKey(0);

	return 0;
}
```


