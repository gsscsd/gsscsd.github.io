# OpenCV之feature2d组件


> 角点检测
>
> 特征检测与匹配

<!--more-->

### 角点检测

#### Harris角点检测

```c++
#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
using namespace std;
using namespace cv;

int main()
{
	//以灰度模式载入图像并显示
	Mat srcImage = imread("4.jpg", 0);
	imshow("srcImage", srcImage);

	//进行Harris角点检测找出角点
	Mat cornerStrength;
	cornerHarris(srcImage, cornerStrength, 2, 3, 0.01);

	//对灰度图进行阈值操作，得到二值图并显示
	Mat harrisCorner;
	threshold(cornerStrength, harrisCorner, 0.00001, 255, THRESH_BINARY);
	imshow("harrisCorner", harrisCorner);

	waitKey(0);

	return 0;
}
 
```

```c++
#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
using namespace std;
using namespace cv;

#define WINDOW_NAME1 "dstImage"
#define WINDOW_NAME2 "scaledImage"

Mat srcImage, dstImage, grayImage;
int thresh = 30; //当前阈值
int max_thresh = 175; //最大阈值

static void on_CornerHarris(int , void *);
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
	imshow("srcImage", srcImage);

	dstImage = srcImage.clone();
	cvtColor(dstImage, grayImage, CV_BGR2GRAY);

	//imshow("grayImage", grayImage);

	//创建窗口和滚动条
	namedWindow(WINDOW_NAME1, CV_WINDOW_AUTOSIZE);
	createTrackbar("CornerHarris", WINDOW_NAME1, &thresh, max_thresh, on_CornerHarris);
	on_CornerHarris(0, 0);

	waitKey(0);
	return 0;
}

void Show()
{
	printf("\n\n\n\t\t\t【欢迎来到Harris角点检测示例程序~】\n\n");
	printf("\n\n\n\t请调整滚动条观察图像效果\n\n");
	printf("\n\n\t\t\t\t\t\t\t\t\t\t by 晴宝");
}

static void on_CornerHarris(int, void *)
{
	//1.定义一些局部变量
	Mat n_dstImage;//目标图
	Mat normImage;//归一化后的图
	Mat scaledImage;//线性变换后的八位无符号整型的图

	//2.初始化
	//置零当前需要显示的两幅图，即清除上一次调用此函数时他们的值
	n_dstImage = Mat::zeros(srcImage.size(), CV_32FC1);
	dstImage = srcImage.clone();

	//3.正式检测
	//进行角点检测
	cornerHarris(grayImage, n_dstImage, 2, 3, 0.04, BORDER_DEFAULT);
	//归一化与转换
	normalize(n_dstImage, normImage, 0, 255, NORM_MINMAX, CV_32FC1, Mat());
	//将归一化的图像线性变换成8位无符号整型
	convertScaleAbs(normImage, scaledImage);

	//4.进行绘制
	//将检测到的，且符合阈值条件的角点绘制出来
	for (int j = 0; j < normImage.rows; j++)
	{
		for (int i = 0; i < normImage.cols; i++)
		{
			if ((int) normImage.at<float>(j, i) > thresh + 80)
			{
				circle(dstImage, Point(i, j), 5, Scalar(10, 10, 255), 2, 8, 0);
				circle(scaledImage, Point(i, j), 5, Scalar(0, 10, 255), 2, 8, 0);
			}
			

		}
	}

	imshow(WINDOW_NAME1, dstImage);
	imshow(WINDOW_NAME2, scaledImage);
}
```



### 特征检测与匹配

#### SURF特征点检测

```c++
//SURF特征点检测
/*
  1.使用FeatureDetector接口发现兴趣点
  2.使用SurFeatureDetector以及其函数detect来实现检测过程
  3.使用函数drawKeypoints绘制检测到的关键点
*/
#include <iostream>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <opencv2/core/core.hpp>
using namespace cv;
using namespace std;
using namespace cv::xfeatures2d;

void Show();

int main()
{
	Show();
	Mat srcImage1 = imread("4.jpg");
	Mat srcImage2 = imread("5.jpg");
	if (!srcImage1.data || !srcImage2.data)
	{
		printf("Oh, no, srcImage is error");
		return false;
	}

	imshow("srcImage1", srcImage1);
	imshow("srcImage2", srcImage2);

	//定义需要用到的变量和类
	int minHessian = 400; //定义SURF中的hessian阈值特征值检测算子
	//SurfFeatureDetector detector(minHessian);//定义一个SurfFeatureDetector（SURF）特征检测对象
    Ptr<SURF> surfDetector = SURF::create(2000);
	vector<KeyPoint> keypoints_1, keypoints_2;//vector模板类是能够存放任意类型的动态数组，能够增加和压缩数据

	//调用detect函数检测出SURF特征关键点，保存在vector容器中
	//detector.detect(srcImage1, keypoints_1);
	//detector.detect(srcImage2, keypoints_2);

    surfDetector->detect(srcImage1, keypoints_1);
    surfDetector->detect(srcImage2, keypoints_2);

	//绘制特征关键点
	Mat img_keypoints_1, img_keypoints_2;
	drawKeypoints(srcImage1, keypoints_1, img_keypoints_1, Scalar::all(-1), DrawMatchesFlags::DEFAULT);
	drawKeypoints(srcImage2, keypoints_2, img_keypoints_2, Scalar::all(-1), DrawMatchesFlags::DEFAULT);

	imshow("img_keypoints_1", img_keypoints_1);
	imshow("img_keypoints_2", img_keypoints_2);

	waitKey(0);
	return 0;
}

void Show()
{
	printf("\n\n\n\t欢迎来到【SURF特征检测】示例程序~\n\n");
	printf("\n\n\t按键操作说明：\n\n"
		"\t\t键盘按键任意键 - 退出程序\n\n"
		"\n\n\t\t\t\t\t\t\t\t by 晴宝\n\n\n");

}
```

```c++
///*
//程序的核心思想：
//  1.使用DescriptorExtractor接口来寻找关键点对应的特征向量
//	2.使用SurfdescriptorExtractor以及它的函数compute来完成特定的计算
//	3.使用BruteForceMatcher来匹配特征向量
//	4.使用函数drawMatches来绘制检测到的匹配点
//*/
#include <iostream>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <opencv2/core/core.hpp>
using namespace cv;
using namespace std;
using namespace cv::xfeatures2d;

void Show();

int main()
{
	Show();

	Mat srcImage1 = imread("4.jpg");
	Mat srcImage2 = imread("5.jpg");
	if (!srcImage1.data || !srcImage2.data)
	{
		printf("Oh, no, srcImage is error");
		return -1;
	}

	//使用SURF算子检测关键点
	int minHessian = 700;
	//Ptr<SIFT> SiftDescriptor = SIFT::create();
	Ptr<SURF> surfDetector = SURF::create(2000); //海塞矩阵阈值，在这里调整精度，值越大点越少，越精准 
	vector<KeyPoint> keyPoints1, keyPoints2;

	//调用detect函数检测出SURF特征关键点，保存在vector容器中
	surfDetector->detect(srcImage1, keyPoints1);
	surfDetector->detect(srcImage2, keyPoints2);

	//计算描述符（特征向量）
	//特征点描述，为下边的特征点匹配做准备    
	Ptr<SURF> SurfDescriptor = SURF::create();
	Mat descriptors1, descriptors2;
	SurfDescriptor->compute(srcImage1, keyPoints1, descriptors1);
	SurfDescriptor->compute(srcImage2, keyPoints2, descriptors2);

	//使用BruteForce进行匹配
	//实例化一个匹配器
	//BruteForceMatcher<L2<float>> matcher;
	BFMatcher matcher;
	vector<DMatch> matches;
	//匹配两幅图中的描述子（descriptors）
	matcher.match(descriptors1, descriptors2, matches);

	//绘制从两个图像中匹配出的关键点
	Mat imgMatches;
	//进行绘制
	drawMatches(srcImage1, keyPoints1, srcImage2, keyPoints2, matches, imgMatches);

	imshow("imgMatches", imgMatches);
	waitKey(0);

	return 0;
}

void Show()
{
	printf("\n\n\n\t欢迎来到【SURF特征描述】示例程序~\n\n");
	printf("\n\n\t\t\t\t\t\t\t by 晴宝\n\n\n");
}
```

```c++
#include <iostream>
#include "opencv2/core.hpp"
#include "opencv2/core/utility.hpp"
#include "opencv2/core/ocl.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/features2d.hpp"
#include "opencv2/calib3d.hpp"
#include "opencv2/imgproc.hpp"
#include"opencv2/flann.hpp"
#include"opencv2/xfeatures2d.hpp"
#include"opencv2/ml.hpp"
using namespace cv;
using namespace std;
using namespace cv::xfeatures2d;
using namespace cv::ml;

int main()
{
	Mat a = imread("4.jpg", IMREAD_GRAYSCALE);    //读取灰度图像
	Mat b = imread("5.jpg", IMREAD_GRAYSCALE);

	Ptr<SURF> surf;      //创建方式和2中的不一样
	surf = SURF::create(800);

	BFMatcher matcher;
	Mat c, d;
	vector<KeyPoint>key1, key2;
	vector<DMatch> matches;

	surf->detectAndCompute(a, Mat(), key1, c);
	surf->detectAndCompute(b, Mat(), key2, d);

	matcher.match(c, d, matches);       //匹配

	sort(matches.begin(), matches.end());  //筛选匹配点
	vector< DMatch > good_matches;
	int ptsPairs = std::min(50, (int)(matches.size() * 0.15));
	cout << ptsPairs << endl;
	for (int i = 0; i < ptsPairs; i++)
	{
		good_matches.push_back(matches[i]);
	}
	Mat outimg;
	drawMatches(a, key1, b, key2, good_matches, outimg, Scalar::all(-1), Scalar::all(-1), vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);  //绘制匹配点

	std::vector<Point2f> obj;
	std::vector<Point2f> scene;

	for (size_t i = 0; i < good_matches.size(); i++)
	{
		obj.push_back(key1[good_matches[i].queryIdx].pt);
		scene.push_back(key2[good_matches[i].trainIdx].pt);
	}

	std::vector<Point2f> obj_corners(4);
	obj_corners[0] = Point(0, 0);
	obj_corners[1] = Point(a.cols, 0);
	obj_corners[2] = Point(a.cols, a.rows);
	obj_corners[3] = Point(0, a.rows);
	std::vector<Point2f> scene_corners(4);

	Mat H = findHomography(obj, scene, RANSAC);      //寻找匹配的图像
	perspectiveTransform(obj_corners, scene_corners, H);

	line(outimg, scene_corners[0] + Point2f((float)a.cols, 0), scene_corners[1] + Point2f((float)a.cols, 0), Scalar(0, 255, 0), 2, LINE_AA);       //绘制
	line(outimg, scene_corners[1] + Point2f((float)a.cols, 0), scene_corners[2] + Point2f((float)a.cols, 0), Scalar(0, 255, 0), 2, LINE_AA);
	line(outimg, scene_corners[2] + Point2f((float)a.cols, 0), scene_corners[3] + Point2f((float)a.cols, 0), Scalar(0, 255, 0), 2, LINE_AA);
	line(outimg, scene_corners[3] + Point2f((float)a.cols, 0), scene_corners[0] + Point2f((float)a.cols, 0), Scalar(0, 255, 0), 2, LINE_AA);
	imshow("aaaa", outimg);
	cvWaitKey(0);
}
```


