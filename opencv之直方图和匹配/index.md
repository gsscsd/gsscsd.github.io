# OpenCV之直方图和匹配


> 模版匹配

<!--more-->

### 模版匹配

```C++
//模板匹配
#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
using namespace std;
using namespace cv;

int main()
{
	Mat img, temp, result;

	img = imread("test_2.jpg");
	temp = imread("test_1.png");

	int result_cols = img.cols - temp.cols + 1;
	int result_rows = img.rows - temp.rows + 1;
	result.create(result_cols, result_rows, CV_32FC1);

	//使用的匹配算法是标准平方差匹配 method = CV_TM_SQDIFF_NORMED, 数值越小匹配度越好
	matchTemplate(img, temp, result, CV_TM_SQDIFF_NORMED);
	normalize(result, result, 0, 1, NORM_MINMAX, -1, Mat());

	double minVal = -1;
	double maxVal;
	Point minLoc;
	Point maxLoc;
	Point matchLoc;
	cout << "匹配度：" << minVal << endl;
	minMaxLoc(result, &minVal, &maxVal, &minLoc, &maxLoc, Mat());

	cout << "匹配度：" << minVal << endl;

	matchLoc = minLoc;

	rectangle(img, matchLoc, Point(matchLoc.x + temp.cols, matchLoc.y + temp.rows), Scalar(0, 255, 0), 2, 8, 0);

	imshow("img", img);

	waitKey(0);


	return 0;
}
```

```C++
//模板匹配
#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
using namespace std;
using namespace cv;

Mat img, temp, result;
int MatchMethod;
int MaxTrackbarNum = 5;

void on_matching(int ,void *)
{
	Mat srcImage;
	img.copyTo(srcImage);

	int result_cols = img.cols - temp.cols + 1;
	int result_rows = img.rows - temp.rows + 1;
	result.create(result_cols, result_rows, CV_32FC1);

	//使用的匹配算法是标准平方差匹配 method = CV_TM_SQDIFF_NORMED, 数值越小匹配度越好
	matchTemplate(img, temp, result, MatchMethod);
	normalize(result, result, 0, 1, NORM_MINMAX, -1, Mat());

	double minVal, maxVal;
	Point minLoc, maxLoc, matchLoc;
	minMaxLoc(result, &minVal, &maxVal, &minLoc, &maxLoc);

	if (MatchMethod == TM_SQDIFF || MatchMethod == CV_TM_SQDIFF_NORMED)
	{
		matchLoc = minLoc;
	}
	else
	{
		matchLoc = maxLoc;
	}

	rectangle(srcImage, matchLoc, Point(matchLoc.x + temp.cols, matchLoc.y + temp.rows),
		     Scalar(0, 255, 0), 2, 8, 0);
	rectangle(img, matchLoc, Point(matchLoc.x + temp.cols, matchLoc.y + temp.rows),
		Scalar(0, 255, 0), 2, 8, 0);

	imshow("srcImage", srcImage);
	imshow("img", img);

}

int main()
{
	img = imread("test_2.jpg");
	temp = imread("test_1.png");
	if (!img.data)
	{
		printf("img is error");
		return -1;
	}
	if (!temp.data)
	{
		printf("temp is error");
		return -1;
	}
	
	namedWindow("img", CV_WINDOW_AUTOSIZE);
	createTrackbar("bar", "img", &MatchMethod, MaxTrackbarNum, on_matching);
	on_matching(0, NULL);

	waitKey(0);
	return 0;
}
```
