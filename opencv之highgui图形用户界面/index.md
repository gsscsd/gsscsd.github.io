# OpenCV之HighGUI图形用户界面初步


> 滑动条的创建使用

<!--more-->

### 滑动条的创建使用

```c++
#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
using namespace std;
using namespace cv;


//全局函数声明
Mat img;
int threshval = 160;   //轨迹条滑块对应的值，给初值160

//
//描述：轨迹条的回调函数
//
static void on_trackbar(int, void*)
{
    Mat bw = threshval < 128 ? (img < threshval):(img > threshval);
	
    //定义点和向量
    vector<vector<Point>> contours;
    vector<Vec4i> hierarchy;

    //查找轮廓
    findContours(bw, contours, hierarchy, CV_RETR_CCOMP, CV_CHAIN_APPROX_SIMPLE);	
	//初始化dst
	Mat dst = Mat::zeros(img.size(), CV_8UC3);
	//开始处理
	if(!contours.empty() && !hierarchy.empty())
	{
	    //遍历所有顶层轮廓，随机生成颜色值绘制个连接组成部分
		int idx = 0;
		for( ; idx >= 0; idx = hierarchy[idx][0])
		{
		    Scalar color((rand()&255), (rand()&255), (rand()&255));
			//绘制填充轮廓
			drawContours(dst, contours, idx, color, CV_FILLED, 8, hierarchy);
		 
		}
	}
	
	//显示窗口
	imshow("Connected Components", dst);
}


int main()
{
   img = imread("img.jpg");
   if(!img.data)
   {
     printf("Oh, no, img失败了 \n");
	 return -1;
   }
   
   //显示原图
   namedWindow("Image", 1);
   imshow("Image", img);
   
   //创建处理窗口
   namedWindow("Connected Components", 1);
   //创建轨迹条
   createTrackbar("Threshold", "Connected Components", &threshval, 255, on_trackbar);
   on_trackbar(threshval, 0); //轨迹条回调函数
   
   waitKey(0);
    
   return 0;
}
```


