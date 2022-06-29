# OpenCV计算机视觉编程攻略


> Chapter_01 : 基础知识
>
> Chapter_02 : 操作像素
>
> Chapter_03 : 处理图像的颜色
>
> Chapter_04 : 用直方图统计像素
>
> Chapter_05 : 用形态学运算变换图像
>
> Chapter_06  :  图像滤波
>
> Chapter_07 : 提取直线、轮廓和区域
>
> Chapter_08 : 检测兴趣点
>
> Chapter_09 : 描述和匹配兴趣点
>
> Chapter_10 : 估算图像之间的投影关系
>
> Chapter_11  :   三维重建
>
> Chapter_12 : 处理视频序列
>
> Chapter_13  :   跟踪运动物体
>
> Chapter_14  :    实用案列

<!--more-->

### Chapter_01 : 基础知识

> 1.图像的水平变换(flip),添加文字（putText）,鼠标的触发事件
> 2.感兴趣区

```c++
#include <iostream>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

void onMouse(int event, int x, int y, int flags, void *param)
{
	//reinterpret_cast允许将任何指针转为任何其他指针类型，
	//也允许将任何整数类型转化为任何指针类型以及反向转换。
	Mat *im = reinterpret_cast<Mat *>(param);

	switch (event)
	{
	case EVENT_LBUTTONDOWN:
		cout << "at(" << x << "," << y << ") value is: " << static_cast<int>(im->at<uchar>(Point(x, y))) << endl;
		break;
	default:
		break;
	}
}

int main()
{
	//1.图像的水平变换(flip), 添加文字（putText）, 鼠标的触发事件
	/*Mat image;
	cout << "This image is " << image.rows << "x" << image.cols << "y" << endl;

	image = imread("./images/puppy.bmp", IMREAD_GRAYSCALE);
	if (image.empty())
	{
		cout << "Error reading image..." << endl;
		return 0;
	}

	imshow("image", image);

	cout << "This image is " << image.rows << "x" << image.cols << "y" << endl;
	cout << "This image has" << image.channels() << "channel(s)" << endl;

	setMouseCallback("image", onMouse, reinterpret_cast<void*>(&image));

	Mat result;
	flip(image, result, 1);

	imshow("flip", result);

	circle(image, Point(155, 100), 65, 0, 3);

	putText(image, "SHA_CUN", Point(40, 200), FONT_HERSHEY_PLAIN, 2.0, 255, 2);

	imshow("sha_cun",image);*/

	//2.感兴趣区域
	Mat image = imread("./images/puppy.bmp");
	Mat logo = imread("./images/smalllogo.png");

	imshow("image", image);
	imshow("logo", logo);
	cout << "logo : " << logo.channels() << endl;

	Mat imageROI(image, Rect(image.cols - logo.cols, image.rows - logo.rows, logo.cols, logo.rows));
	logo.copyTo(imageROI);

	imshow("Image_logo", image);

	image = imread("./images/puppy.bmp");
	imageROI = image(Rect(image.cols - logo.cols, image.rows - logo.rows, logo.cols, logo.rows));
	
	Mat mask(logo); //必须是灰度图，只复制值不为0的部分（0为黑色， 255为白色）
	logo.copyTo(imageROI, mask);
	imshow("image_mask", image);


	waitKey(0);
	return 0;
}

```

> 图像像素的遍历:.at()、迭代器、指针

```c++
//1.at()方法实现图像像素的遍历
#include <iostream>
#include <opencv2/opencv.hpp>
using namespace std;
using namespace cv;

int main()
{
  Mat grayim(600, 800, CV_8UC1);
  Mat colorim(600, 800, CV_8UC3);
  
  for(int i = 0; i < grayim.rows; i++)
  {
    for(int j = 0; j < grayim.cols; j++)
	{
	  grayim.at<uchar>(i,j) = (i + j) % 255;
	}
  }

  for(int i = 0; i < colorim.rows; i++)
  {
     for(int j = 0; j < colorim.cols; j++)
	 {
	   Vec3b pixel;
	   pixel[0] = i % 255; //Blue
	   pixel[1] = i % 255; //Green
	   pixel[2] = 0;       //Red
	   
	   colorim.at<Vec3b>(i, j) = pixel;
	 }
  }
  
   imshow("grayim", grayim);
   imshow("colorim", colorim);

   waitKey(0);
   return 0;
}

//2.简单的二值化处理代码
//如果，我们只需要图像的轮廓或者边缘的时候，那么其他像素是不是就可以认为噪声，
//为了减少数据量和方便计算，此时，我们就可以进行二值化
//而且二值化，还能去用来进行图像分割，前景后景分割
#include <iostream>
#include <opencv2/opencv.hpp>
using namespace std;
using namespace cv;

int main()
{
  Mat srcImage;
  srcImage = imread("4.jpg");
  imshow("srcImage1",srcImage);
  
  for(int i = 0; i < srcImage.rows; i++)
  {
    for(int j = 0; j < srcImage.cols; j++)
	{
	   Vec3b value = srcImage.at<Vec3b>(i,j);
	   
	   for(int h = 0; h < 3; h++)
	   {
	    if(value[h] > 128)
		   value[h] = 255;
		else
		   value[h] = 0;
	   }
	   
	   
	}
	
	srcImage.at<Vec3b>(i, j) = value;
  }
	
  imshow("srcImage2", srcImage);

  waitKey(0);

  return 0;
}
//3.使用迭代器遍历矩阵
#include <iostream>
#include <opencv2/opencv.hpp>
using namespace std;
using namespace cv;

int main()
{
	Mat grayim(800, 600, CV_8UC1);
	Mat colorim(800, 600, CV_8UC3);

	MatIterator_<uchar> grayit, grayend;
	for (grayit = grayim.begin<uchar>(), grayend = grayim.end<uchar>(); 
		grayit != grayend; grayit++)
		*grayit = rand() % 255;

	MatIterator_<Vec3b> colorit, colorend;
	for (colorit = colorim.begin<Vec3b>(), colorend = colorim.end<Vec3b>();
		colorit != colorend; colorit++)
		(*colorit)[0] = rand() % 255; //Blue
	(*colorit)[1] = rand() % 255; //Green
	(*colorit)[2] = rand() % 255; //Red

	imshow("grayim", grayim);
	imshow("colorim", colorim);

	waitKey(0);
	return 0;
}
//4.使用指针来遍历
#include <iostream>
#include <opencv2/opencv.hpp>
using namespace std;
using namespace cv;

int main()
{
	Mat grayim(800, 600, CV_8UC1);
	Mat colorim(800, 600, CV_8UC3);

	for (int i = 0; i < grayim.rows; i++)
	{
		uchar *p = grayim.ptr<uchar>(i);

		for (int j = 0; j < grayim.cols; j++)
		{
			p[j] = (i + j) % 255;
		}
	}

	for (int i = 0; i < colorim.rows; i++)
	{
		Vec3b *p = colorim.ptr<Vec3b>(i);

		for (int j = 0; j < colorim.cols; j++)
		{
			p[j][0] = i % 255;
			p[j][1] = j % 255;
			p[j][2] = 0;
		}
	}

	imshow("grayim", grayim);
	imshow("colorim", colorim);

	waitKey(0);
	return 0;
}

```



### Chapter_02 : 操作像素

> 1.创建椒盐噪声
> 2.创建波浪影响remap()  ---- 重映射
> 3.扫描图像并访问相邻像素（1、代码的运行时间 2、锐化）
> 4.增加图片
> 5.减少图片中颜色的数量(没敲，太麻烦

<!--more-->

```c++

#include <iostream>
#include <opencv2/opencv.hpp>
//5.减少图片中颜色的数量
#define NTESTS 15
#define NITERATIONS 10

using namespace std;
using namespace cv;

//2.创建波浪影响remap()
void wave( const Mat &image, Mat &result)
{
	//the map functiuons
	Mat srcX(image.rows, image.cols, CV_32F);
	Mat srcY(image.rows, image.cols, CV_32F);

	//creating the mapping
	for (int i = 0; i < image.rows; i++)
	{
		for (int j = 0; j < image.cols; j++)
		{
			srcX.at<float>(i, j) = j;
			srcY.at<float>(i, j) = i + 3 * sin(j / 6.0);

			// horizontal flipping
			// srcX.at<float>(i,j)= image.cols-j-1;
			// srcY.at<float>(i,j)= i;
		}
	}

	//applying the mapping
	remap(image, result, srcX, srcY, INTER_LINEAR);
}

//3.扫描图像并访问相邻像素
void sharpen(const Mat &image, Mat &result)
{
	result.create(image.size(), image.type());
	int nchannels = image.channels();
	//处理所有的行，（除了第一行和最后一行）
	for (int j = 1; j < image.rows - 1; j++)
	{
		const uchar *previous = image.ptr<const uchar>(j - 1);  //previous row
		const uchar *current = image.ptr<const uchar>(j);       //current row
		const uchar *next = image.ptr<const uchar>(j + 1);      //next row

		uchar *output = result.ptr<uchar>(j);                   //output row

		for (int i = nchannels; i < (image.cols - 1) * nchannels; i++)
		{
			//应用锐化算子
			*output++ = saturate_cast<uchar>(5 * current[i]   -   current[i-nchannels]
				                             - current[i+nchannels] - previous[i] - next[i]);
		}
	}

	//把未处理的像素设置为0
	result.row(0).setTo(Scalar(0));
	result.row(result.rows - 1).setTo(Scalar(0));
	result.col(0).setTo(Scalar(0));
	result.col(result.cols - 1).setTo(Scalar(0));
}

//和sharpen一样的功能，只不过用了迭代
void sharpenIterator(const Mat &image, Mat &result)
{
	//must be a gray-level image
	CV_Assert(image.type() == CV_8UC1);

	//initialize iterator at row 1
	Mat_<uchar> :: const_iterator it = image.begin<uchar>() + image.cols;
	Mat_<uchar> :: const_iterator itend = image.end<uchar>() - image.cols;
	Mat_<uchar> :: const_iterator itup = image.begin<uchar>();
	Mat_<uchar> ::const_iterator itdown = image.begin<uchar>() + 2 * image.cols;

	//setup output image and iterator
	result.create(image.size(), image.type());   //allocate if necessary
	Mat_<uchar> ::iterator itout = result.begin<uchar>() + result.cols;

	for (; it != itend; ++it, ++itout, ++itup, ++itdown)
	{
		*itout = cv::saturate_cast<uchar>(*it * 5 - *(it - 1) - *(it + 1) - *itup - *itdown);
	}

	// Set the unprocessed pixels to 0
	result.row(0).setTo(cv::Scalar(0));
	result.row(result.rows - 1).setTo(cv::Scalar(0));
	result.col(0).setTo(cv::Scalar(0));
	result.col(result.cols - 1).setTo(cv::Scalar(0));
}

void sharpen2D(const Mat &image, Mat &result)
{
	//构造内核（所有入口都初始化为0）
	Mat kernel(3, 3, CV_32F, Scalar(0));
	//对内核赋值
	kernel.at<float>(1, 1) = 5.0;
	kernel.at<float>(0, 1) = -1.0;
	kernel.at<float>(2, 1) = -1.0;
	kernel.at<float>(1, 0) = -1.0;
	kernel.at<float>(1, 2) = -1.0;

	//对图像滤波
	filter2D(image, result, image.depth(), kernel);
}

//5.减少图片中颜色的数量
// 1st version
// see recipe Scanning an image with pointers
void colorReduce(Mat image, int div = 64) {

	int nl = image.rows; // number of lines
	int nc = image.cols * image.channels(); // total number of elements per line

	for (int j = 0; j<nl; j++) {

		// get the address of row j
		uchar* data = image.ptr<uchar>(j);

		for (int i = 0; i<nc; i++) {

			// process each pixel ---------------------

			data[i] = data[i] / div*div + div / 2;

			// end of pixel processing ----------------

		} // end of line
	}
}

// version with input/ouput images
// see recipe Scanning an image with pointers
void colorReduceIO(const cv::Mat &image, // input image
	cv::Mat &result,      // output image
	int div = 64) {

	int nl = image.rows; // number of lines
	int nc = image.cols; // number of columns
	int nchannels = image.channels(); // number of channels

									  // allocate output image if necessary
	result.create(image.rows, image.cols, image.type());

	for (int j = 0; j<nl; j++) {

		// get the addresses of input and output row j
		const uchar* data_in = image.ptr<uchar>(j);
		uchar* data_out = result.ptr<uchar>(j);

		for (int i = 0; i<nc*nchannels; i++) {

			// process each pixel ---------------------

			data_out[i] = data_in[i] / div*div + div / 2;

			// end of pixel processing ----------------

		} // end of line
	}
}

// Test 1
// this version uses the dereference operator *
void colorReduce1(cv::Mat image, int div = 64) {

	int nl = image.rows; // number of lines
	int nc = image.cols * image.channels(); // total number of elements per line
	uchar div2 = div >> 1; // div2 = div/2

	for (int j = 0; j<nl; j++) {

		uchar* data = image.ptr<uchar>(j);

		for (int i = 0; i<nc; i++) {


			// process each pixel ---------------------

			*data++ = *data / div*div + div2;

			// end of pixel processing ----------------

		} // end of line
	}
}

// Test 2
// this version uses the modulo operator
void colorReduce2(cv::Mat image, int div = 64) {

	int nl = image.rows; // number of lines
	int nc = image.cols * image.channels(); // total number of elements per line
	uchar div2 = div >> 1; // div2 = div/2

	for (int j = 0; j<nl; j++) {

		uchar* data = image.ptr<uchar>(j);

		for (int i = 0; i<nc; i++) {

			// process each pixel ---------------------

			int v = *data;
			*data++ = v - v%div + div2;

			// end of pixel processing ----------------

		} // end of line
	}
}

// Test 3
// this version uses a binary mask
void colorReduce3(cv::Mat image, int div = 64) {

	int nl = image.rows; // number of lines
	int nc = image.cols * image.channels(); // total number of elements per line
	int n = static_cast<int>(log(static_cast<double>(div)) / log(2.0) + 0.5);
	// mask used to round the pixel value
	uchar mask = 0xFF << n; // e.g. for div=16, mask= 0xF0
	uchar div2 = 1 << (n - 1); // div2 = div/2

	for (int j = 0; j<nl; j++) {

		uchar* data = image.ptr<uchar>(j);

		for (int i = 0; i < nc; i++) {

			// process each pixel ---------------------

			*data &= mask;     // masking
			*data++ |= div2;   // add div/2

							   // end of pixel processing ----------------

		} // end of line
	}
}


// Test 4
// this version uses direct pointer arithmetic with a binary mask
void colorReduce4(cv::Mat image, int div = 64) {

	int nl = image.rows; // number of lines
	int nc = image.cols * image.channels(); // total number of elements per line
	int n = static_cast<int>(log(static_cast<double>(div)) / log(2.0) + 0.5);
	int step = image.step; // effective width
						   // mask used to round the pixel value
	uchar mask = 0xFF << n; // e.g. for div=16, mask= 0xF0
	uchar div2 = div >> 1; // div2 = div/2

						   // get the pointer to the image buffer
	uchar *data = image.data;

	for (int j = 0; j<nl; j++) {

		for (int i = 0; i<nc; i++) {

			// process each pixel ---------------------

			*(data + i) &= mask;
			*(data + i) += div2;

			// end of pixel processing ----------------

		} // end of line

		data += step;  // next line
	}
}

// Test 5
// this version recomputes row size each time
void colorReduce5(cv::Mat image, int div = 64) {

	int nl = image.rows; // number of lines
	int n = static_cast<int>(log(static_cast<double>(div)) / log(2.0) + 0.5);
	// mask used to round the pixel value
	uchar mask = 0xFF << n; // e.g. for div=16, mask= 0xF0

	for (int j = 0; j<nl; j++) {

		uchar* data = image.ptr<uchar>(j);

		for (int i = 0; i<image.cols * image.channels(); i++) {

			// process each pixel ---------------------

			*data &= mask;
			*data++ += div / 2;

			// end of pixel processing ----------------

		} // end of line
	}
}

// Test 6
// this version optimizes the case of continuous image
void colorReduce6(cv::Mat image, int div = 64) {

	int nl = image.rows; // number of lines
	int nc = image.cols * image.channels(); // total number of elements per line

	if (image.isContinuous()) {
		// then no padded pixels
		nc = nc*nl;
		nl = 1;  // it is now a 1D array
	}

	int n = static_cast<int>(log(static_cast<double>(div)) / log(2.0) + 0.5);
	// mask used to round the pixel value
	uchar mask = 0xFF << n; // e.g. for div=16, mask= 0xF0
	uchar div2 = div >> 1; // div2 = div/2

						   // this loop is executed only once
						   // in case of continuous images
	for (int j = 0; j<nl; j++) {

		uchar* data = image.ptr<uchar>(j);

		for (int i = 0; i<nc; i++) {

			// process each pixel ---------------------

			*data &= mask;
			*data++ += div2;

			// end of pixel processing ----------------

		} // end of line
	}
}

// Test 7
// this versions applies reshape on continuous image
void colorReduce7(cv::Mat image, int div = 64) {

	if (image.isContinuous()) {
		// no padded pixels
		image.reshape(1,   // new number of channels
			1); // new number of rows
	}
	// number of columns set accordingly

	int nl = image.rows; // number of lines
	int nc = image.cols*image.channels(); // number of columns

	int n = static_cast<int>(log(static_cast<double>(div)) / log(2.0) + 0.5);
	// mask used to round the pixel value
	uchar mask = 0xFF << n; // e.g. for div=16, mask= 0xF0
	uchar div2 = div >> 1; // div2 = div/2

	for (int j = 0; j<nl; j++) {

		uchar* data = image.ptr<uchar>(j);

		for (int i = 0; i<nc; i++) {

			// process each pixel ---------------------

			*data &= mask;
			*data++ += div2;

			// end of pixel processing ----------------

		} // end of line
	}
}

// Test 8
// this version processes the 3 channels inside the loop with Mat_ iterators
void colorReduce8(cv::Mat image, int div = 64) {

	// get iterators
	cv::Mat_<cv::Vec3b>::iterator it = image.begin<cv::Vec3b>();
	cv::Mat_<cv::Vec3b>::iterator itend = image.end<cv::Vec3b>();
	uchar div2 = div >> 1; // div2 = div/2

	for (; it != itend; ++it) {

		// process each pixel ---------------------

		(*it)[0] = (*it)[0] / div*div + div2;
		(*it)[1] = (*it)[1] / div*div + div2;
		(*it)[2] = (*it)[2] / div*div + div2;

		// end of pixel processing ----------------
	}
}

// Test 9
// this version uses iterators on Vec3b
void colorReduce9(cv::Mat image, int div = 64) {

	// get iterators
	cv::MatIterator_<cv::Vec3b> it = image.begin<cv::Vec3b>();
	cv::MatIterator_<cv::Vec3b> itend = image.end<cv::Vec3b>();

	const cv::Vec3b offset(div / 2, div / 2, div / 2);

	for (; it != itend; ++it) {

		// process each pixel ---------------------

		*it = *it / div*div + offset;
		// end of pixel processing ----------------
	}
}

// Test 10
// this version uses iterators with a binary mask
void colorReduce10(cv::Mat image, int div = 64) {

	// div must be a power of 2
	int n = static_cast<int>(log(static_cast<double>(div)) / log(2.0) + 0.5);
	// mask used to round the pixel value
	uchar mask = 0xFF << n; // e.g. for div=16, mask= 0xF0
	uchar div2 = div >> 1; // div2 = div/2

						   // get iterators
	cv::Mat_<cv::Vec3b>::iterator it = image.begin<cv::Vec3b>();
	cv::Mat_<cv::Vec3b>::iterator itend = image.end<cv::Vec3b>();

	// scan all pixels
	for (; it != itend; ++it) {

		// process each pixel ---------------------

		(*it)[0] &= mask;
		(*it)[0] += div2;
		(*it)[1] &= mask;
		(*it)[1] += div2;
		(*it)[2] &= mask;
		(*it)[2] += div2;

		// end of pixel processing ----------------
	}
}

// Test 11
// this versions uses ierators from Mat_ 
void colorReduce11(cv::Mat image, int div = 64) {

	// get iterators
	cv::Mat_<cv::Vec3b> cimage = image;
	cv::Mat_<cv::Vec3b>::iterator it = cimage.begin();
	cv::Mat_<cv::Vec3b>::iterator itend = cimage.end();
	uchar div2 = div >> 1; // div2 = div/2

	for (; it != itend; it++) {

		// process each pixel ---------------------

		(*it)[0] = (*it)[0] / div*div + div2;
		(*it)[1] = (*it)[1] / div*div + div2;
		(*it)[2] = (*it)[2] / div*div + div2;

		// end of pixel processing ----------------
	}
}


// Test 12
// this version uses the at method
void colorReduce12(cv::Mat image, int div = 64) {

	int nl = image.rows; // number of lines
	int nc = image.cols; // number of columns
	uchar div2 = div >> 1; // div2 = div/2

	for (int j = 0; j<nl; j++) {
		for (int i = 0; i<nc; i++) {

			// process each pixel ---------------------

			image.at<cv::Vec3b>(j, i)[0] = image.at<cv::Vec3b>(j, i)[0] / div*div + div2;
			image.at<cv::Vec3b>(j, i)[1] = image.at<cv::Vec3b>(j, i)[1] / div*div + div2;
			image.at<cv::Vec3b>(j, i)[2] = image.at<cv::Vec3b>(j, i)[2] / div*div + div2;

			// end of pixel processing ----------------

		} // end of line
	}
}


// Test 13
// this version uses Mat overloaded operators
void colorReduce13(cv::Mat image, int div = 64) {

	int n = static_cast<int>(log(static_cast<double>(div)) / log(2.0) + 0.5);
	// mask used to round the pixel value
	uchar mask = 0xFF << n; // e.g. for div=16, mask= 0xF0

							// perform color reduction
	image = (image&cv::Scalar(mask, mask, mask)) + cv::Scalar(div / 2, div / 2, div / 2);
}

// Test 14
// this version uses a look up table
void colorReduce14(cv::Mat image, int div = 64) {

	cv::Mat lookup(1, 256, CV_8U);

	for (int i = 0; i<256; i++) {

		lookup.at<uchar>(i) = i / div*div + div / 2;
	}

	cv::LUT(image, lookup, image);
}
       
int main()
{
	//2.创建波浪影响remap()  ---- 重映射
	/*Mat image = imread("./images/boldt.jpg", 0);

	imshow("image", image);

	Mat result;
	wave(image, result);

	imshow("result", result);*/

	////3.扫描图像并访问相邻像素
	//Mat image = imread("./images/boldt.jpg");
	//if (!image.data)
	//{
	//	return 0;
	//}

	//imshow("image", image);

	////调用sharpen()
	//Mat result_sharpen;
	////用来测试函数或者代码段的运行时间
	////getTickCount()返回从最近一次计算机开机到当前的时钟周期数
	//double time = static_cast<double>(getTickCount());
	//sharpen(image, result_sharpen);
	////getTickFrequency()返回每秒的时钟数
	//time = (static_cast<double>(getTickCount()) - time) / getTickFrequency();
	//cout << "time_sharpen = " << time << endl;

	//imshow("sharpen_Image", result_sharpen);

	////调用sharpenIterator(要使用灰度图)
	//Mat image3 = imread("./images/boldt.jpg", 0);
	//if (!image3.data)
	//{
	//	return 0;
	//}

	//imshow("image3", image3);
	//Mat result_Iterator;
	//double time3 = static_cast<double>(getTickCount());
	//sharpenIterator(image3, result_Iterator);
	//time3 = (static_cast<double>(getTickCount()) - time3) / getTickFrequency();
	//cout << "time_Iterator = " << time3 << endl;

	//imshow("Iterator_Image", result_Iterator);
	////调用sharpen2D()
	//Mat result_sharpen2D;

	//double time2 = static_cast<double>(getTickCount());
	//sharpen2D(image, result_sharpen2D);

	//time2 = (static_cast<double>(getTickCount()) - time2) / getTickFrequency();
	//cout << "time_sharpen2D = " << time2 << endl;

	//imshow("sharpen2D_Image", result_sharpen2D);

	//4.增加图片
	//Mat image1;
	//Mat image2;

	//image1 = imread("./images/boldt.jpg");
	//image2 = imread("./images/rain.jpg");
	//if (!image1.data)
	//	return 0;
	//if (!image2.data)
	//	return 0;

	//imshow("Image1", image1);
	//imshow("Image2", image2);

	//Mat result;

	//addWeighted(image1, 0.7, image2, 0.9, 0, result);

	//imshow("result", result);

	////using over loaded operator
	////result = 0.7 * image1 + 0.9 * image2;
	////imshow("result with operators", result);

	//image2 = imread("./images/rain.jpg", 0);
	////imshow("image2_rain", image2);
	//vector<Mat> planes;

	//split(image1, planes);
	////aadd to blue channel
	//planes[0] += image2;
	////merge the 3 1-channel images into 1 3-channel image
	//merge(planes, result);

	//imshow("Blue channel", result);

	//5.减少图片中颜色的数量

	Mat image = imread("./images/boldt.jpg");

	// time and process the image
	const int64 start = getTickCount();
	colorReduce(image, 64);
	//Elapsed time in seconds
	double duration = (getTickCount() - start) / getTickFrequency();

	// display the image
	cout << "Duration= " << duration << "secs" << std::endl;
	namedWindow("Image");
	imshow("Image", image);

	// test different versions of the function

	int64 t[NTESTS], tinit;
	// timer values set to 0
	for (int i = 0; i<NTESTS; i++)
		t[i] = 0;

	Mat images[NTESTS];
	Mat result;

	// the versions to be tested
	typedef void(*FunctionPointer)(Mat, int);
	FunctionPointer functions[NTESTS] = { colorReduce, colorReduce1, colorReduce2, colorReduce3, colorReduce4,
		colorReduce5, colorReduce6, colorReduce7, colorReduce8, colorReduce9,
		colorReduce10, colorReduce11, colorReduce12, colorReduce13, colorReduce14 };
	// repeat the tests several times
	int n = NITERATIONS;
	for (int k = 0; k<n; k++) {

		cout << k << " of " << n << std::endl;

		// test each version
		for (int c = 0; c < NTESTS; c++) {

			images[c] = imread("./images/boldt.jpg");

			// set timer and call function
			tinit = getTickCount();
			functions[c](images[c], 64);
			t[c] += getTickCount() - tinit;

			cout << ".";
		}

		cout << endl;
	}

	// short description of each function
	string descriptions[NTESTS] = {
		"original version:",
		"with dereference operator:",
		"using modulo operator:",
		"using a binary mask:",
		"direct ptr arithmetic:",
		"row size recomputation:",
		"continuous image:",
		"reshape continuous image:",
		"with iterators:",
		"Vec3b iterators:",
		"iterators and mask:",
		"iterators from Mat_:",
		"at method:",
		"overloaded operators:",
		"look-up table:",
	};

	for (int i = 0; i < NTESTS; i++) {

		namedWindow(descriptions[i]);
		imshow(descriptions[i], images[i]);
	}

	// print average execution time
	cout << endl << "-------------------------------------------" << endl << endl;
	for (int i = 0; i < NTESTS; i++) {

		cout << i << ". " << descriptions[i] << 1000.*t[i] / getTickFrequency() / n << "ms" << endl;
	}


	waitKey(0);
	return 0;
}

```

> 1.使用flip来进行水平垂直方向的改变
> 2.使用putText添加文本
> 3.使用椒盐噪声
> 4.减色函数clolrReduce

```c++

#include <iostream>
#include <opencv2/opencv.hpp>
#include <random>

using namespace std;
using namespace cv;

Mat srcImage;
Mat result_1, result_0, result;
Mat src_gray_img;

////3.椒盐噪声
//void salt(Mat image, int n)
//{
//	//C++11的随机生成器
//	default_random_engine generator;
//	uniform_int_distribution<int> randomRow(0, image.rows - 1);
//	uniform_int_distribution<int> randomCol(0, image.cols - 1);
//
//	int i, j;
//	for (int k = 0; k < n; k++)
//	{
//		//随机生成图形位置
//		i = randomCol(generator);
//		j = randomRow(generator);
//
//		if (image.type() == CV_8UC1) // 灰度图像
//		{
//			//单通道8位图像
//			image.at<uchar>(j, i) = 255;
//		}
//		else if (image.type() == CV_8UC3) //彩色图像
//		{
//			//3通道图像
//			image.at<Vec3b>(j, i)[0] = 255;
//			image.at<Vec3b>(j, i)[1] = 255;
//			image.at<Vec3b>(j, i)[2] = 255;
//		}
//	}
//}

//4.减色函数
void colorReduce(Mat image, int div = 64)
{
	int nr = image.rows; //行数
	//每一行的元素
	int nc = image.cols * image.channels();

	for (int j = 0; j < nr; j++)
	{
		//取的行j的地址
		uchar *data = image.ptr<uchar>(j);

		for (int i = 0; i < nc; i++)
		{
			//处理每一个元素
			data[i] = data[i] / div * div + div / 2;
		}
	}
}

int main()
{
	srcImage = imread("1.jpg");

	if (srcImage.empty())
	{
		cout << "srcImage is error !" << endl;
		return -1;
	}

	imshow("srcImage", srcImage);

	//3.调用椒盐噪声
	//salt(srcImage, 3000);
	//imshow("srcImage_salt", srcImage);

	//4.减色函数
	colorReduce(srcImage, 64);
	imshow("srcImage_colorreduce", srcImage);

	//src_gray_img = imread("1.jpg", IMREAD_GRAYSCALE);
	//imshow("src_grat_img", src_gray_img);

	//1.使用flip来进行水平垂直方向的改变
	//flip(srcImage, result_1, 1); //正数表示水平
	//flip(srcImage, result_0, 0); //0表示垂直
	//flip(srcImage, result, -1); //负数表示水平和垂直

	//2.使用putText添加文本
	//putText(result_1, "Wangdasha is ppp", Point(4, 200), FONT_HERSHEY_PLAIN, 2.0, 125, 2);

	//imshow("result_1", result_1);
	//imshow("result_0", result_0);
	//imshow("result", result);

	waitKey(0);

	return 0;
}
```



### Chapter_03 : 处理图像的颜色

> 1.用策略设计模式比较颜色
> 2.用GrabCut算法分割图像
> 3.用色调、饱和度、和亮度表示颜色

<!--more-->

```c++
#include <iostream>
#include <opencv2/opencv.hpp>

#include "ColorDetector.h"

using namespace std;
using namespace cv;

//3.用色调、饱和度、和亮度表示颜色 ---- 检测肤色
       //输入图像  色调区间  饱和度区间  输出掩码
void detectHScolor(const Mat& image, double minHue, double maxHue, double minSat, double maxSat, Mat &mask)
{
	//转到HSV空间
	Mat hsv;
	cvtColor(image, hsv, CV_BGR2HSV);
	//将3通道分割成3幅图像
	vector<Mat> channels;
	split(hsv, channels);
	//色调掩码
	Mat mask1;  //小于maxHue
	threshold(channels[0], mask1, maxHue, 255, THRESH_BINARY_INV);
	Mat mask2; //大于minHue
	threshold(channels[0], mask2, minHue, 255, THRESH_BINARY);

	Mat hueMask; //色调掩码
	if (minHue < maxHue)
		hueMask = mask1 & mask2;
	else //如果区间穿越0度中轴线
		hueMask = mask1 | mask2;

	//饱和度掩码
	//从minSat 到 maxSat
	//threshold(channels[1], mask1, maxSat, 255, THRESH_BINARY_INV);
	//threshold(channels[1], mask2, minSat, 255, THRESH_BINARY);

	//Mat satMask;
	//satMask = mask1 & mask2;

	Mat satMask;
	inRange(channels[1], minSat, maxSat, satMask);

	//组合掩码
	mask = hueMask & satMask;
}

int main()
{
	////1.用策略设计模式比较颜色

	////创建图像处理器对象
	//ColorDetector cdetect;

	////读取图像
	//Mat image = imread("./images/boldt.jpg");
	//if (image.empty())
	//	return 0;

	//imshow("image", image);

	////设置输入参数
	//cdetect.setTargetColor(230, 190, 130); // 这里表示蓝天

	////处理图像并显示结果
	//Mat result = cdetect.process(image);

	//imshow("result", result);

	//// or using functor
	//// here distance is measured with the Lab color space
	//ColorDetector colordetector(230, 190, 130,  // color
	//	45, true); // Lab threshold
	//namedWindow("result (functor)");
	//result = colordetector(image);
	//imshow("result (functor)", result);

	//// testing floodfill
	//floodFill(image,            // input/ouput image
	//	Point(100, 50),         // seed point
	//	Scalar(255, 255, 255),  // repainted color
	//	(Rect*)0,  // bounding rectangle of the repainted pixel set
	//	Scalar(35, 35, 35),     // low and high difference threshold
	//	Scalar(35, 35, 35),     // most of the time will be identical
	//	FLOODFILL_FIXED_RANGE); // pixels are compared to seed color

	//namedWindow("Flood Fill result");
	//result = colordetector(image);
	//imshow("Flood Fill result", image);

	//// Creating artificial images to demonstrate color space properties
	//Mat colors(100, 300, CV_8UC3, Scalar(100, 200, 150));
	//Mat range = colors.colRange(0, 100);
	//range = range + Scalar(10, 10, 10);
	//range = colors.colRange(200, 300);
	//range = range + Scalar(-10, -10, 10);

	//namedWindow("3 colors");
	//imshow("3 colors", colors);

	//Mat labImage(100, 300, CV_8UC3, Scalar(100, 200, 150));
	//cvtColor(labImage, labImage, CV_BGR2Lab);
	//range = colors.colRange(0, 100);
	//range = range + Scalar(10, 10, 10);
	//range = colors.colRange(200, 300);
	//range = range + Scalar(-10, -10, 10);
	//cvtColor(labImage, labImage, CV_Lab2BGR);

	//namedWindow("3 colors (Lab)");
	//imshow("3 colors (Lab)", colors);

	//// brightness versus luminance
	//Mat grayLevels(100, 256, CV_8UC3);
	//for (int i = 0; i < 256; i++) {
	//	grayLevels.col(i) = Scalar(i, i, i);
	//}

	//range = grayLevels.rowRange(50, 100);
	//Mat channels[3];
	//split(range, channels);
	//channels[1] = 128;
	//channels[2] = 128;
	//merge(channels, 3, range);
	//cvtColor(range, range, CV_Lab2BGR);


	//namedWindow("Luminance vs Brightness");
	//imshow("Luminance vs Brightness", grayLevels);


	//2.用GrabCut算法分割图像
	//Mat image = imread("./images/boldt.jpg");
	//if (image.empty())
	//	return 0;

	//imshow("image", image);

	//Rect rectangle(5, 70, 260, 120);

	//Mat result;
	//Mat bg, fg;
	////GrabCut分割算法
	//grabCut(image, result, rectangle, bg, fg, 5, GC_INIT_WITH_RECT);
	////取得标记可能为前景的像素
	//compare(result, GC_PR_FGD, result, CMP_EQ);
	////生成输出图像
	//Mat foreground(image.size(), CV_8UC3, Scalar(255, 255, 255));
	//image.copyTo(foreground, result);        //A.copyTo(B, mask);

	//imshow("foreground", foreground);
	//imshow("result", result);
	//imshow("image_copy", image);

	//3.用色调、饱和度、和亮度表示颜色
	Mat image = imread("./images/boldt.jpg");
	if (image.empty())
		return 0;

	imshow("image", image);

	//转换成HSV
	Mat hsv;
	cvtColor(image, hsv, CV_BGR2HSV);

	imshow("hsv", hsv);
	
	//把三个通道放进三个图片中
	vector<Mat> channels;
	split(hsv, channels); //0-H, 1-S, 2-V

	imshow("H", channels[0]); //色调
	imshow("S", channels[1]); //饱和度
	imshow("V", channels[2]); //亮度

	Mat newImage;
	Mat tmp(channels[2].clone());

	//改变亮度（V）
	channels[2] = 255;
	merge(channels, hsv);
	cvtColor(hsv, newImage, CV_HSV2BGR);

	imshow("newImage_V", newImage);
	//改变饱和度（S）
	channels[1] = 255;
	channels[2] = tmp;
	merge(channels, hsv);
	cvtColor(hsv, newImage, CV_HSV2BGR);

	imshow("newImage_S", newImage);
    //亮度、饱和度一起改变
	channels[1] = 255;
	channels[2] = 255;
	merge(channels, hsv);
	cvtColor(hsv, newImage, CV_HSV2BGR);

	imshow("newImage_SV", newImage);

	//展现出所有的HS颜色
	Mat hs(128, 360, CV_8UC3);

	for (int h = 0; h < 360; h++)
	{
		for (int s = 0; s < 128; s++)
		{
			hs.at<Vec3b>(s, h)[0] = h / 2;
			hs.at<Vec3b>(s, h)[1] = 255 - s * 2; //从高到低
			hs.at<Vec3b>(s, h)[2] = 255;
		}
	}

	cvtColor(hs, newImage, CV_HSV2BGR);
	imshow("newImage_hs", newImage);

	//肤色检测
	image = cv::imread("./images/girl.jpg");
	if (!image.data)
		return 0;

	imshow("girl", image);

	//检测肤色
	Mat mask;
	detectHScolor(image, 160, 10, 25, 166, mask); //色调为320度- 20度 饱和度为~0.1 到 0.65

	//显示使用掩码后的图像
	Mat detected(image.size(), CV_8UC3, Scalar(0, 0, 0));
	image.copyTo(detected, mask);

	imshow("detected", detected);




	// A test comparing luminance and brightness

	// create linear intensity image
	Mat linear(100, 256, CV_8U);
	for (int i = 0; i<256; i++) {

		linear.col(i) = i;
	}

	// create a Lab image
	linear.copyTo(channels[0]);
	Mat constante(100, 256, CV_8U, Scalar(128));
	constante.copyTo(channels[1]);
	constante.copyTo(channels[2]);
	merge(channels, image);

	// convert back to BGR
	Mat brightness;
	cvtColor(image, brightness, CV_Lab2BGR);
	split(brightness, channels);

	// create combined image
	Mat combined(200, 256, CV_8U);
	Mat half1(combined, Rect(0, 0, 256, 100));
	linear.copyTo(half1);
	Mat half2(combined, Rect(0, 100, 256, 100));
	channels[0].copyTo(half2);

	namedWindow("Luminance vs Brightness");
	imshow("Luminance vs Brightness", combined);


	waitKey(0);
	return 0;
}



#pragma once
#if !defined COLORDETECT
#define COLORDETECT

#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>

using namespace std;
using namespace cv;

class ColorDetector {
private:

	//允许的最小差距
	int maxDist;

	//目标颜色
	Vec3b target;

	//image containing color converted image
	Mat converted;
	bool useLab;

	//存储二值映射结果的图像
	Mat result;

public:
	//空的构造函数
	ColorDetector() : maxDist(100), target(0, 0, 0), useLab(false) {}

	ColorDetector(bool useLab) : maxDist(100), target(0, 0, 0), useLab(useLab) {}
	//另一只构造函数，使用目标颜色和颜色距离作为参数
	ColorDetector(uchar blue, uchar green, uchar red, int mxDist = 100, bool useLab = false) :
		maxDist(mxDist), useLab(useLab) {

		//target color 
		setTargetColor(blue, green, red);
	}

	//计算与目标颜色的差距
	int getDistanceToTargetColor(const Vec3b& color) const
	{
		return getColorDistance(color, target);
	}

	//计算两个颜色之间的城区距离
	int getColorDistance(const Vec3b& color1, const Vec3b &color2) const
	{
		return abs(color1[0] - color2[0]) + abs(color1[1] - color2[1]) + abs(color1[2] - color2[2]);
	}

	Mat process(const Mat &image);

	Mat operator() (const Mat &image) {

		Mat input;

		if (useLab)
		{
			cvtColor(image, input, CV_BGR2Lab);
		}
		else
		{
			input = image;
		}

		Mat output;
		//compute absolute difference with target color
		absdiff(input, Scalar(target), output);
		//split the channel into 3 images
		vector<Mat> images;
		split(output, images);
		//add the 3 channels (saturation might occurs here)
		output = images[0] + images[1] + images[2];
		//apply threshold
		threshold(output, output, maxDist, 255, THRESH_BINARY_INV);

		return output;
	}


	//Getters and setters

	//设置颜色差距的阈值
	//阈值必须是正数，否则就设置为0
	void setColorDistanceThreshold(int distance)
	{
		if (distance < 0)
			distance = 0;
		maxDist = distance;
	}

	//取得颜色差距的阈值
	int getColorDistanceThreshold() const
	{
		return maxDist;
	}

	//设置需要检测的颜色
	//given in BGR color space
	void setTargetColor(uchar blue, uchar green, uchar red)
	{
		//次序为BGR
		target = Vec3b(blue, green, red);

		if (useLab)
		{
			//Temporary 1-pixel image
			Mat tmp(1, 1, CV_8UC3);
			tmp.at<Vec3b>(0, 0) = Vec3b(blue, green, red);

			//Converting the target to Lab color space
			cvtColor(tmp, tmp, CV_BGR2Lab);

			target = tmp.at<Vec3b>(0, 0);
		}
	}

	//设置需要检测的颜色
	void setTargetColor(Vec3b color)
	{
		target = color;
	}

	//取得需要检测的颜色
	Vec3b getTargetColor() const
	{
		return target;
	}
};

#endif // !defined COLORDETECT





#include "ColorDetector.h"
#include <vector>

Mat ColorDetector::process(const Mat &image)
{
	//必要时重新分配二值映射
	//与输入图像的尺寸相同，不过是单通道
	result.create(image.size(), CV_8U);

	// Converting to Lab color space 
	if (useLab)
		cvtColor(image, converted, CV_BGR2Lab);

	// 取得迭代器
	Mat_<Vec3b>::const_iterator it = image.begin<Vec3b>();
	Mat_<Vec3b>::const_iterator itend = image.end<Vec3b>();
	Mat_<uchar>::iterator itout = result.begin<uchar>();

	// get the iterators of the converted image 
	if (useLab) {
		it = converted.begin<Vec3b>();
		itend = converted.end<Vec3b>();
	}

	// 对于每一个像素
	for (; it != itend; ++it, ++itout) {

		//比较与目标颜色的差距
		if (getDistanceToTargetColor(*it)<maxDist) {

			*itout = 255;

		}
		else {

			*itout = 0;
		}

		// end of pixel processing ----------------
	}

	return result;
}
```



### Chapter_04 : 用直方图统计像素

> 计算图像直方图

<!--more-->

```c++
//#include "Histogram1D.h"
//#include "ContentFinder.h"
//#include  "ColorHistogram.h"
//
//
//int main()
//{
//	//读入图像，并转成灰度图
//	Mat image = imread("./images/waves.jpg", 0);
//	if (!image.data)
//		return 0;
//
//	imshow("image", image);
//
//	Mat imageROI;
//	imageROI = image(Rect(216, 33, 24, 30)); //设置感兴趣区域
//
//	imshow("imageROI", imageROI);
//
//	//直方图对象
//	//计算直方图
//	Histogram1D h;
//	//计算直方图
//	Mat hist = h.getHistogram(imageROI);
//	imshow("Histogram1D", h.getHistogramImage(imageROI));
//
//	//创建内容搜寻器
//	ContentFinder finder;
//	//设置用来反向投影的直方图
//
//	// set histogram to be back-projected
//	finder.setHistogram(hist);
//	finder.setThreshold(-1.0f);
//
//	// Get back-projection
//	cv::Mat result1;
//	result1 = finder.find(image);
//
//	// Create negative image and display result
//	cv::Mat tmp;
//	result1.convertTo(tmp, CV_8U, -1.0, 255.0);
//	cv::namedWindow("Backprojection result");
//	cv::imshow("Backprojection result", tmp);
//
//	// Get binary back-projection
//	finder.setThreshold(0.12f);
//	result1 = finder.find(image);
//
//	// Draw a rectangle around the reference area
//	cv::rectangle(image, cv::Rect(216, 33, 24, 30), cv::Scalar(0, 0, 0));
//
//	// Display image
//	cv::namedWindow("Image");
//	cv::imshow("Image", image);
//
//	// Display result
//	cv::namedWindow("Detection Result");
//	cv::imshow("Detection Result", result1);
//	
//
//
//	//装载彩色图片
//	ColorHistogram hc;
//	Mat color = imread("./images/waves.jpg");
//
//	imshow("color", color);
//
//	//提取ROI
//	imageROI = color(Rect(0, 0, 100, 45));  //蓝色天空的区域
//    //获取3D颜色直方图（每个通道适合8个箱子）
//	hc.setSize(8);        //8 * 8 * 8
//	Mat shist = hc.getHistogram(imageROI);
//	//创建内容搜寻器
//	//设置用来反向投影的直方图
//	finder.setHistogram(shist);
//	finder.setThreshold(0.05f);
//	//取得颜色直方图的反向投影
//	result1 = finder.find(color);
//
//	imshow("result1", result1);
//
//	//装载彩色图片
//	Mat color2 = imread("./images/dog.jpg");
//	imshow("color2", color2);
//
//	Mat result2 = finder.find(color2);
//
//	imshow("result2", result2);
//
//
//	// Get ab color histogram
//	hc.setSize(256); // 256x256
//	cv::Mat colorhist = hc.getabHistogram(imageROI);
//
//	// display 2D histogram
//	colorhist.convertTo(tmp, CV_8U, -1.0, 255.0);
//	cv::namedWindow("ab histogram");
//	cv::imshow("ab histogram", tmp);
//
//
//	// set histogram to be back - projected
//	finder.setHistogram(colorhist);
//	finder.setThreshold(0.05f);
//
//	// Convert to Lab space
//	cv::Mat lab;
//	cv::cvtColor(color, lab, CV_BGR2Lab);
//
//	// Get back-projection of ab histogram
//	int ch[2] = { 1,2 };
//	result1 = finder.find(lab, 0, 256.0f, ch);
//
//	cv::namedWindow("Result ab (1)");
//	cv::imshow("Result ab (1)", result1);
//
//
//
//	// Second colour image
//	cv::cvtColor(color2, lab, CV_BGR2Lab);
//
//	// Get back-projection of ab histogram
//	result2 = finder.find(lab, 0, 256.0, ch);
//
//	cv::namedWindow("Result ab (2)");
//	cv::imshow("Result ab (2)", result2);
//
//	// Draw a rectangle around the reference sky area
//	cv::rectangle(color, cv::Rect(0, 0, 100, 45), cv::Scalar(0, 0, 0));
//	cv::namedWindow("Color Image");
//	cv::imshow("Color Image", color);
//
//	// Get Hue colour histogram
//	hc.setSize(180); // 180 bins
//	colorhist = hc.getHueHistogram(imageROI);
//
//	// set histogram to be back-projected
//	finder.setHistogram(colorhist);
//
//
//
//	// Convert to HSV space
//	cv::Mat hsv;
//	cv::cvtColor(color, hsv, CV_BGR2HSV);
//	// Get back-projection of hue histogram
//	ch[0] = 0;
//	result1 = finder.find(hsv, 0.0f, 180.0f, ch);
//
//	cv::namedWindow("Result Hue (1)");
//	cv::imshow("Result Hue (1)", result1);
//
//	//// Second colour image
//	//color2 = cv::imread("./images/dog.jpg");
//
//	//// Convert to HSV space
//	//cv::cvtColor(color2, hsv, CV_BGR2HSV);
//
//	//// Get back-projection of hue histogram
//	//result2 = finder.find(hsv, 0.0f, 180.0f, ch);
//
//	//cv::namedWindow("Result Hue (2)");
//	//cv::imshow("Result Hue (2)", result2);
//
//
//	waitKey(0);
//	return 0;
//}


#pragma once
#if !defined HISTOGRAM
#define HISTOGRAM

#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

using namespace std;
using namespace cv;


//创建灰度图像的直方图
class Histogram1D
{
private:
	int histSize[1];        //直方图中箱子的数量
	float hranges[2];       //值范围
	const float *ranges[1]; //值范围的指针
	int channels[1];        //要检查的通道数量

public:
	Histogram1D()
	{
		//准备一维直方图的默认参数
		histSize[0] = 256;    //256个箱子
		hranges[0] = 0.0;     //从0开始（包含0）
		hranges[1] = 256.0;   //到256（不包含256）
		ranges[0] = hranges;
		channels[0] = 0;      //先关注通道0
	}
	////
	//void setChannel(int c)
	//{
	//	channels[0] = c;
	//}
	////
	//int getChannel()
	//{
	//	return channels[0];
	//}
	////
	//void setRange(float minValue, float maxValue)
	//{
	//	hranges[0] = minValue;
	//	hranges[1] = maxValue;
	//}
	////
	//float getMinValue()
	//{
	//	return hranges[0];
	//}
	////
	//float getMaxValue()
	//{
	//	return hranges[1];
	//}
	////
	//void setNBins(int nbins)
	//{
	//	histSize[0] = nbins;
	//}
	////
	//int getNBins()
	//{
	//	return histSize[0];
	//}

	//计算一维直方图
	Mat getHistogram(const Mat &image)
	{
		Mat hist;
		//用calcHist函数计算一维直方图
		//一幅图像的直方图   一幅   使用的通道， 不使用掩码  作为结果的直方图  
		//这是一维的直方图  箱子数量  像素值的范围
		calcHist(&image, 1, channels, Mat(), hist, 1, histSize, ranges);

		return hist;
	}

	//计算一维直方图，并返回它的图像
	Mat getHistogramImage(const Mat &image, int zoom = 1)
	{
		//先计算直方图
		Mat hist = getHistogram(image);

		//创建图像
		return Histogram1D::getImageOfHistogram(hist, zoom);
	}


	//// Stretches the source image using min number of count in bins.
	//Mat stretch(const Mat &image, int minValue = 0) {

	//	// Compute histogram first
	//	Mat hist = getHistogram(image);

	//	// find left extremity of the histogram
	//	int imin = 0;
	//	for (; imin < histSize[0]; imin++) {
	//		// ignore bins with less than minValue entries
	//		if (hist.at<float>(imin) > minValue)
	//			break;
	//	}

	//	// find right extremity of the histogram
	//	int imax = histSize[0] - 1;
	//	for (; imax >= 0; imax--) {

	//		// ignore bins with less than minValue entries
	//		if (hist.at<float>(imax) > minValue)
	//			break;
	//	}

	//	// Create lookup table
	//	int dims[1] = { 256 };
	//	Mat lookup(1, dims, CV_8U);

	//	for (int i = 0; i<256; i++) {

	//		if (i < imin) lookup.at<uchar>(i) = 0;
	//		else if (i > imax) lookup.at<uchar>(i) = 255;
	//		else lookup.at<uchar>(i) = cvRound(255.0*(i - imin) / (imax - imin));
	//	}

	//	// Apply lookup table
	//	Mat result;
	//	result = applyLookUp(image, lookup);

	//	return result;
	//}

	//// Stretches the source image using percentile.
	//Mat stretch(const Mat &image, float percentile) {

	//	// number of pixels in percentile
	//	float number = image.total() * percentile;

	//	// Compute histogram first
	//	Mat hist = getHistogram(image);

	//	// find left extremity of the histogram
	//	int imin = 0;
	//	for (float count = 0.0; imin < 256; imin++) {
	//		// number of pixel at imin and below must be > number
	//		if ((count += hist.at<float>(imin)) >= number)
	//			break;
	//	}

	//	// find right extremity of the histogram
	//	int imax = 255;
	//	for (float count = 0.0; imax >= 0; imax--) {
	//		// number of pixel at imax and below must be > number
	//		if ((count += hist.at<float>(imax)) >= number)
	//			break;
	//	}

	//	// Create lookup table
	//	int dims[1] = { 256 };
	//	Mat lookup(1, dims, CV_8U);

	//	for (int i = 0; i<256; i++) {

	//		if (i < imin) lookup.at<uchar>(i) = 0;
	//		else if (i > imax) lookup.at<uchar>(i) = 255;
	//		else lookup.at<uchar>(i) = cvRound(255.0*(i - imin) / (imax - imin));
	//	}

	//	// Apply lookup table
	//	Mat result;
	//	result = applyLookUp(image, lookup);

	//	return result;
	//}


	//创建一个表示直方图的图像（静态方法）
	static Mat getImageOfHistogram(const Mat &hist, int zoom) {
		//取得箱子值的最大值和最小值
		double maxVal = 0;
		double minVal = 0;
		minMaxLoc(hist, &minVal, &maxVal, 0, 0);

		//取得直方图的大小
		int histSize = hist.rows;
		//用于显示直方图的方形图像
		Mat histImg(histSize *zoom, histSize *zoom, CV_8U, Scalar(255));
		//设置最高点为90%（即图像高度）的箱子个数
		int hpt = static_cast<int>(0.9 * histSize);
		//为每个箱子画垂直线
		for (int h = 0; h < histSize; h++)
		{
			float binVal = hist.at<float>(h);

			if (binVal > 0)
			{
				int intensity = static_cast<int>(binVal * hpt / maxVal);
				line(histImg, Point(h * zoom, histSize * zoom),
					Point(h * zoom, (histSize - intensity) * zoom), Scalar(0), zoom);
			}
		}

		return histImg;
	}



	//// Equalizes the source image.
	//static Mat equalize(const Mat &image) {

	//	Mat result;
	//	equalizeHist(image, result);

	//	return result;
	//}


	//// Applies a lookup table transforming an input image into a 1-channel image
	//static Mat applyLookUp(const Mat &image, // input image
	//	const Mat &lookup) { // 1x256 uchar matrix

	//						 // the output image
	//	Mat result;

	//	// apply lookup table
	//	LUT(image, lookup, result);

	//	return result;
	//}

	//// Applies a lookup table transforming an input image into a 1-channel image
	//// this is a test version with iterator; always use function cv::LUT
	//static Mat applyLookUpWithIterator(const Mat& image, const Mat& lookup) {

	//	// Set output image (always 1-channel)
	//	Mat result(image.rows, image.cols, CV_8U);
	//	Mat_<uchar>::iterator itr = result.begin<uchar>();

	//	// Iterates over the input image
	//	Mat_<uchar>::const_iterator it = image.begin<uchar>();
	//	Mat_<uchar>::const_iterator itend = image.end<uchar>();

	//	// Applies lookup to each pixel
	//	for (; it != itend; ++it, ++itr) {

	//		*itr = lookup.at<uchar>(*it);
	//	}

	//	return result;
	//}
};
#endif // !defined HISTOGRAM


#pragma once
#if !define OFINDER
#define OFINDER

#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/core/core.hpp>

using namespace std;
using namespace cv;

class ContentFinder
{
private:
	//直方图参数
	float hranges[2];
	const float *ranges[3];
	int channels[3];

	float threshold;    //判断阈值
	Mat histogram;      //输入直方图
	SparseMat shistogram;
	bool  isSparse;

public:

	ContentFinder() : threshold(0.1f), isSparse(false)
	{
		//本类中所有的通道的范围相同
		ranges[0] = hranges;
		ranges[1] = hranges;
		ranges[2] = hranges;
	}

	// Sets the threshold on histogram values [0,1]
	void setThreshold(float t) {

		threshold = t;
	}

	// Gets the threshold
	float getThreshold() {

		return threshold;
	}

	//设置应用的直方图
	void setHistogram(const Mat &h)
	{
		isSparse = false;
		normalize(h, histogram, 1.0);
	}
	//设置应用的直方图----SparseMat
	void setHistogram(const SparseMat &h)
	{
		isSparse = true;
		normalize(h, shistogram, 1.0, NORM_L2);
	}

	//使用全部通道，范围【0， 256】
	Mat find(const Mat &image)
	{
		Mat result;

		hranges[0] = 0.0;    //默认范围【0， 256】， hranges[1] = 256.0;
		hranges[1] = 256.0;
		channels[0] = 0;     //三个通道
		channels[1] = 1;
		channels[2] = 2;

		return find(image, hranges[0], hranges[1], channels);
	}

	//查找属于直方图的像素
	Mat find(const Mat& image, float minValue, float maxValue, int *channels)
	{
		Mat result;
		hranges[0] = minValue;
		hranges[1] = maxValue;

		if (isSparse)
		{
			for (int i = 0; i<shistogram.dims(); i++)
				this->channels[i] = channels[i];

			calcBackProject(&image,
				1,          //只使用一幅图像
				channels,   //通道
				histogram,  //直方图
				result,     //反向投影
				ranges,     //每个维度的值范围
				255.0       //选用的换算技术
							//把概率从1映射到255
			);
		}
		else {
			//直方图的维度数与通道列表一致
			for (int i = 0; i < histogram.dims; i++)
			{
				this->channels[i] = channels[i];
			}
			//calcBackProject函数和calcHist函数有些类似，而calcBackProject不会增加箱子的数量，而是从箱子读取的值赋给方向投影图像中对应的像素
			calcBackProject(&image,
				1,          //只使用一幅图像
				channels,   //通道
				histogram,  //直方图
				result,     //反向投影
				ranges,     //每个维度的值范围
				255.0       //选用的换算技术
							//把概率从1映射到255
			);
		}

		//对反射投影结果做阈值化，得到二值图像
		if (threshold > 0.0)

			cv::threshold(result, result, 255.0*threshold, 255.0, cv::THRESH_BINARY);

		return result;
	}
};

#endif // !define OFINDER




#pragma once
#if !define COLORHISTOGRAM
#define COLORHISTOGRAM

#include <iostream>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

class ColorHistogram
{
private:

	int histSize[3];         //每个维度的大小
	float hranges[2];        //值的大小（三个维度用同一个值）
	const float* ranges[3];  //每个维度的范围
	int channels[3];         //需要处理的通道

public:

	ColorHistogram()
	{
		//准备用于彩色图像的默认参数
		//每个维度的范围和大小是相同的
		histSize[0] = histSize[1] = histSize[2] = 256;
		hranges[0] = 0.0;      // BGR范围为0-256
		hranges[1] = 256.0;
		ranges[0] = hranges;   //这个类中
		ranges[1] = hranges;   //所有通道的范围均相等
		ranges[2] = hranges; 
		channels[0] = 0;           //三个通道： B
		channels[1] = 1;           //G
		channels[2] = 2;           //R
	}

	//Ã¿¸öÎ¬¶ÈµÃµ½Ö±·½Í¼µÄ´óÐ¡
	void setSize(int size)
	{
		//Ã¿¸öÎ¬¶ÈµÄ´óÐ¡¶¼ÏàµÈ
		histSize[0] = histSize[1] = histSize[3] = size;
	}

	//计算直方图
	Mat getHistogram(const Mat &image)
	{
		Mat hist;

		
		calcHist(&image, 1, //单幅图像的直方图
			channels,  //用到的通道
			Mat(),     //不使用掩码
			hist,      //得到的直方图
			3,         //这是一个三维的直方图
			histSize,  //箱子数量
			ranges);   //像素值的范围

		return hist;
	}

	

	//计算直方图
	SparseMat getSparseHistogram(const Mat &image)
	{
		SparseMat hist(3,       //维数
			histSize,          //每个维度的大小
			CV_32F);

		//计算直方图
		calcHist(&image, 1, 
			channels,  
			Mat(),    
			hist,      
			3,         
			histSize, 
			ranges);  


		return hist;
	}

	//用均值平均算法查找目标
	//计算一维色调直方图
	//BGR 的原图转化为 HSV
	//忽略低饱和度的像素
	cv::Mat getHueHistogram(const cv::Mat &image,
		int minSaturation = 0) {

		cv::Mat hist;

		//转化为HSV空间
		cv::Mat hsv;
		cv::cvtColor(image, hsv, CV_BGR2HSV);

		// 掩码（可能用到，可能用不到）
		cv::Mat mask;
		//根据需要创建掩码
		if (minSaturation>0) {

			// 将3个通道分割进是3个图像
			std::vector<cv::Mat> v;
			cv::split(hsv, v);

			//屏蔽低饱和度的像素
			cv::threshold(v[1], mask, minSaturation, 255,
				cv::THRESH_BINARY);
		}

		// 准备一维色调直方图的参数
		hranges[0] = 0.0;    //范围为 0 - 180
		hranges[1] = 180.0;
		channels[0] = 0;    //色调通道

		// 计算直方图
		cv::calcHist(&hsv,
			1,			
			channels,	
			mask,		
			hist,		
			1,			
			histSize,	
			ranges		
		);

		return hist;
	}


	// Computes the 2D ab histogram.
	// BGR source image is converted to Lab
	cv::Mat getabHistogram(const cv::Mat &image) {

		cv::Mat hist;

		// Convert to Lab color space
		cv::Mat lab;
		cv::cvtColor(image, lab, CV_BGR2Lab);

		// Prepare arguments for a 2D color histogram
		hranges[0] = 0;
		hranges[1] = 256.0;
		channels[0] = 1; // the two channels used are ab 
		channels[1] = 2;

		// Compute histogram
		cv::calcHist(&lab,
			1,			// histogram of 1 image only
			channels,	// the channel used
			cv::Mat(),	// no mask is used
			hist,		// the resulting histogram
			2,			// it is a 2D histogram
			histSize,	// number of bins
			ranges		// pixel value range
		);

		return hist;
	}


};

#endif // !define COLORHISTOGRAM

```



### Chapter_05 : 用形态学运算变换图像

> 1.腐蚀、膨胀、开始、闭合等形态学运算
> 2.MSER算法提取特征区域
> 3.用分水岭算法实现图像分割

<!--more-->

```c++

#include "WatershedSegmenter.h"

int main()
{
	////1.腐蚀、膨胀、开始、闭合等形态学运算
	//Mat image = imread("./images/binary.bmp");
	//if (!image.data)
	//	return 0;

	//imshow("image", image);


	//Mat eroded; 
	//erode(image, eroded, Mat());
	//imshow("erode", eroded);

	//Mat dilated; 
	//dilate(image, dilated, Mat());
	//imshow("dilate", dilated);


	//Mat element(7, 7, CV_8U, Scalar(1));

	//erode(image, eroded, element);
	//imshow("Erode-7*7", eroded);


	//erode(image, eroded, Mat(), Point(-1, -1), 3);
	//imshow("erode--3time", eroded);



	//Mat element5(5, 5, CV_8U, Scalar(1));
	//Mat closed;
	//morphologyEx(image, closed, MORPH_CLOSE, element5);
	//imshow("close", closed);


	//Mat opened;
	//morphologyEx(image, opened, MORPH_OPEN, element5);
	//imshow("open", opened);

	//// explicit closing
	//// 1. dilate original image
	//cv::Mat result;
	//cv::dilate(image, result, element5);
	//// 2. in-place erosion of the dilated image
	//cv::erode(result, result, element5);

	//// Display the closed image
	//cv::namedWindow("Closed Image (2)");
	//cv::imshow("Closed Image (2)", result);

	//// Close and Open the image
	//cv::morphologyEx(image, image, cv::MORPH_CLOSE, element5);
	//cv::morphologyEx(image, image, cv::MORPH_OPEN, element5);

	//// Display the close/opened image
	//cv::namedWindow("Closed|Opened Image");
	//cv::imshow("Closed|Opened Image", image);
	//cv::imwrite("binaryGroup.bmp", image);

	//// Read input image
	//image = cv::imread("./images/binary.bmp");

	//// Open and Close the image
	//cv::morphologyEx(image, image, cv::MORPH_OPEN, element5);
	//cv::morphologyEx(image, image, cv::MORPH_CLOSE, element5);

	//// Display the close/opened image
	//cv::namedWindow("Opened|Closed Image");
	//cv::imshow("Opened|Closed Image", image);


	//Mat image2 = imread("./images/boldt.jpg", 0);
	//if (!image2.data)
	//	return 0;
	//imshow("image2", image2);

	//morphologyEx(image2, result, MORPH_GRADIENT, Mat());
	//imshow("Edge_result", result);
	//imshow("Edge_255-result", 255 - result);

	//int threshold(80);
	//cv::threshold(result, result, threshold, 255, THRESH_BINARY);
	//imshow("threshold_result", result);

	
	//Mat image3 = imread("./images/book.jpg", 0);
	//if (!image3.data)
	//	return 0;
	//imshow("Image3", image3);

	//transpose(image3, image3);
	//imshow("transpose", image3);
	//flip(image3, image3, 0);
	//imshow("image3", image3);


	//Mat element7(7, 7, CV_8U, Scalar(1));
	//morphologyEx(image3, result, MORPH_BLACKHAT, element7);
	//imshow("Blackhat", result);

	//threshold = 25;
	//cv::threshold(result, result,
	//	threshold, 255, cv::THRESH_BINARY);

	//imshow("Thresholded Black Top-hat", 255 - result);


	////2.MSER算法提取特征区域
	//Mat image = imread("./images/building.jpg", 0);
	//if (!image.data)
	//	return 0;
	//imshow("image", image);

	////基本的MSER检测器
	//Ptr<MSER> ptrMSER = MSER::create(5,  //局部检测时使用的增量值
	//	200,                             //允许的最小面积
	//	2000);                           //允许的最大面积

	////点集的容器
	//vector<vector<Point> > points;
	////矩形的容器
	//vector<Rect> rects;
	////检测MSER特征 ----检测的结果放在两个区域，第一个是区域的容器，每个区域用组成它的像素点表示，
	////第二个是矩形的容器，每个矩形包围一个区域
	//ptrMSER->detectRegions(image, points, rects);

	//cout << points.size() << "MSERs detected" << endl;

	////创建白色区域
	//Mat output(image.size(), CV_8UC3);
	//output = Scalar(255, 255, 255);
 //   //Opencv随机数生成器
	//RNG rng;
	////针对每个检测到的特征区域， 在彩色区域显示MSER
	////反向排序，先显示较大的MSER
	//for (vector<vector<Point> > ::reverse_iterator it = points.rbegin(); it != points.rend(); ++it)
	//{
	//	//生成随机颜色
	//	Vec3b c(rng.uniform(0, 254), rng.uniform(0, 254), rng.uniform(0, 254));

	//	cout << "MSER size = " << it->size() << endl;

	//	//针对MSER集合中的每个点
	//	for (vector<Point> ::iterator itPts = it->begin(); itPts != it->end(); ++itPts)
	//	{
	//		//不重复MSER的像素
	//		if (output.at<Vec3b>(*itPts)[0] == 255)
	//		{
	//			output.at<Vec3b>(*itPts) = c;
	//		}
	//	}
	//}

	//imshow("MSER point sets", output);
	//imwrite("./images/mser.bmp", output);


	////提取并显示矩形的MSER
	//vector<Rect> ::iterator itr = rects.begin();
	//vector<vector<Point> > ::iterator  itp = points.begin();
	//for (; itr != rects.end(); ++itr, ++itp)
	//{
	//	//检查两者比例
	//	if (static_cast<double> (itp->size()) / itr->area() > 0.6)
	//		rectangle(image, *itr, Scalar(0, 0, 255), 2);
	//}

	////显示结果
	//imshow("Rectangle MSERs", image);

	////提取并显示椭圆形的MSER
	//image = imread("./images/building.jpg", 0);
	//if (!image.data)
	//	return 0;
	//for (vector<vector<Point> > ::iterator it = points.begin(); it != points.end(); ++it)
	//{
	//	//遍历MSER集合中的每一个点
	//	for (vector<Point> ::iterator itPts = it->begin(); itPts != it->end(); ++itPts)
	//	{
	//		//提取封闭的矩形
	//		RotatedRect rr = minAreaRect(*it);
	//		//检查椭圆得长宽比
	//		if (rr.size.height / rr.size.width > 0.2 || rr.size.height / rr.size.width < 1.6)
	//			ellipse(image, rr, Scalar(255), 2);
	//		
	//	}

	//}

	//imshow("MSER ellipses", image);


	//3.用分水岭算法实现图像分割
	Mat image = imread("./images/group.jpg");
	if (!image.data)
		return 0;

	imshow("image", image);

	Mat binary;
	binary = imread("./images/binary.bmp", 0);
	imshow("Binary Image", binary);

	//消除噪声和细小的物体
	Mat fg;
	erode(binary, fg, Mat(), Point(-1, -1), 4);
	imshow("erode", fg);
	//标识不含物体的图像像素
	Mat bg;
	dilate(binary, bg, Mat(), Point(-1, -1), 4);
	threshold(bg, bg, 1, 128, THRESH_BINARY_INV);
	imshow("dilate_threshold", bg);

	//创建标记图像
	Mat markers(binary.size(), CV_8U, Scalar(0));
	markers = fg + bg;
	imshow("Markers", markers);
	//创建分水岭分割类的对象
	WatershedSegmenter segmenter;
	//设置标记图像，然后执行分割过程
	segmenter.setMarkers(markers);
	segmenter.process(image);

	imshow("Segmentation", segmenter.getSegmentation());

	imshow("Watersheds", segmenter.getWatersheds());

	// Open another image
	image = cv::imread("./images/tower.jpg");

	// Identify background pixels
	cv::Mat imageMask(image.size(), CV_8U, cv::Scalar(0));
	cv::rectangle(imageMask, cv::Point(5, 5), cv::Point(image.cols - 5, image.rows - 5), cv::Scalar(255), 3);
	// Identify foreground pixels (in the middle of the image)
	cv::rectangle(imageMask, cv::Point(image.cols / 2 - 10, image.rows / 2 - 10),
		cv::Point(image.cols / 2 + 10, image.rows / 2 + 10), cv::Scalar(1), 10);

	// Set markers and process
	segmenter.setMarkers(imageMask);
	segmenter.process(image);

	// Display the image with markers
	cv::rectangle(image, cv::Point(5, 5), cv::Point(image.cols - 5, image.rows - 5), cv::Scalar(255, 255, 255), 3);
	cv::rectangle(image, cv::Point(image.cols / 2 - 10, image.rows / 2 - 10),
		cv::Point(image.cols / 2 + 10, image.rows / 2 + 10), cv::Scalar(1, 1, 1), 10);
	cv::namedWindow("Image with marker");
	cv::imshow("Image with marker", image);

	// Display watersheds
	cv::namedWindow("Watershed");
	cv::imshow("Watershed", segmenter.getWatersheds());

	waitKey(0);

	return 0;
}



#pragma once
#if !define WATERSHS
#define WATERSHS

#include <opencv2/opencv.hpp>
#include <iostream>

using namespace std;
using namespace cv;

class WatershedSegmenter {

private:
	Mat markers;

public:

	void setMarkers(const Mat &markerImage)
	{
		//转化为整数型图像
		markerImage.convertTo(markers, CV_32S);
	}

	Mat process(const Mat &image)
	{
		//应用分水岭
		watershed(image, markers);

		return markers;
	}
	//以图像的形式返回结果
	Mat getSegmentation()
	{
		Mat tmp;
		//所有标签值大于255的区域都赋值为255
		markers.convertTo(tmp, CV_8U);

		return tmp;
	}

	//以图像的形式返回分水岭
	Mat getWatersheds()
	{
		Mat tmp;
		markers.convertTo(tmp, CV_8U, 255, 255);

		return tmp;
	}

};

#endif // !define WATERSHS

```

### Chapter_06  :  图像滤波

> 1.方块滤波、高斯滤波、resize(), pyrUp(), pyrDown()
> 2.Sobel滤波器
> 3.Laplacian算子
> 4.高斯差分

<!--more-->

```c++
//#include <iostream>
//#include <opencv2/opencv.hpp>
//
//#include "LaplacianZC.h"
//
//using namespace std;
//using namespace cv;
//
//int main()
//{
//	////1.方块滤波、高斯滤波、resize(), pyrUp(), pyrDown()
//	//Mat image = imread("./images/boldt.jpg", 0);
//	//if (!image.data)
//	//{
//	//	cout << "image is error" << endl;
//	//	return 0;
//	//}
//	//imshow("image", image);
//	//  
//	////方块滤波
//	//Mat result;
//	//blur(image, result, Size(5, 5));
//	//imshow("Mean Image", result);
// //   //Blur the image with a mean filter 9*9
//	//blur(image, result, Size(9, 9));
//	//imshow("Mean Image(9 * 9)", result);
//
//	////高斯滤波
//	//GaussianBlur(image, result, Size(5, 5), //滤波尺寸
//	//	1.5);                               //控制高斯曲线形状的参数
//	//imshow("Gaussian Filtered Image", result);
//
//	//// Display the blurred image
//	//cv::namedWindow("Gaussian filtered Image (9x9)");
//	//cv::imshow("Gaussian filtered Image (9x9)", result);
//
//	////Get the gaussian kernel (1.5)
//	//Mat gauss = getGaussianKernel(9, 1.5, CV_32F);
//
//	////Display kernal value 
//	//Mat_<float>::const_iterator it = gauss.begin<float>();
//	//Mat_<float>::const_iterator itend = gauss.end<float>();
//	//cout << "1.5 = [";
//	//for (; it != itend; ++it)
//	//{
//	//	cout << "]" << endl;
//	//}
//
//	////Get the gaussian kernel(0.5)
//	//gauss = getGaussianKernel(9, 0.5, CV_32F);
//	//// Display kernel values
//	//it = gauss.begin<float>();
//	//itend = gauss.end<float>();
//	//std::cout << "0.5 = [";
//	//for (; it != itend; ++it) {
//	//	std::cout << *it << " ";
//	//}
//	//std::cout << "]" << std::endl;
//
//	//// Get the gaussian kernel (2.5)
//	//gauss = cv::getGaussianKernel(9, 2.5, CV_32F);
//
//	//// Display kernel values
//	//it = gauss.begin<float>();
//	//itend = gauss.end<float>();
//	//std::cout << "2.5 = [";
//	//for (; it != itend; ++it) {
//	//	std::cout << *it << " ";
//	//}
//	//std::cout << "]" << std::endl;
//
//	//// Get the gaussian kernel(9 elements)
//	//gauss = cv::getGaussianKernel(9, -1, CV_32F);
//
//	//// Display kernel values
//	//it = gauss.begin<float>();
//	//itend = gauss.end<float>();
//	//std::cout << "9 = [";
//	//for (; it != itend; ++it) {
//	//	std::cout << *it << " ";
//	//}
//	//std::cout << "]" << std::endl;
//
//	//// Get the Deriv kernel (2.5)
//	//cv::Mat kx, ky;
//	//cv::getDerivKernels(kx, ky, 2, 2, 7, true);
//
//	//// Display kernel values
//	//cv::Mat_<float>::const_iterator kit = kx.begin<float>();
//	//cv::Mat_<float>::const_iterator kitend = kx.end<float>();
//	//std::cout << "[";
//	//for (; kit != kitend; ++kit) {
//	//	std::cout << *kit << " ";
//	//}
//	//std::cout << "]" << std::endl;
//
//	//// Read input image with salt&pepper noise
//	//image = cv::imread("./images/salted.bmp", 0);
//	//if (!image.data)
//	//	return 0;
//
//	//// Display the S&P image
//	//cv::namedWindow("S&P Image");
//	//cv::imshow("S&P Image", image);
//
//	//// Blur the image with a mean filter
//	//cv::blur(image, result, cv::Size(5, 5));
//
//	//// Display the blurred image
//	//cv::namedWindow("Mean filtered S&P Image");
//	//cv::imshow("Mean filtered S&P Image", result);
//
//	//// Applying a median filter
//	//cv::medianBlur(image, result, 5);
//
//	//// Display the blurred image
//	//cv::namedWindow("Median filtered Image");
//	//cv::imshow("Median filtered Image", result);
//
//	//// Reduce by 4 the size of the image (the wrong way)
//	////只保留每四个像素中的一个
//	//image = cv::imread("./images/boldt.jpg", 0);
//	//cv::Mat reduced(image.rows / 4, image.cols / 4, CV_8U);
//
//	//for (int i = 0; i<reduced.rows; i++)
//	//	for (int j = 0; j<reduced.cols; j++)
//	//		reduced.at<uchar>(i, j) = image.at<uchar>(i * 4, j * 4);
//
//	//// Display the reduced image
//	//cv::namedWindow("Badly reduced Image");
//	//cv::imshow("Badly reduced Image", reduced);
//
//	//cv::resize(reduced, reduced, cv::Size(), 4, 4, cv::INTER_NEAREST);
//
//	//// Display the (resized) reduced image
//	//cv::namedWindow("Badly reduced");
//	//cv::imshow("Badly reduced", reduced);
//
//	//cv::imwrite("badlyreducedimage.bmp", reduced);
//
//	//// first remove high frequency component
//	//cv::GaussianBlur(image, image, cv::Size(11, 11), 1.75);
//	//// keep only 1 of every 4 pixels
//	//cv::Mat reduced2(image.rows / 4, image.cols / 4, CV_8U);
//	//for (int i = 0; i<reduced2.rows; i++)
//	//	for (int j = 0; j<reduced2.cols; j++)
//	//		reduced2.at<uchar>(i, j) = image.at<uchar>(i * 4, j * 4);
//
//	//// Display the reduced image
//	//cv::namedWindow("Reduced Image, original size");
//	//cv::imshow("Reduced Image, original size", reduced2);
//
//	//cv::imwrite("reducedimage.bmp", reduced2);
//
//	//// resizing with NN
//	//cv::Mat newImage;
//	//cv::resize(reduced2, newImage, cv::Size(), 4, 4, cv::INTER_NEAREST);
//
//	//// Display the (resized) reduced image
//	//cv::namedWindow("Reduced Image");
//	//cv::imshow("Reduced Image", newImage);
//
//	//// resizing with bilinear
//	//cv::resize(reduced2, newImage, cv::Size(), 4, 4, cv::INTER_LINEAR);
//
//	//// Display the (resized) reduced image
//	//cv::namedWindow("Bilinear resizing");
//	//cv::imshow("Bilinear resizing", newImage);
//
//	//// Creating an image pyramid
//	//cv::Mat pyramid(image.rows, image.cols + image.cols / 2 + image.cols / 4 + image.cols / 8, CV_8U, cv::Scalar(255));
//	//image.copyTo(pyramid(cv::Rect(0, 0, image.cols, image.rows)));
//
//	//cv::pyrDown(image, reduced); // reduce image size by half
//	//reduced.copyTo(pyramid(cv::Rect(image.cols, image.rows / 2, image.cols / 2, image.rows / 2)));
//	//cv::pyrDown(reduced, reduced2); // reduce image size by another half
//	//reduced2.copyTo(pyramid(cv::Rect(image.cols + image.cols / 2, image.rows - image.rows / 4, image.cols / 4, image.rows / 4)));
//	//cv::pyrDown(reduced2, reduced); // reduce image size by another half
//	//reduced.copyTo(pyramid(cv::Rect(image.cols + image.cols / 2 + image.cols / 4, image.rows - image.rows / 8, image.cols / 8, image.rows / 8)));
//
//	//// Display the pyramid
//	//cv::namedWindow("Pyramid of images");
//	//cv::imshow("Pyramid of images", pyramid);
//
//
//	//2.Sobel滤波器
//    Mat image = imread("./images/boldt.jpg", 0);
//	if (!image.data)
//		return 0;
//	imshow("Image", image);
//
//	//计算Sobel滤波器的X方向
//	Mat sobelX;
//	Sobel(image,    //输入 
//		sobelX,     // 输出
//		CV_8U,      //图像类型
//		1, 0,       //内核规格
//		3,          //正方形内核的尺寸
//		0.4, 128);  // 比例和偏移量
//
//	imshow("Sobel X Image", sobelX);
//
//	//计算Sobel滤波器的Y方向
//	Mat sobelY;
//	Sobel(image,    //输入 
//		sobelY,     // 输出
//		CV_8U,      //图像类型
//		0, 1,       //内核规格
//		3,          //正方形内核的尺寸
//		0.4, 128);  // 比例和偏移量
//
//	imshow("Sobel Y Image", sobelY);
//
//	//组合这两个结果   计算Sobel滤波器的范数
//	Sobel(image, sobelX, CV_16S, 1, 0);
//	Sobel(image, sobelY, CV_16S, 0, 1);
//
//	Mat sobel;
//	//计算L1范数
//	sobel = abs(sobelX) + abs(sobelY);
//	//找到Sobel最大值
//	double sobmin, sobmax;
//	minMaxLoc(sobel, &sobmin, &sobmax);
//	cout << "sobel value range:" << sobmin << "  " << sobmax << endl;
//
//	// Compute Sobel X derivative (7x7)
//	cv::Sobel(image, sobelX, CV_8U, 1, 0, 7, 0.001, 128);
//
//	// Display the image
//	cv::namedWindow("Sobel X Image (7x7)");
//	cv::imshow("Sobel X Image (7x7)", sobelX);
//
//	//Print window pixel values
//	for (int i = 0; i < 12; i++)
//	{
//		for (int j = 0; j < 12; j++)
//		{
//			cout << setw(5) << static_cast<int>(sobel.at<short>(i + 79, j + 215)) << " ";
//		}
//		cout << endl;
//
//	}
//
//	cout << endl;
//	cout << endl;
//	cout << endl;
//
//	//转换成8位图像
//	Mat sobelImage;
//	sobel.convertTo(sobelImage, CV_8U, -255./sobmax, 255);
//
//	imshow("sobelImage", sobelImage);
//	//进行阈值化
//	Mat sobelThresholded;
//	threshold(sobelImage, sobelThresholded, 225, 255, THRESH_BINARY);
//	imshow("sobelThresholded", sobelThresholded);
//
//
//
//	//3.Laplacian算子
//	//默认的拉普拉斯算子的核的值的大小是 3 * 3
//	Mat laplace;
//	Laplacian(image, laplace, CV_8U, 1, 1, 128);
//
//	imshow("Laplacian Image", laplace);
//
//	int cx(238), cy(90);
//	int dx(12), dy(12);
//
//	//一个小的窗口
//	Mat window(image, Rect(cx, cy, dx, dy));
//	imshow("window", window);
//	imwrite("./images/window.bmp", window);
//
//	//用LaplacianZC类计算拉普拉斯
//	LaplactionZC laplacian;
//	laplacian.setAperture(7);
//	Mat flap = laplacian.computeLaplacian(image);
//
//	// display min max values of the lapalcian
//	double lapmin, lapmax;
//	cv::minMaxLoc(flap, &lapmin, &lapmax);
//	// display laplacian image
//	laplace = laplacian.getLaplacianImage();
//	cv::namedWindow("Laplacian Image (7x7)");
//	cv::imshow("Laplacian Image (7x7)", laplace);
//
//	// Print image values
//	std::cout << std::endl;
//	std::cout << "Image values:\n\n";
//	for (int i = 0; i<dx; i++) {
//		for (int j = 0; j<dy; j++)
//			std::cout << std::setw(5) << static_cast<int>(image.at<uchar>(i + cy, j + cx)) << " ";
//		std::cout << std::endl;
//	}
//	std::cout << std::endl;
//
//	// Print Laplacian values
//	std::cout << "Laplacian value range=[" << lapmin << "," << lapmax << "]\n";
//	std::cout << std::endl;
//	for (int i = 0; i<dx; i++) {
//		for (int j = 0; j<dy; j++)
//			std::cout << std::setw(5) << static_cast<int>(flap.at<float>(i + cy, j + cx) / 100) << " ";
//		std::cout << std::endl;
//	}
//	std::cout << std::endl;
//
//	// Compute and display the zero-crossing points
//	cv::Mat zeros;
//	zeros = laplacian.getZeroCrossings(flap);
//	cv::namedWindow("Zero-crossings");
//	cv::imshow("Zero-crossings", 255 - zeros);
//
//	// Print window pixel values
//	std::cout << "Zero values:\n\n";
//	for (int i = 0; i<dx; i++) {
//		for (int j = 0; j<dy; j++)
//			std::cout << std::setw(2) << static_cast<int>(zeros.at<uchar>(i + cy, j + cx)) / 255 << " ";
//		std::cout << std::endl;
//	}
//
//	// down-sample and up-sample the image
//	cv::Mat reduced, rescaled;
//	cv::pyrDown(image, reduced);
//	cv::pyrUp(reduced, rescaled);
//
//	// Display the rescaled image
//	cv::namedWindow("Rescaled Image");
//	cv::imshow("Rescaled Image", rescaled);
//
//
//	//4.高斯差分（DoG）
//	cv::Mat dog;
//	//计算高斯差分
//	cv::subtract(rescaled, image, dog, cv::Mat(), CV_16S);
//	cv::Mat dogImage;
//	dog.convertTo(dogImage, CV_8U, 1.0, 128);
//
//	// Display the DoG image
//	cv::namedWindow("DoG Image (from pyrdown/pyrup)");
//	cv::imshow("DoG Image (from pyrdown/pyrup)", dogImage);
//
//	// Apply two Gaussian filters
//	cv::Mat gauss05;
//	cv::Mat gauss15;
//	cv::GaussianBlur(image, gauss05, cv::Size(), 0.5);
//	cv::GaussianBlur(image, gauss15, cv::Size(), 1.5);
//
//	// compute a difference of Gaussians 
//	cv::subtract(gauss15, gauss05, dog, cv::Mat(), CV_16S);
//	dog.convertTo(dogImage, CV_8U, 2.0, 128);
//
//	// Display the DoG image
//	cv::namedWindow("DoG Image");
//	cv::imshow("DoG Image", dogImage);
//
//	// Apply two Gaussian filters
//	cv::Mat gauss20;
//	cv::GaussianBlur(image, gauss20, cv::Size(), 2.0);
//	cv::Mat gauss22;
//	cv::GaussianBlur(image, gauss22, cv::Size(), 2.2);
//
//	// compute a difference of Gaussians 
//	cv::subtract(gauss22, gauss20, dog, cv::Mat(), CV_32F);
//	dog.convertTo(dogImage, CV_8U, 10.0, 128);
//
//	// Display the DoG image
//	cv::namedWindow("DoG Image (2)");
//	cv::imshow("DoG Image (2)", dogImage);
//
//
//	// 计算高斯差分的过零点
//	zeros = laplacian.getZeroCrossings(dog);
//	cv::namedWindow("Zero-crossings of DoG");
//	cv::imshow("Zero-crossings of DoG", 255 - zeros);
//
//	// Display the image with window
//	cv::rectangle(image, cv::Rect(cx, cy, dx, dy), cv::Scalar(255, 255, 255));
//	cv::namedWindow("Original Image with window");
//	cv::imshow("Original Image with window", image);
//
//
//	waitKey(0);
//	return 0;
//}
```

### Chapter_07 : 提取直线、轮廓和区域

> 1.Sobel算子的部分   Canny边缘提取
> 2.用霍夫变换检测直线
> 3.用概率霍夫变换检测直线
> 4.点集的直线拟合
> 5.提取连续区域

<!--more-->

```c++
//#include <iostream>
//#include <opencv2/opencv.hpp>
//#include <vector>
//
//#include "EdgeDetector.h"
//#include "LineFinder.h"
//
//using namespace std;
//using namespace cv;
//
//int main()
//{
//	//Mat image = imread("./images/road.jpg", 0);
//	//if (!image.data)
//	//	return 0;
//	//imshow("Image", image);
//
//	////计算Sobel
//	//EdgeDetector ed;
//	//ed.computeSobel(image);
//
//	////展示Sobel
//	//imshow("Sobel (orientation)", ed.getSobelOrientationImage());
//	//imwrite("./images/ori.bmp", ed.getSobelOrientationImage());
//	//// Display the Sobel low threshold
//	//cv::namedWindow("Sobel (low threshold)");
//	//cv::imshow("Sobel (low threshold)", ed.getBinaryMap(125));
//
//	//// Display the Sobel high threshold
//	//cv::namedWindow("Sobel (high threshold)");
//	//cv::imshow("Sobel (high threshold)", ed.getBinaryMap(350));
//
//	////运用Canny边缘算子
//	//Mat contours;
//	//Canny(image,             //灰度图像
//	//	contours,            //输出轮廓
//	//	125,                 //低阈值
//	//	350);                //高阈值
//
//	//imshow("Canny", 255 - contours);
//
//	////创建一个测试的图像
//	//Mat test(200, 200, CV_8U, Scalar(0));
//	//cv::line(test, cv::Point(100, 0), cv::Point(200, 200), cv::Scalar(255));
//	//cv::line(test, cv::Point(0, 50), cv::Point(200, 200), cv::Scalar(255));
//	//cv::line(test, cv::Point(0, 200), cv::Point(200, 0), cv::Scalar(255));
//	//cv::line(test, cv::Point(200, 0), cv::Point(0, 200), cv::Scalar(255));
//	//cv::line(test, cv::Point(100, 0), cv::Point(100, 200), cv::Scalar(255));
//	//cv::line(test, cv::Point(0, 100), cv::Point(200, 100), cv::Scalar(255));
//
//	//// Display the test image
//	//cv::namedWindow("Test Image");
//	//cv::imshow("Test Image", test);
//	//cv::imwrite("test.bmp", test);
//
//	////2.用霍夫变换检测直线
//	//vector<Vec2f> lines;
//	//HoughLines(contours, lines, 1,
//	//	PI/180,          //步长
//	//    60);             //最小投票数
//	////画线
//	//Mat result(contours.rows, contours.cols, CV_8U, Scalar(255));
//	//image.copyTo(result);
//
//	//cout << "Lines detected" << lines.size() << endl;
//
//	//vector<Vec2f>::const_iterator it = lines.begin();
//	//while (it != lines.end())
//	//{
//	//	float rho = (*it)[0];     //第一个元素是距离rho
//	//	float theta = (*it)[1];   //第二个元素是角度theta
//
//	//	if (theta < PI/4. || theta > 3.*PI / 4) //垂直线（大致）
//	//	{
//	//		//直线与第一行的交叉点
//	//		Point pt1(rho/cos(theta), 0);
//	//		//直线与最后一行的交叉点
//	//		Point pt2((rho - result.rows * sin(theta)) / cos(theta), result.rows);
//	//		//画白色的线
//	//		line(result, pt1, pt2, Scalar(255), 1);
//
//	//	}
//	//	else //水平线（大致）
//	//	{
//	//		//直线与第一列的交叉点
//	//		Point pt1(0, rho/sin(theta));
//	//		//直线与最后一列的交叉点
//	//		Point pt2(result.cols, (rho - result.cols * cos(theta)) / sin(theta));
//	//		//画白色的线
//	//		line(result, pt1, pt2, Scalar(255), 1);
//	//	}
//	//	cout << "line: (" << rho << "," << theta << ")\n";
//
//	//	++it;
//	//}
//	//imshow("HoughLines", result);
//	//
//	////3.用概率霍夫变换检测直线
//	////创建LineFinder类的实例
//	//LineFinder finder;
//	////设置概率霍夫变换的参数
//	//finder.setLineLengthAndGap(100, 20);
//	//finder.setMinVote(60);
//
//	////检测直线并画直线
//	//vector<Vec4i> linesp = finder.findLines(contours);
//	//finder.drawDetectedLines(image);
//
//	//std::vector<cv::Vec4i>::const_iterator it2 = linesp.begin();
//	//while (it2 != linesp.end()) {
//
//	//	std::cout << "(" << (*it2)[0] << "," << (*it2)[1] << ")-("
//	//		<< (*it2)[2] << "," << (*it2)[3] << ")" << std::endl;
//
//	//	++it2;
//	//}
//
//	//imshow("HoughLinesp", image);
//
//	////4.点集的直线拟合
//	//image = imread("./images/road.jpg", 0);
//
//	//int n = 0;  //选用直线0
//	//line(image, Point(linesp[n][0], linesp[n][1]), Point(linesp[n][2], linesp[n][3]),Scalar(255), 5);
//	//imshow("One line of the Image", image);
//	////黑白图像
//	//Mat oneline(image.size(), CV_8U, Scalar(0));
//	////白色直线
//	//line(oneline, Point(linesp[n][0], linesp[n][1]), Point(linesp[n][2], linesp[n][3]), Scalar(255), 3);
//	////轮廓与白色直线进行与&操作
//	//bitwise_and(contours, oneline, oneline);
//	//imshow("One line", 255 - oneline);
//
//	//vector<Point> points;
//	////迭代遍历像素，得到所有点的位置
//	//for (int y = 0; y < oneline.rows; y++)
//	//{
//	//	//行y
//	//	uchar *rowPtr = oneline.ptr<uchar>(y);
//	//	for (int x = 0; x < oneline.cols; x++)
//	//	{
//	//		//列x
//
//	//		//如果在轮廓上
//	//		if (rowPtr[x])
//	//		{
//	//			points.push_back(Point(x, y));
//	//		}
//	//	}
//	//}
//
//	////得到点集后， 利用这些点集拟合出直线， fitLine可以轻松的得到最优的拟合直线
//	//Vec4f line;
//	//fitLine(points, line,
//	//	DIST_L2,    //距离类型
//	//	0,          //L2距离不用这个参数
//	//	0.01, 0.01);//精度
//
//	//cout << "line: (" << line[0] << "," << line[1] << ")(" << line[2] << "," << line[3] << ")\n";
//
//	//int x0 = line[2];      //直线上一个点
//	//int y0 = line[3];      
//	//int x1 = x0 + 100 * line[0];  //加上长度为100的向量
//	//int y1 = y0 + 100 * line[1];  //（用单位向量生成）
//	//image = imread("./images/road.jpg", 0);
//	////绘制这条线
//	//cv::line(image, Point(x0, y0), Point(x1, y1), Scalar(0), 0.2); //颜色和宽度
//	//imshow("Fitted line", image);
//
//	//// eliminate inconsistent lines
//	//finder.removeLinesOfInconsistentOrientations(ed.getOrientation(), 0.4, 0.1);
//
//	//// Display the detected line image
//	//image = cv::imread("./images/road.jpg", 0);
//
//	//finder.drawDetectedLines(image);
//	//cv::namedWindow("Detected Lines (2)");
//	//cv::imshow("Detected Lines (2)", image);
//
//
//	////创建霍夫累加器
//	////这里的用的图像类型是uchar，实际类型应该是int
//	//cv::Mat acc(200, 180, CV_8U, cv::Scalar(0));
//
//	////选取一个像素点
//	//int x = 50, y = 30;
//
//	////循环遍历所有角度
//	//for (int i = 0; i<180; i++) {
//
//	//	double theta = i*PI / 180.;
//
//	//	//找到对应的rho值
//	//	double rho = x*std::cos(theta) + y*std::sin(theta);
//	//	//j对应-100到100的rho
//	//	int j = static_cast<int>(rho + 100.5);
//
//	//	std::cout << i << "," << j << std::endl;
//
//	//	//增加累加器
//	//	acc.at<uchar>(j, i)++;
//	//}
//
//	//// draw the axes
//	//cv::line(acc, cv::Point(0, 0), cv::Point(0, acc.rows - 1), 255);
//	//cv::line(acc, cv::Point(acc.cols - 1, acc.rows - 1), cv::Point(0, acc.rows - 1), 255);
//
//	//cv::imwrite("./images/hough1.bmp", 255 - (acc * 100));
//
//	//// Choose a second point
//	//x = 30, y = 10;
//
//	//// loop over all angles
//	//for (int i = 0; i<180; i++) {
//
//	//	double theta = i*PI / 180.;
//	//	double rho = x*cos(theta) + y*sin(theta);
//	//	int j = static_cast<int>(rho + 100.5);
//
//	//	acc.at<uchar>(j, i)++;
//	//}
//
//	//cv::namedWindow("Hough Accumulator");
//	//cv::imshow("Hough Accumulator", acc * 100);
//	//cv::imwrite("./images/hough2.bmp", 255 - (acc * 100));
//
//	////检测圆
//	//image = cv::imread("./images/chariot.jpg", 0);
//
//	//cv::GaussianBlur(image, image, cv::Size(5, 5), 1.5);
//	//std::vector<cv::Vec3f> circles;
//	//cv::HoughCircles(image, circles, cv::HOUGH_GRADIENT,
//	//	2,   // 累加器分辨率(图像尺寸 / 2) 
//	//	20,  // 两个圆之间的最小距离
//	//	200, // Canny算子的高阈值
//	//	60, // 最小投票数
//	//	15, 50); // 最小和最大半径
//
//	//std::cout << "Circles: " << circles.size() << std::endl;
//
//	//// Draw the circles
//	//image = cv::imread("./images/chariot.jpg", 0);
//
//	//std::vector<cv::Vec3f>::const_iterator itc = circles.begin();
//
//	//while (itc != circles.end()) {
//
//	//	cv::circle(image,
//	//		cv::Point((*itc)[0], (*itc)[1]), // 圆心
//	//		(*itc)[2], //半径
//	//		cv::Scalar(255), // 颜色
//	//		2); // 厚度
//
//	//	++itc;
//	//}
//
//	//cv::namedWindow("Detected Circles");
//	//cv::imshow("Detected Circles", image);
//
//	//5.提取连续区域
//    Mat image = imread("./images/binaryGroup.bmp", 0);
//	if (!image.data)
//		return 0;
//	imshow("Binary Image", image);
//	//用于存储轮廓的向量
//	vector<vector<Point> > contours;
//	findContours(image,
//		contours,           //存储轮廓的向量
//		RETR_EXTERNAL,      //检查外部轮廓
//		CHAIN_APPROX_NONE); //每个轮廓的全部像素
//
//	//查看轮廓得数量
//	cout << "Contours: " << contours.size() << endl;
//	vector<vector<Point> >::const_iterator itContours = contours.begin();
//	for (; itContours != contours.end(); ++itContours)
//	{
//		cout << "Size: " << itContours->size() << endl;
//	}
//	//在白色图像上画黑色轮廓
//	Mat result(image.size(), CV_8U, Scalar(255));
//	drawContours(result, contours,
//		-1,  //画出全部轮廓
//		0,   //用黑色画
//		2    //厚度为2
//	);
//	imshow("Contours", result);
//	//删除太短或者太长的轮廓
//	int cmin = 50;
//	int cmax = 500;
//	vector< vector<Point> > ::iterator itc = contours.begin();
//	while (itc != contours.end())
//	{
//		if (itc->size() < cmin || itc->size() > cmax)
//			itc = contours.erase(itc);
//		else
//			++itc;
//
//	}
//
//	// draw contours on the original image
//	cv::Mat original = cv::imread("./images/group.jpg");
//
//	cv::drawContours(original, contours,
//		-1, // draw all contours
//		cv::Scalar(255, 255, 255), // in white
//		2); // with a thickness of 2
//
//	cv::namedWindow("Contours on Animals");
//	cv::imshow("Contours on Animals", original);
//
//	result.setTo(Scalar(255));
//	drawContours(result, contours, -1, 0, 1);
//
//	image = imread("./images/binaryGroup.bmp", 0);
//
//	//需要思考的是这个二维的矩阵，行代表什么，列代表什么？
//	//首先说，行：代表了一个轮廓，每行的数据是一个数组，数组里面是一系列的点
//	//这些点代表了这个轮廓的位置
//	//所以，contours[0]，代表了第一个轮廓，contours[0][0]，代表了第一个轮廓的某个点
//
//	//A.测试边界框
//	Rect r0 = boundingRect(contours[0]);
//	//画矩形
//	rectangle(result, r0, 0, 2);
//	//B.测试覆盖圆
//	float radius;
//	Point2f center;
//	minEnclosingCircle(contours[1], center, radius);
//	//画圆型
//	circle(result, center, static_cast<int>(radius), 0, 2);
//	//C.测试多边形逼近
//	vector<Point> poly;
//	approxPolyDP(contours[2], poly, 5, true);
//	//画多边形
//	polylines(result, poly, true, 0, 2);
//
//	cout << "Polygon size: " << poly.size() << endl;
//	//D.测试凸包（另一种形式的多边形逼近）
//	std::vector<cv::Point> hull;
//	cv::convexHull(contours[3], hull);
//	//画多边形
//	cv::polylines(result, hull, true, 0, 2);
//	//检测轮廓矩
//	//迭代遍历所有轮廓
//	//在所有区域内部画出重心
//	itc = contours.begin();
//	while (itc != contours.end()) {
//
//		//计算所有矩形轮廓
//		cv::Moments mom = cv::moments(*itc++);
//
//		//画重心
//		cv::circle(result,
//			// 将重心位置转换成整数
//			cv::Point(mom.m10 / mom.m00, mom.m01 / mom.m00),
//			2, cv::Scalar(0), 2); // 画黑点
//	}
//
//	cv::namedWindow("Some Shape descriptors");
//	cv::imshow("Some Shape descriptors", result);
//
//	// 打开一副新的图像
//	image = cv::imread("./images/binaryGroup.bmp", 0);
//
//
//	cv::findContours(image,
//		contours, 
//		cv::RETR_LIST, 
//		cv::CHAIN_APPROX_NONE); 
//
//	result.setTo(255);
//	cv::drawContours(result, contours,
//		-1, 
//		0,  
//		2); 
//	cv::namedWindow("All Contours");
//	cv::imshow("All Contours", result);
//
//	// get a MSER image
//	cv::Mat components;
//	components = cv::imread("./images/mser.bmp", 0);
//
//	// create a binary version
//	components = components == 255;
//	// open the image (white background)
//	cv::morphologyEx(components, components, cv::MORPH_OPEN, cv::Mat(), cv::Point(-1, -1), 3);
//
//	cv::namedWindow("MSER image");
//	cv::imshow("MSER image", components);
//
//	contours.clear();
//	//翻转图像 (background must be black)
//	cv::Mat componentsInv = 255 - components;
//	//得到连续区域的轮廓
//	cv::findContours(componentsInv,
//		contours, 
//		cv::RETR_EXTERNAL, //检索外部向量
//		cv::CHAIN_APPROX_NONE); 
//
//	// white image
//	cv::Mat quadri(components.size(), CV_8U, 255);
//
//	//针对全部轮廓
//	std::vector<std::vector<cv::Point> >::iterator it = contours.begin();
//	while (it != contours.end()) {
//		poly.clear();
//		// 多边形逼近轮廓
//		cv::approxPolyDP(*it, poly, 5, true);
//
//		// 是否为四边形
//		if (poly.size() == 4) {
//			// 画出来
//			cv::polylines(quadri, poly, true, 0, 2);
//		}
//
//		++it;
//	}
//
//	cv::namedWindow("MSER quadrilateral");
//	cv::imshow("MSER quadrilateral", quadri);
//
//
//	waitKey(0);
//	return 0;
//}




#pragma once
#if !defineed SOBELEDGES
#define SOBELEDGES

#define PI 3.1415926

#include <iostream>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

class EdgeDetector {
private:

	// original image
	cv::Mat img;

	// 16-bit signed int image
	cv::Mat sobel;

	// Aperture size of the Sobel kernel
	int aperture;

	// Sobel magnitude
	cv::Mat sobelMagnitude;

	// Sobel orientation
	cv::Mat sobelOrientation;

public:

	EdgeDetector() : aperture(3) {}

	// Set the aperture size of the kernel
	void setAperture(int a) {

		aperture = a;
	}

	// Get the aperture size of the kernel
	int getAperture() const {

		return aperture;
	}

	// Compute the Sobel
	void computeSobel(const cv::Mat& image) {

		cv::Mat sobelX;
		cv::Mat sobelY;

		// Compute Sobel
		cv::Sobel(image, sobelX, CV_32F, 1, 0, aperture);
		cv::Sobel(image, sobelY, CV_32F, 0, 1, aperture);

		// 将笛卡尔坐标系转化为极坐标，得到幅值和角度
		cv::cartToPolar(sobelX, sobelY, sobelMagnitude, sobelOrientation);
	}

	// Compute the Sobel
	void computeSobel(const cv::Mat& image, cv::Mat &sobelX, cv::Mat &sobelY) {

		// Compute Sobel
		cv::Sobel(image, sobelX, CV_32F, 1, 0, aperture);
		cv::Sobel(image, sobelY, CV_32F, 0, 1, aperture);

		// Compute magnitude and orientation
		cv::cartToPolar(sobelX, sobelY, sobelMagnitude, sobelOrientation);
	}

	// Get Sobel magnitude
	cv::Mat getMagnitude() {

		return sobelMagnitude;
	}

	// Get Sobel orientation
	cv::Mat getOrientation() {

		return sobelOrientation;
	}

	// Get a thresholded binary map
	cv::Mat getBinaryMap(double threshold) {

		cv::Mat bin;
		cv::threshold(sobelMagnitude, bin, threshold, 255, cv::THRESH_BINARY_INV);

		return bin;
	}

	// Get a CV_8U image of the Sobel
	cv::Mat getSobelImage() {

		cv::Mat bin;

		double minval, maxval;
		cv::minMaxLoc(sobelMagnitude, &minval, &maxval);
		sobelMagnitude.convertTo(bin, CV_8U, 255 / maxval);

		return bin;
	}

	// Get a CV_8U image of the Sobel orientation
	// 1 gray-level = 2 degrees
	cv::Mat getSobelOrientationImage() {

		cv::Mat bin;

		sobelOrientation.convertTo(bin, CV_8U, 90 / PI);

		return bin;
	}

};

#endif // !defineed SOBELEDGES





#pragma once
#if !define LINEF
#define LINEF

#include <iostream>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;
#define PI 3.1415926

class LineFinder
{
private:
	//原始图像
	Mat img;
	//包含被检测直线的端点的向量
	vector<Vec4i> lines;
	//累加器分辨率参数
	double deltaRho;
	double deltaTheta;
	//确认直线之前必须收到的最小投票数
	int minVote;
	//直线的最小长度
	double minLength;
	//直线上允许的最大空隙
	double maxGap;

public:
	//默认累加器分辨率是1像素，1度
	//没有空隙，没有最小长度
	LineFinder() :deltaRho(1), deltaTheta(PI / 180), minVote(10), minLength(0.), maxGap(0.) {}

	//设置累加器的分辨率
	void setAccResolution(double dRho, double dTheta)
	{
		deltaRho = dRho;
		deltaTheta = dTheta;
	}
	//设置最小投票数
	void setMinVote(int minv) {

		minVote = minv;
	}

	//设置直线长度和空隙
	void setLineLengthAndGap(double length, double gap) {

		minLength = length;
		maxGap = gap;
	}
	//应用概率霍夫变换
	vector<Vec4i> findLines(Mat &binary)
	{
		lines.clear();
		HoughLinesP(binary, lines, deltaRho, deltaTheta, minVote, minLength, maxGap);

		return lines;
	}

	//在图像上绘制出检测出来的直线
	void drawDetectedLines(Mat &image, Scalar color = Scalar(255, 255, 255))
	{
		//画直线
		vector<Vec4i>::const_iterator it2 = lines.begin();

		while (it2 != lines.end())
		{
			Point pt1((*it2)[0], (*it2)[1]);
			Point pt2((*it2)[2], (*it2)[3]);

			line(image, pt1, pt2, color);

			++it2;
		}

	}

	// Eliminates lines that do not have an orientation equals to
	// the ones specified in the input matrix of orientations
	// At least the given percentage of pixels on the line must 
	// be within plus or minus delta of the corresponding orientation
	std::vector<cv::Vec4i> removeLinesOfInconsistentOrientations(
		const cv::Mat &orientations, double percentage, double delta) {

		std::vector<cv::Vec4i>::iterator it = lines.begin();

		// check all lines
		while (it != lines.end()) {

			// end points
			int x1 = (*it)[0];
			int y1 = (*it)[1];
			int x2 = (*it)[2];
			int y2 = (*it)[3];

			// line orientation + 90o to get the parallel line
			double ori1 = atan2(static_cast<double>(y1 - y2), static_cast<double>(x1 - x2)) + PI / 2;
			if (ori1>PI) ori1 = ori1 - 2 * PI;

			double ori2 = atan2(static_cast<double>(y2 - y1), static_cast<double>(x2 - x1)) + PI / 2;
			if (ori2>PI) ori2 = ori2 - 2 * PI;

			// for all points on the line
			cv::LineIterator lit(orientations, cv::Point(x1, y1), cv::Point(x2, y2));
			int i, count = 0;
			for (i = 0, count = 0; i < lit.count; i++, ++lit) {

				float ori = *(reinterpret_cast<float *>(*lit));

				// is line orientation similar to gradient orientation ?
				if (std::min(fabs(ori - ori1), fabs(ori - ori2))<delta)
					count++;

			}

			double consistency = count / static_cast<double>(i);

			// set to zero lines of inconsistent orientation
			if (consistency < percentage) {

				(*it)[0] = (*it)[1] = (*it)[2] = (*it)[3] = 0;

			}

			++it;
		}

		return lines;
	}

};


#endif // !define LINEF

```

### Chapter_08 : 检测兴趣点

> 1.Harris角点检测
> 2.GFTT(good-features-to-track)
> 3.drawKeypoints() 换关键点的通用函数
> 4.FAST角点检测
> 5.SURF尺度不变的特征检测
> 6.SIFT尺度不变特征转换
> 7.BRISK(二元稳健恒定可扩展关键点)检测法
> 8.ORB特征检测算法

<!--more-->

```c++

//#include <iostream>
//#include <opencv2/opencv.hpp>
//#include <opencv2/core.hpp>
//#include <opencv2/features2d.hpp>
//#include <opencv2/imgproc.hpp>
//#include <opencv2/highgui.hpp>
//#include <opencv2/xfeatures2d.hpp>
//
//#include "HarrisDetector.h"
//
//using namespace std;
//using namespace cv;
//
//int main()
//{
//	//1.Harris角点检测
//	Mat image = imread("./images/church01.jpg", 0);
//	if (!image.data)
//		return 0;
//	//变成水平的
//	transpose(image, image);
//	flip(image, image, 0);
//
//	imshow("Image", image);
//
//	//检测Harris角点
//	Mat cornerStrength;
//	cornerHarris(image, cornerStrength, 
//		         3,     //领域尺寸
//		         3,     //口径尺寸
//		         0.01); //Harris参数
//	//对角点强度进行阈值化
//	Mat harrisCorners;
//	double threshold = 0.0001;
//	cv::threshold(cornerStrength, harrisCorners, threshold, 255, THRESH_BINARY_INV);
//	imshow("Harris_threshold", harrisCorners);
//
//	//用一个类去封装Harris
//	HarrisDetected harris;
//	//计算Harris值
//	harris.detect(image);
//	//检测Harris角点
//	vector<Point> pts;
//	harris.getCorners(pts, 0.02);
//	//画出Harris角点
//	harris.drawOnImage(image, pts);
//
//	imshow("HarrisDetected", image);
//
//	//2.GFTT(good-features-to-track)
//	image = imread("./images/church01.jpg", 0);
//	// rotate the image (to produce a horizontal image)
//	cv::transpose(image, image);
//	cv::flip(image, image, 0);
//	//计算适合跟踪的特征
//	vector<KeyPoint> keypoints;
//	//GFTT检测器
//	Ptr<GFTTDetector> ptrGFTT = GFTTDetector::create(
//		500,   //关键点的最大值
//		0.01,  //质量等级
//		10     //角点之间允许的最短距离
//	);
//	//检测GFTT
//	ptrGFTT->detect(image, keypoints);
//	//展示所有关键点
//	vector<KeyPoint>::const_iterator it = keypoints.begin();
//	while (it != keypoints.end())
//	{
//		//对每个关键点画圆
//		circle(image, it->pt, 3, Scalar(255, 255, 255), 1);
//		++it;
//	}
//
//	imshow("GFTT", image);
//	//3.drawKeypoints() 换关键点的通用函数
//	// Read input image
//	image = cv::imread("./images/church01.jpg", 0);
//	// rotate the image (to produce a horizontal image)
//	cv::transpose(image, image);
//	cv::flip(image, image, 0);
//
//	// Opencv也提供了在图像上画关键点的通用函数
//	cv::drawKeypoints(image,		// 原始图像
//		keypoints,					// 关键点的向量
//		image,						// 输出图像
//		cv::Scalar(255, 255, 255),	// 关键点颜色
//		cv::DrawMatchesFlags::DRAW_OVER_OUTIMG); //画图标志
//
//	// Display the keypoints
//	cv::namedWindow("Good Features to Track Detector");
//	cv::imshow("Good Features to Track Detector", image);
//
//	//4.FAST角点检测
//	image = imread("./images/church01.jpg", 0);
//	transpose(image, image);
//	flip(image, image, 0);
//	//最终的关键点容器
//	keypoints.clear();
//	//FAST特征检测器，阈值为40
//	Ptr<FastFeatureDetector> ptrFAST = FastFeatureDetector::create(40);
//	//检测关键点
//	ptrFAST->detect(image, keypoints);
//	//画关键点
//	drawKeypoints(image, keypoints, image, Scalar(255, 255,255), DrawMatchesFlags::DRAW_OVER_OUTIMG);
//	cout << "Number of keypoints(FAST): " << keypoints.size() << endl;
//
//	imshow("FAST_01", image);
//	//FAST_02----非极大抑制
//	image = imread("./images/church01.jpg", 0);
//	transpose(image, image);
//	flip(image, image, 0);
//
//	keypoints.clear();
//	//检测关键点
//	ptrFAST->setNonmaxSuppression(false);
//	ptrFAST->detect(image, keypoints);
//	//画出关键点
//	drawKeypoints(image, keypoints, image, Scalar(255, 255, 255), DrawMatchesFlags::DRAW_OVER_OUTIMG);
//	imshow("FAST FEATURES(ALL)", image);
//	//FAST_03----Grid
//	image = imread("./images/church01.jpg", 0);
//	transpose(image, image);
//	flip(image, image, 0);
//
//	int total(100);
//	int hstep(5), vstep(3);
//	int hsize(image.cols / hstep), vsize(image.rows / vstep);
//	int subtotal(total / (hstep * vstep));
//
//	Mat imageROI;
//	vector<KeyPoint> gridpoints;
//	cout << "Grid of" << vstep << "by" << hstep << "each of size" << vsize << "by" << hsize << endl;
//
//	//检测低阈值
//	ptrFAST->setThreshold(20);
//	//非极大抑制
//	ptrFAST->setNonmaxSuppression(true);
//	//最终的关键点容器
//	keypoints.clear();
//	//检测每一个网格
//	for (int i = 0; i < vstep; i++)
//	{
//		for (int j = 0; j < hstep; j++)
//		{
//			//在当前网格创建ROI
//			imageROI = image(Rect(j * hsize, i * vsize, hsize, vsize));
//			//在网格中检测关键点
//			gridpoints.clear();
//			ptrFAST->detect(imageROI, gridpoints);
//			cout << "Number of FAST in grid" << i << "," << j << ":" << gridpoints.size() << endl;
//			if (gridpoints.size() > subtotal)
//			{
//				for (auto it = gridpoints.begin(); it != gridpoints.begin() + subtotal; ++it)
//				{
//					std::cout << "  " << it->response << std::endl;
//				}
//			}
//			//获取最大强度的FAST特征
//			auto itEnd(gridpoints.end());
//			if (gridpoints.size() > subtotal)
//			{
//				//选取最强的特征
//				nth_element(gridpoints.begin(), gridpoints.begin() + subtotal, gridpoints.end(),
//					[](KeyPoint &a, KeyPoint &b) {return a.response > b.response; });
//
//				itEnd = gridpoints.begin() + subtotal;
//			}
//			//加入全局特征容器
//			for (auto it = gridpoints.begin(); it != itEnd; ++it)
//			{
//				//转换成图上的坐标
//				it->pt += Point2f(j * hsize, i * vsize);
//				keypoints.push_back(*it);
//				cout << "  " << it->response << std::endl;
//			}
//		}
//	}
//	//画关键点
//	drawKeypoints(image, keypoints, image, cv::Scalar(255, 255, 255), cv::DrawMatchesFlags::DRAW_OVER_OUTIMG);
//
//	imshow("FAST Features (grid)", image);
//
//	//5.SURF尺度不变的特征检测
//	image = imread("./images/church01.jpg", 0);
//	transpose(image, image);
//	flip(image, image, 0);
//   
//	keypoints.clear();
//	//创建SURF特征检测器对象
//	Ptr<xfeatures2d::SurfFeatureDetector> ptrSURF = xfeatures2d::SurfFeatureDetector::create(2000.0);
//	//检测关键点
//	ptrSURF->detect(image, keypoints);
//	//画出关键点，包括尺度和方向信息
//	Mat featureImage;
//	drawKeypoints(image, keypoints, featureImage, Scalar(255, 255, 255), DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
//
//	imshow("SURF", featureImage);
//
//	cout << "Number of SURF keypoints: " << keypoints.size() << endl;
//
//	// Read a second input image
//	image = cv::imread("./images/church03.jpg", cv::IMREAD_GRAYSCALE);
//	// rotate the image (to produce a horizontal image)
//	cv::transpose(image, image);
//	cv::flip(image, image, 0);
//
//	// Detect the SURF features
//	ptrSURF->detect(image, keypoints);
//
//	cv::drawKeypoints(image, keypoints, featureImage, cv::Scalar(255, 255, 255), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
//
//	// Display the keypoints
//	cv::namedWindow("SURF (2)");
//	cv::imshow("SURF (2)", featureImage);
//
//	//6.SIFT尺度不变特征转换
//	image = imread("./images/church01.jpg", IMREAD_GRAYSCALE);
//	transpose(image, image);
//	flip(image, image, 0);
//
//	keypoints.clear();
//	Ptr<xfeatures2d::SiftFeatureDetector> ptrSIFT = xfeatures2d::SiftFeatureDetector::create();
//	ptrSIFT->detect(image, keypoints);
//	//画关键点
//	drawKeypoints(image, keypoints, featureImage, Scalar(255, 255,255), DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
//	
//	imshow("SIFT", featureImage);
//	cout << "Number of SIFT keypoints: " << keypoints.size() << endl;
//
//
//	//7.BRISK(二元稳健恒定可扩展关键点)检测法
//	image = imread("./images/church01.jpg", CV_LOAD_IMAGE_GRAYSCALE);
//	transpose(image, image);
//	flip(image, image, 0);
//
//	keypoints.clear();
//	//构造BRISK特征检测器对象
//	Ptr<BRISK> ptrBRISK = BRISK::create(60, 5);
//	//检测关键点
//	ptrBRISK->detect(image, keypoints);
//	drawKeypoints(image, keypoints, featureImage, Scalar(255, 255, 255), DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
//	imshow("BRISK", featureImage);
//	cout << "Number of BRISK keypoints: " << keypoints.size() << endl;
//
//	//8.ORB特征检测算法
//	image = imread("./images/church01.jpg", CV_LOAD_IMAGE_GRAYSCALE);
//	transpose(image, image);
//	flip(image, image, 0);
//
//	keypoints.clear();
//	//构造ORB特征检测器
//	Ptr<ORB> ptrORB = ORB::create(75, 1.2, 8);
//	//检测关键点
//	ptrORB->detect(image, keypoints);
//	//画出关键点
//	drawKeypoints(image, keypoints, featureImage, Scalar(255, 255, 255), DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
//
//	imshow("ORB", featureImage);
//	cout << "Number of ORB keypoints: " << keypoints.size() << endl;
//
//
//
//
//
//	waitKey(0);
//	return 0;
//}





#pragma once
#if !define HARRISD
#define HARRISD

#include <iostream>
#include <opencv2/opencv.hpp>
#include <vector>

using namespace std;
using namespace cv;

class HarrisDetected
{
private:
	//32位浮点整数的角点强度图像
	Mat cornerStrength;
	//32位浮点整数的阈值化角点图像
	Mat cornerTh;
	//局部最大值图像（内部）
	Mat localMax;
	//平滑导数的领域尺寸
	int neighborhood;
	//梯度计算的口径
	int aperture;
	//Harris参数
	double k;
	//阈值计算的最大强度
	double maxStrength;
	//计算得到的阈值（内部）
	double threshold;
	//非最大值抑制的领域尺寸
	double nonMaxSize;
	//非最大值抑制的内核
	Mat kernel;

public:
	HarrisDetected() : neighborhood(3), aperture(3), k(0.1), maxStrength(0.0), threshold(0.01), nonMaxSize(3)
	{
		setLocalMaxWindowSize(nonMaxSize);
	}
	//创建用于抑制非最大值的内核
	void setLocalMaxWindowSize(int size)
	{
		nonMaxSize = size;
		kernel.create(nonMaxSize, nonMaxSize, CV_8U);
	}
	//计算Harris角点
	void detect(const Mat& image)
	{
		//计算Harris
		cornerHarris(image, cornerStrength, neighborhood, aperture, k);
		//计算内部阈值
		minMaxLoc(cornerStrength, 0, &maxStrength);
		//检测局部最大值
		Mat dilated;  //临时图像
		dilate(cornerStrength, dilated, Mat());
		compare(cornerStrength, dilated, localMax, CMP_EQ);
	}

	//用Harris值得到角点分布图
	Mat getCornerMap(double qualityLeval)
	{
		Mat cornerMap;
		//对角点强度进行阈值化
		threshold = qualityLeval * maxStrength;
		cv::threshold(cornerStrength, cornerTh, threshold, 255, THRESH_BINARY);
		//转换成8位图像
		cornerTh.convertTo(cornerMap, CV_8U);
		//非最大值抑制
		bitwise_and(cornerMap, localMax, cornerMap);

		return cornerMap;

	}

	//用Harris值得到特征点
	void getCorners(vector<Point> &points, double qualityLevel)
	{
		//获得角点分布图
		Mat cornerMap = getCornerMap(qualityLevel);
		//获得角点
		getCorners(points, cornerMap);
	}
	//用角点分布图得到特征点
	void getCorners(vector<Point> &points, const Mat &cornerMap)
	{
		//迭代遍历像素，得到所有特征
		for (int y = 0; y < cornerMap.rows; y++)
		{
			const uchar *rowPtr = cornerMap.ptr<uchar>(y);

			for (int x = 0; x < cornerMap.cols; x++)
			{
				//如果他是一个特征点
				if (rowPtr[x])
				{
					points.push_back(Point(x, y));
				}
			}
		}
	}

	//在特征点的位置画圆
	void drawOnImage(Mat &image, const vector<Point> &points, Scalar color = Scalar(255, 255, 255), int radius = 3, int thickness = 1)
	{
		vector <Point> ::const_iterator it = points.begin();
		//针对所有角点
		while (it != points.end())
		{
			//在每个角点位置画圆
			circle(image, *it, radius, color, thickness);
			++it;
		}
	}
};

#endif // !define HARRISD

```

### Chapter_09 : 描述和匹配兴趣点

> 1.局部模板匹配----两个图像的匹配和找一样的模板
> 2.描述并匹配局部强度值模式----SURF和SIFT
> 3.用二值描述子匹配关键点

<!--more-->

```c++
//#include <iostream>
//#include <opencv2/core.hpp>
//#include <opencv2/imgproc.hpp>
//#include <opencv2/highgui.hpp>
//#include <opencv2/features2d.hpp>
//#include <opencv2/objdetect.hpp>
//#include <opencv2/xfeatures2d.hpp>
//
//using namespace std;
//using namespace cv;
//
//int main()
//{
//	////1.局部模板匹配 ---- 两个图像的匹配
//	//Mat image1 = imread("./images/church01.jpg", CV_LOAD_IMAGE_GRAYSCALE);
//	//Mat image2 = imread("./images/church02.jpg", CV_LOAD_IMAGE_GRAYSCALE);
//
//	//imshow("image1", image1);
//	//imshow("image2", image2);
//	////定义关键点容器
//	//vector<KeyPoint> keypoints1;
//	//vector<KeyPoint> keypoints2;
//	////定义特征检测器
//	//Ptr<FeatureDetector> ptrDetector;  //泛型检测器指针
//	////这里使用FAST检测器
//	//ptrDetector = FastFeatureDetector::create(80);
//	////检测关键点
//	//ptrDetector->detect(image1, keypoints1);
//	//ptrDetector->detect(image2, keypoints2);
//
//	//std::cout << "Number of keypoints (image 1): " << keypoints1.size() << std::endl;
//	//std::cout << "Number of keypoints (image 2): " << keypoints2.size() << std::endl;
//
//	////定义一个特定大小的矩形（11 * 11）， 用于表示每个关键点周围的图像块
//	////定义正方形的领域
//	//const int nsize(11);
//	//Rect neighborhood(0, 0, nsize, nsize); //11 * 11
//	//Mat patch1;
//	//Mat patch2;
//	////将一副图像的关键点与另一幅图像的所有关键点进行比较，找出最相似的
//	////在第二幅图像中找出与第一幅图像中的每个关键点最匹配的
//	//Mat result;
//	//vector<DMatch> matches;
//	////针对图像一的全部关键点
//	//for (int i = 0; i < keypoints1.size(); i++)
//	//{
//	//	//定义图像块
//	//	neighborhood.x = keypoints1[i].pt.x - nsize / 2;
//	//	neighborhood.y = keypoints1[i].pt.y - nsize / 2;
//	//	//如果领域超出图像范围，就继续处理下一个点
//	//	if (neighborhood.x < 0 || neighborhood.y < 0 || 
//	//		neighborhood.x + nsize >= image1.cols || neighborhood.y + nsize >= image1.rows)
//	//		continue;
//	//	//第一幅图像的块
//	//	patch1 = image1(neighborhood);
//	//	//存放最匹配的值
//	//	DMatch bestMatch;
//	//	//针对第二幅图像的全部关键点
//	//	for (int j = 0; j < keypoints2.size(); j++)
//	//	{
//	//		//定义图像块
//	//		neighborhood.x = keypoints2[j].pt.x - nsize / 2;
//	//		neighborhood.y = keypoints2[j].pt.y - nsize / 2;
//
//	//		//如果领域超出图像范围，就继续处理下一个点
//	//		if (neighborhood.x < 0 || neighborhood.y < 0 ||
//	//			neighborhood.x + nsize >= image2.cols || neighborhood.y + nsize >= image2.rows)
//	//			continue;
//	//		//第二幅图像的块
//	//		patch2 = image2(neighborhood);
//
//	//	    //匹配两个图像块
//	//		matchTemplate(patch1, patch2, result, TM_SQDIFF);
//	//		//cv::matchTemplate(patch1, patch2, result, cv::TM_SQDIFF);
//	//		//检查是否为最佳匹配
//	//		if (result.at<float>(0, 0) < bestMatch.distance)
//	//		{
//	//			bestMatch.distance = result.at <float>(0, 0);
//	//			bestMatch.queryIdx = i;
//	//			bestMatch.trainIdx = j;
//	//		}
//	//	}
//	//	//添加最佳匹配
//	//	matches.push_back(bestMatch);
//	//}
//
//	//std::cout << "Number of matches: " << matches.size() << std::endl;
//	////提取50个最佳匹配项
//	//nth_element(matches.begin(), matches.begin() + 50, matches.end());
//	//matches.erase(matches.begin() + 50, matches.end());
//
//	//std::cout << "Number of matches (after): " << matches.size() << std::endl;
//
//	////画出匹配结果
//	//Mat matchImage;
//	//drawMatches(image1, keypoints1, image2, keypoints2, matches, matchImage, Scalar(255, 255, 255), Scalar(255, 255, 255));
//
//	//imshow("Matches", matchImage);
//
//	////模板匹配----寻找一样的部分
//	////定义一个模板
//	//Mat target(image1, Rect(80, 105, 30, 30));
//	////展示模板
//	//imshow("Template", target);
//	////定义搜索区域-----这里用图像的上半部分
//	//Mat roi(image2, Rect(0, 0, image2.cols, image2.rows / 2));
//	////进行模板匹配
//	//matchTemplate(roi,  //搜索区域
//	//	target,         //模板
//	//	result,         //结果
//	//	TM_SQDIFF);     //相似度
//	////找到最相似的位置
//	//double minVal, maxVal;
//	//Point minPt, maxPt;
//	//minMaxLoc(result, &minVal, &maxVal, &minPt, &maxPt);
//	////在相似度最高的位置画矩形框
//	////本例为minPt
//	//rectangle(roi, Rect(minPt.x, minPt.y, target.cols, target.rows), 255);
//	////展示模板
//	//imshow("Best", image2);
//
//
//	//////2.描述并匹配局部强度值模式----SURF和SIFT
//	////SURF
//	////读图片
//	//Mat image1 = imread("./images/church01.jpg", IMREAD_GRAYSCALE);
//	//Mat image2 = imread("./images/church02.jpg", IMREAD_GRAYSCALE);
//
//	//if (!image1.data || !image2.data)
//	//	return 0;
//	//imshow("image1", image1);
//	//imshow("image2", image2);
//	////定义关键点的检测器
//	//vector<KeyPoint> keypoints1;
//	//vector<KeyPoint> keypoints2;
//	////定义特征检测器----SURF
//	//Ptr<Feature2D> ptrFeature2D = xfeatures2d::SURF::create(2000.0);
//	////检测关键点
//	//ptrFeature2D->detect(image1, keypoints1);
//	//ptrFeature2D->detect(image2, keypoints2);
//
//	////画出特征点
//	//cv::Mat featureImage;
//	//cv::drawKeypoints(image1, keypoints1, featureImage, cv::Scalar(255, 255, 255), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
//
//	//cv::namedWindow("SURF");
//	//cv::imshow("SURF", featureImage);
//
//	//std::cout << "Number of SURF keypoints (image 1): " << keypoints1.size() << std::endl;
//	//std::cout << "Number of SURF keypoints (image 2): " << keypoints2.size() << std::endl;
//
//	////提取描述子
//	//Mat descriptors1;
//	//Mat descriptors2;
//	//ptrFeature2D->compute(image1, keypoints1, descriptors1);
//	//ptrFeature2D->compute(image2, keypoints2, descriptors2);
//	////构造匹配器
//	//BFMatcher matcher(NORM_L2);  //度量距离
//	////匹配两幅图片的描述子
//	//vector<DMatch> matches;
//	//matcher.match(descriptors1, descriptors2, matches);
//	////画出匹配的描述子
//	//Mat imageMatches;
//	//drawMatches(image1, keypoints1, image2, keypoints2, matches, imageMatches, 
//	//	Scalar(255, 255, 255), Scalar(255, 255, 255), vector<char>(), 
//	//	DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS | DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
//	//imshow("SURF Matches", imageMatches);
//	//cout << "Number of matches: " << matches.size() << endl;
//
//	////对于每个关键点找出最符合的2个匹配点 ----比率检验法
//	//std::vector<std::vector<cv::DMatch> > matches2;
//	//matcher.knnMatch(descriptors1, descriptors2,
//	//	matches2,
//	//	2); //找出k个最佳匹配项
//	//matches.clear();
//
//	//// 执行比率检验法
//	//double ratioMax = 0.6;
//	//std::vector<std::vector<cv::DMatch> >::iterator it;
//	//for (it = matches2.begin(); it != matches2.end(); ++it) {
//	//	//  第一个最佳匹配项/第二个最佳匹配项
//	//	if ((*it)[0].distance / (*it)[1].distance < ratioMax) {
//	//		// 这个匹配项可以接收
//	//		matches.push_back((*it)[0]);
//
//	//	}
//	//}
//	//// 
//	//cv::drawMatches(
//	//	image1, keypoints1, // 1st image and its keypoints
//	//	image2, keypoints2, // 2nd image and its keypoints
//	//	matches,           // the matches
//	//	imageMatches,      // the image produced
//	//	cv::Scalar(255, 255, 255),  // color of lines
//	//	cv::Scalar(255, 255, 255),  // color of points
//	//	std::vector< char >(),    // masks if any 
//	//	cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS | cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
//
//	//std::cout << "Number of matches (after ratio test): " << matches.size() << std::endl;
//
//	//cv::namedWindow("SURF Matches (ratio test at 0.6)");
//	//cv::imshow("SURF Matches (ratio test at 0.6)", imageMatches);
//
//	////匹配差值的阈值化
//	//// 指定范围的匹配
//	//float maxDist = 0.3;
//	//matches2.clear();
//	//matcher.radiusMatch(descriptors1, descriptors2, matches2,maxDist); //两个描述子之间的最大允许差值
//	//			
//	//cv::drawMatches(
//	//	image1, keypoints1, // 1st image and its keypoints
//	//	image2, keypoints2, // 2nd image and its keypoints
//	//	matches2,          // the matches
//	//	imageMatches,      // the image produced
//	//	cv::Scalar(255, 255, 255),  // color of lines
//	//	cv::Scalar(255, 255, 255),  // color of points
//	//	std::vector<std::vector< char >>(),    // masks if any 
//	//	cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS | cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
//
//	//int nmatches = 0;
//	//for (int i = 0; i< matches2.size(); i++)
//	//	nmatches += matches2[i].size();
//
//	//std::cout << "Number of matches (with max radius): " << nmatches << std::endl;
//
//	//cv::namedWindow("SURF Matches (with max radius)");
//	//cv::imshow("SURF Matches (with max radius)", imageMatches);
//
//
//	////SIFT
//	//image1 = cv::imread("./images/church01.jpg", CV_LOAD_IMAGE_GRAYSCALE);
//	//image2 = cv::imread("./images/church03.jpg", CV_LOAD_IMAGE_GRAYSCALE);
//
//	//std::cout << "Number of SIFT keypoints (image 1): " << keypoints1.size() << std::endl;
//	//std::cout << "Number of SIFT keypoints (image 2): " << keypoints2.size() << std::endl;
//
//	//imshow("image1_sift", image1);
//	//imshow("image2_sift", image2);
//
//	//ptrFeature2D = xfeatures2d::SIFT::create();
//	//ptrFeature2D->detectAndCompute(image1, noArray(), keypoints1, descriptors1);
//	//ptrFeature2D->detectAndCompute(image2, noArray(), keypoints2, descriptors2);
//
//	//matcher.match(descriptors1, descriptors2, matches);
//
//	//// extract the 50 best matches
//	//std::nth_element(matches.begin(), matches.begin() + 50, matches.end());
//	//matches.erase(matches.begin() + 50, matches.end());
//
//	//// draw matches
//	//cv::drawMatches(
//	//	image1, keypoints1, // 1st image and its keypoints
//	//	image2, keypoints2, // 2nd image and its keypoints
//	//	matches,            // the matches
//	//	imageMatches,      // the image produced
//	//	cv::Scalar(255, 255, 255),  // color of lines
//	//	cv::Scalar(255, 255, 255), // color of points
//	//	std::vector<char>(), cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS | cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
//	//// Display the image of matches
//	//cv::namedWindow("Multi-scale SIFT Matches");
//	//cv::imshow("Multi-scale SIFT Matches", imageMatches);
//
//	//std::cout << "Number of matches: " << matches.size() << std::endl;
//
//
//
//
//
//	//3.用二值描述子匹配关键点
//    Mat image1 = imread("./images/church01.jpg", CV_LOAD_IMAGE_GRAYSCALE);
//	Mat image2 = imread("./images/church02.jpg", CV_LOAD_IMAGE_GRAYSCALE);
//
//	//定义关键点容器和描述子
//	vector<KeyPoint> keypoints1;
//	vector<KeyPoint> keypoints2;
//	Mat descriptors1;
//	Mat descriptors2;
//	//定义特征检测器/描述子
//	Ptr<Feature2D> feature = ORB::create(60);   //大约60个特征点
//	//检测并描述关键点
//	//检测ORB特征
//	feature->detectAndCompute(image1, noArray(), keypoints1, descriptors1);
//	feature->detectAndCompute(image2, noArray(), keypoints2, descriptors2);
//	
//	Mat featureImage;
//	drawKeypoints(image1, keypoints1, featureImage, Scalar(255, 255, 255), DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
//	//展示角点
//	imshow("ORB", featureImage);
//
//	std::cout << "Number of ORB keypoints (image 1): " << keypoints1.size() << std::endl;
//	std::cout << "Number of ORB keypoints (image 2): " << keypoints2.size() << std::endl;
//
//	//构建匹配器
//	BFMatcher matcher(NORM_HAMMING);
//	//匹配两幅图像的描述子
//	vector<DMatch> matches;
//	matcher.match(descriptors1, descriptors2, matches);
//
//	cv::Mat imageMatches;
//	cv::drawMatches(
//		image1, keypoints1, // 1st image and its keypoints
//		image2, keypoints2, // 2nd image and its keypoints
//		matches,           // the matches
//		imageMatches,      // the image produced
//		cv::Scalar(255, 255, 255),  // color of lines
//		cv::Scalar(255, 255, 255),  // color of points
//		std::vector< char >(),    // masks if any 
//		cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS | cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
//
//	cv::imshow("ORB Matches", imageMatches);
//	std::cout << "Number of matches: " << matches.size() << std::endl;
//
//
//	waitKey(0);
//	return 0;
//}
```

### Chapter_10 : 估算图像之间的投影关系

> 1.计算图像对的基础矩阵
> 2.用RANSAC算法匹配图像
> 3.计算两幅图像之间的单应矩阵----找到对应的点和拼接两幅图像
> 4.检测图像中的平面目标

<!--more-->

```c++
//#include <iostream>
//#include <vector>
//#include <opencv2/core.hpp>
//#include <opencv2/imgproc.hpp>
//#include <opencv2/highgui.hpp>
//#include <opencv2/features2d.hpp>
//#include <opencv2/calib3d.hpp>
//#include <opencv2/objdetect.hpp>
//#include <opencv2/xfeatures2d.hpp>
//#include <opencv2/stitching.hpp>
//
//#include "RobustMatcher.h"
//#include "TargetMatcher.h"
//
//using namespace std;
//using namespace cv;
//
//int main()
//{
//	////1.计算图像对的基础矩阵
//	//Mat image1 = imread("./images/church01.jpg", 0);
//	//Mat image2 = imread("./images/church03.jpg", 0);
//	//if (!image1.data || !image2.data)
//	//	return 0;
//
//	//// Display the images
//	//cv::namedWindow("Right Image");
//	//cv::imshow("Right Image", image1);
//	//cv::namedWindow("Left Image");
//	//cv::imshow("Left Image", image2);
//
//	////定义关键点容器和描述子、
//	//vector<KeyPoint> keypoints1;
//	//vector<KeyPoint> keypoints2;
//	//Mat descriptors1, descriptors2;
//	////构建SIFT特征检测器
//	//Ptr<Feature2D> ptrFeature2D = xfeatures2d::SIFT::create(74);
//
//	//ptrFeature2D->detectAndCompute(image1, noArray(), keypoints1, descriptors1);
//	//ptrFeature2D->detectAndCompute(image2, noArray(), keypoints2, descriptors2);
//
//	//std::cout << "Number of SIFT points (1): " << keypoints1.size() << std::endl;
//	//std::cout << "Number of SIFT points (2): " << keypoints2.size() << std::endl;
//
//	////画关键点
//	//cv::Mat imageKP;
//	//cv::drawKeypoints(image1, keypoints1, imageKP, cv::Scalar(255, 255, 255), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
//	//cv::namedWindow("Right SIFT Features");
//	//cv::imshow("Right SIFT Features", imageKP);
//	//cv::drawKeypoints(image2, keypoints2, imageKP, cv::Scalar(255, 255, 255), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
//	//cv::namedWindow("Left SIFT Features");
//	//cv::imshow("Left SIFT Features", imageKP);
//
//	////构建匹配类的实例
//	//BFMatcher matcher(NORM_L2, true);
//	////匹配描述子
//	//vector<DMatch> matches;
//	//matcher.match(descriptors1, descriptors2, matches);
//	//std::cout << "Number of matched points: " << matches.size() << std::endl;
//	////手动的选择一些匹配的描述子
//	//vector<DMatch> selMatches;
//	//// make sure to double-check if the selected matches are valid
//	//selMatches.push_back(matches[2]);
//	//selMatches.push_back(matches[5]);
//	//selMatches.push_back(matches[16]);
//	//selMatches.push_back(matches[19]);
//	//selMatches.push_back(matches[14]);
//	//selMatches.push_back(matches[34]);
//	//selMatches.push_back(matches[29]);
//
//	////画出选择的描述子
//	//cv::Mat imageMatches;
//	//cv::drawMatches(image1, keypoints1,  // 1st image and its keypoints
//	//	image2, keypoints2,  // 2nd image and its keypoints
//	//	selMatches,			// the selected matches
//	//	imageMatches,		// the image produced
//	//	cv::Scalar(255, 255, 255),
//	//	cv::Scalar(255, 255, 255),
//	//	std::vector<char>(),
//	//	2
//	//); // color of the lines
//	//cv::namedWindow("Matches");
//	//cv::imshow("Matches", imageMatches);
//	////将一维关键点转变为二维的点
//	//vector<int> pointIndexes1;
//	//vector<int> pointIndexes2;
//	//for (vector<DMatch>::const_iterator it = selMatches.begin(); it != selMatches.end(); ++it)
//	//{
//	//	pointIndexes1.push_back(it->queryIdx);
//	//	pointIndexes2.push_back(it->trainIdx);
//	//}
//	////为了在findFundamentalMat中使用，需要先把这些关键点转化为Point2f类型
//	//vector<Point2f> selPoints1, selPoints2;
//	//KeyPoint::convert(keypoints1, selPoints1, pointIndexes1);
//	//KeyPoint::convert(keypoints2, selPoints2, pointIndexes2);
//	////通过画点来检查
//	//vector<Point2f> ::const_iterator it = selPoints1.begin();
//	//while (it != selPoints1.end())
//	//{
//	//	//在每个角点位置画圆
//	//	circle(image1, *it, 3, Scalar(255, 255, 255), 2);
//	//	++it;
//	//}
//	//it = selPoints2.begin();
//	//while (it != selPoints2.end()) {
//
//	//	// draw a circle at each corner location
//	//	cv::circle(image2, *it, 3, cv::Scalar(255, 255, 255), 2);
//	//	++it;
//	//}
//	////用7对匹配项计算基础矩阵
//	//Mat fundamental = findFundamentalMat(
//	//    selPoints1, //第一幅图像的7个点
//	//    selPoints2, //第二幅图像的7个点
//	//	FM_7POINT); //7个点的方法
//	//cout << "F-Matrix size= " << fundamental.rows << "," << fundamental.cols << endl;
//	//Mat fund(fundamental, cv::Rect(0, 0, 3, 3));
//	////在右侧图像上画出对极线的左侧点
//	//vector<Vec3f> lines1;
//	//computeCorrespondEpilines(selPoints1, //图像点
//	//	1,                 //在第一副图像中（也可在第二幅图像中）
//	//	fund,       //基础矩阵
//	//	lines1);           //对极线的向量
//	//std::cout << "size of F matrix:" << fund.rows << "x" << fund.cols << std::endl;
//	////遍历全部对极线
//	//for (std::vector<cv::Vec3f>::const_iterator it = lines1.begin();
//	//	it != lines1.end(); ++it) {
//
//	//	// 画出第一列和最后一列之间的线条
//	//	cv::line(image2, cv::Point(0, -(*it)[2] / (*it)[1]),
//	//		cv::Point(image2.cols, -((*it)[2] + (*it)[0] * image2.cols) / (*it)[1]),
//	//		cv::Scalar(255, 255, 255));
//	//}
//
//	//// draw the left points corresponding epipolar lines in left image 
//	//std::vector<cv::Vec3f> lines2;
//	//cv::computeCorrespondEpilines(cv::Mat(selPoints2), 2, fund, lines2);
//	//for (std::vector<cv::Vec3f>::const_iterator it = lines2.begin();
//	//	it != lines2.end(); ++it) {
//
//	//	// draw the epipolar line between first and last column
//	//	cv::line(image1, cv::Point(0, -(*it)[2] / (*it)[1]),
//	//		cv::Point(image1.cols, -((*it)[2] + (*it)[0] * image1.cols) / (*it)[1]),
//	//		cv::Scalar(255, 255, 255));
//	//}
//
//	//// combine both images
//	//cv::Mat both(image1.rows, image1.cols + image2.cols, CV_8U);
//	//image1.copyTo(both.colRange(0, image1.cols));
//	//image2.copyTo(both.colRange(image1.cols, image1.cols + image2.cols));
//
//	//// Display the images with points and epipolar lines
//	//cv::namedWindow("Epilines");
//	//cv::imshow("Epilines", both);
//	///*
//	//// Convert keypoints into Point2f
//	//std::vector<cv::Point2f> points1, points2, newPoints1, newPoints2;
//	//cv::KeyPoint::convert(keypoints1, points1);
//	//cv::KeyPoint::convert(keypoints2, points2);
//	//cv::correctMatches(fund, points1, points2, newPoints1, newPoints2);
//	//cv::KeyPoint::convert(newPoints1, keypoints1);
//	//cv::KeyPoint::convert(newPoints2, keypoints2);
//
//	//cv::drawMatches(image1, keypoints1,  // 1st image and its keypoints
//	//image2, keypoints2,  // 2nd image and its keypoints
//	//matches,			// the matches
//	//imageMatches,		// the image produced
//	//cv::Scalar(255, 255, 255),
//	//cv::Scalar(255, 255, 255),
//	//std::vector<char>(),
//	//2
//	//); // color of the lines
//	//cv::namedWindow("Corrected matches");
//	//cv::imshow("Corrected matches", imageMatches);
//	//*/
//
//
//
//	//2.用RANSAC算法匹配图像
//    
//	// Read input images
// //   cv::Mat image1 = cv::imread("./images/church01.jpg", 0);
// //   cv::Mat image2 = cv::imread("./images/church03.jpg", 0);
//
// //   if (!image1.data || !image2.data)
// //        return 0;
//
// //   // Display the images
// //   cv::namedWindow("Right Image");
// //   cv::imshow("Right Image", image1);
// //   cv::namedWindow("Left Image");
//	//cv::imshow("Left Image", image2);
//
//	////准备匹配器（用默认参数）
//	////SIFT检测器和描述子
//	//RobustMatcher rmatcher(xfeatures2d::SIFT::create(250));
//	////匹配两幅图像
//	//vector<DMatch> matches;
//
//	//vector<KeyPoint> keypoints1, keypoints2;
//	//Mat fundamental = rmatcher.match(image1, image2, matches, keypoints1, keypoints2);
//
//	//Mat imageMatches;
//	//drawMatches(image1, keypoints1, image2, keypoints2, matches, imageMatches,
//	//	Scalar(255, 255, 255), Scalar(255, 255, 255), vector<char>(), 2);
//
//	//imshow("Matches", imageMatches);
//
//	//// Convert keypoints into Point2f	
//	//std::vector<cv::Point2f> points1, points2;
//
//	//for (std::vector<cv::DMatch>::const_iterator it = matches.begin();
//	//	it != matches.end(); ++it) {
//
//	//	// Get the position of left keypoints
//	//	float x = keypoints1[it->queryIdx].pt.x;
//	//	float y = keypoints1[it->queryIdx].pt.y;
//	//	points1.push_back(keypoints1[it->queryIdx].pt);
//	//	cv::circle(image1, cv::Point(x, y), 3, cv::Scalar(255, 255, 255), 3);
//	//	// Get the position of right keypoints
//	//	x = keypoints2[it->trainIdx].pt.x;
//	//	y = keypoints2[it->trainIdx].pt.y;
//	//	cv::circle(image2, cv::Point(x, y), 3, cv::Scalar(255, 255, 255), 3);
//	//	points2.push_back(keypoints2[it->trainIdx].pt);
//	//}
//
//	//// Draw the epipolar lines
//	//std::vector<cv::Vec3f> lines1;
//	//cv::computeCorrespondEpilines(points1, 1, fundamental, lines1);
//
//	//for (std::vector<cv::Vec3f>::const_iterator it = lines1.begin();
//	//	it != lines1.end(); ++it) {
//
//	//	cv::line(image2, cv::Point(0, -(*it)[2] / (*it)[1]),
//	//		cv::Point(image2.cols, -((*it)[2] + (*it)[0] * image2.cols) / (*it)[1]),
//	//		cv::Scalar(255, 255, 255));
//	//}
//
//	//std::vector<cv::Vec3f> lines2;
//	//cv::computeCorrespondEpilines(points2, 2, fundamental, lines2);
//
//	//for (std::vector<cv::Vec3f>::const_iterator it = lines2.begin();
//	//	it != lines2.end(); ++it) {
//
//	//	cv::line(image1, cv::Point(0, -(*it)[2] / (*it)[1]),
//	//		cv::Point(image1.cols, -((*it)[2] + (*it)[0] * image1.cols) / (*it)[1]),
//	//		cv::Scalar(255, 255, 255));
//	//}
//
//	//// Display the images with epipolar lines
//	//cv::namedWindow("Right Image Epilines (RANSAC)");
//	//cv::imshow("Right Image Epilines (RANSAC)", image1);
//	//cv::namedWindow("Left Image Epilines (RANSAC)");
//	//cv::imshow("Left Image Epilines (RANSAC)", image2);
//
//
//
//	////3.计算两幅图像之间的单应矩阵----找到对应的点和拼接两幅图像
//	//// Read input images
// //   cv::Mat image1 = cv::imread("./images/parliament1.jpg", 0);
// //   cv::Mat image2 = cv::imread("./images/parliament2.jpg", 0);
// //   if (!image1.data || !image2.data)
// //         return 0;
//
// //   // Display the images
// //   cv::namedWindow("Image 1");
// //   cv::imshow("Image 1", image1);
// //   cv::namedWindow("Image 2");
// //   cv::imshow("Image 2", image2);
//
//	////构建关键点容器和描述子
//	//vector<KeyPoint> keypoints1;
//	//vector<KeyPoint> keypoints2;
//	//Mat descriptors1, descriptors2;
// //   //构建SIFT特征检测器
//	//Ptr<Feature2D> ptrFeature2D = xfeatures2d::SIFT::create(74);
//	//ptrFeature2D->detectAndCompute(image1, noArray() ,keypoints1, descriptors1);
//	//ptrFeature2D->detectAndCompute(image2, noArray(), keypoints2, descriptors2);
//
//	//cout << " Number of feature points (1):" << keypoints1.size() << endl;
//	//cout << " Number of feature points (2):" << keypoints2.size() << endl;
//
//	//BFMatcher matcher(NORM_L2, true);
//	//vector<DMatch> matches;
//	//matcher.match(descriptors1, descriptors2, matches);
//
//	//Mat imageMatches;
//	//drawMatches(image1, keypoints1, image2, keypoints2, matches, imageMatches, 
//	//	Scalar(255, 255, 255), Scalar(255, 255, 255), vector<char>(), 2);
//	//imshow("Matches (pure rotation case)", imageMatches);
//
//	////接下来使用findHomography函数实现，和findFundamentalMat函数相似
//	////我们要将关键点转变为Point2f
//	//vector<Point2f> points1, points2;
//	//for (vector<DMatch>::const_iterator it = matches.begin(); it != matches.end(); ++it)
//	//{
//	//	//获得左边关键点的位置
//	//	float x = keypoints1[it->queryIdx].pt.x;
//	//	float y = keypoints1[it->queryIdx].pt.y;
//	//	points1.push_back(Point2f(x, y));
//	//	//获得右边关键点的位置
//	//    x = keypoints2[it->trainIdx].pt.x;
//	//	y = keypoints2[it->trainIdx].pt.y;
//	//	points2.push_back(Point2f(x, y));
//	//}
//
//	//cout << points1.size() << " " << points2.size() << endl;
//	////找到第一幅图像和第二幅图像之间的单应矩阵
//	//vector<char> inliers;
//	//Mat homography = findHomography(
//	//    points1, points2,   //对应的点
//	//	inliers,            //输出的局部匹配项
//	//	RANSAC,             //RANSAC方法
//	//	1.);                //到重复投影点最大的距离
//	//						// Draw the inlier points
//	//cv::drawMatches(image1, keypoints1,  // 1st image and its keypoints
//	//	image2, keypoints2,  // 2nd image and its keypoints
//	//	matches,			// the matches
//	//	imageMatches,		// the image produced
//	//	cv::Scalar(255, 255, 255),  // color of the lines
//	//	cv::Scalar(255, 255, 255),  // color of the keypoints
//	//	inliers,
//	//	2);
//	//cv::namedWindow("Homography inlier points");
//	//cv::imshow("Homography inlier points", imageMatches);
//
//	////将第一幅图像扭曲到第二幅图像----实现两幅图像的拼接
//	//Mat result;
//	//warpPerspective(image1,     //输入图像
//	//	result,                 //输出图像
//	//	homography,             //单应矩阵
//	//	Size(2 * image1.cols, image1.rows)); //输出图像的尺寸
//
//	////把第一幅图像复制到完整图像的第一个半边
//	//Mat harf(result, Rect(0, 0, image2.cols, image2.rows));
//	//image2.copyTo(harf);  //把image2复制到image1的感兴趣区域
//
//	//imshow("Image mosaic", result);
//
//	////图像拼接技术----用Stitcher生成全景图
//	////Mat img1 = imread("./images/parliament1.jpg");
//	////Mat img2 = imread("./images/parliament2.jpg");
//	////imshow("img1", img1);
//	////imshow("img2", img2);
//	////vector<Mat> images;
//	////images.push_back(img1);
//	////images.push_back(img2);
//
//	//vector<Mat> images;
//	//images.push_back(imread("./images/parliament1_1.jpg"));
//	//images.push_back(imread("./images/parliament2_1.jpg"));
//
//	//Mat panorama;  //输出的全景图
//	////创建拼接器
//	//Stitcher stitcher = Stitcher::createDefault();
//	////拼接图像
//	//Stitcher::Status status = stitcher.stitch(images, panorama);
//
//	//if (status == cv::Stitcher::OK) // success?
//	//{
//	//
// //    	cv::namedWindow("Panorama");
//	//	cv::imshow("Panorama", panorama);
//	//}
//
//	
//
//	//4.检测图像中的平面目标
//	// Read input images
//    cv::Mat target = cv::imread("./images/cookbook1.bmp", 0);
//    cv::Mat image = cv::imread("./images/objects.jpg", 0);
//    if (!target.data || !image.data)
//      return 0;
//
//    // Display the images
//    cv::namedWindow("Target");
//    cv::imshow("Target", target);
//    cv::namedWindow("Image");
//    cv::imshow("Image", image);
//    
//	//初始化匹配器
//	TargetMatcher tmatcher(FastFeatureDetector::create(10), BRISK::create());
//	tmatcher.setNormType(NORM_HAMMING);
//	//定义输出数据
//	vector<DMatch> matches;
//	vector<KeyPoint> keypoints1, keypoints2;
//	vector<Point2f> corners;
//	//设定目标图像
//	tmatcher.setTarget(target);
//	//匹配目标图像
//	tmatcher.detectTarget(image, corners);
//	//画出目标角点
//	if (corners.size() == 4) { //已获得检测结果
//
//		cv::line(image, cv::Point(corners[0]), cv::Point(corners[1]), cv::Scalar(255, 255, 255), 6);
//		cv::line(image, cv::Point(corners[1]), cv::Point(corners[2]), cv::Scalar(255, 255, 255), 6);
//		cv::line(image, cv::Point(corners[2]), cv::Point(corners[3]), cv::Scalar(255, 255, 255), 6);
//		cv::line(image, cv::Point(corners[3]), cv::Point(corners[0]), cv::Scalar(255, 255, 255), 6);
//	}
//
//	cv::namedWindow("Target detection");
//	cv::imshow("Target detection", image);
//
//	waitKey(0);
//	return 0;
//}



#pragma once
#if !defined MATCHER
#define MATCHER

#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/xfeatures2d.hpp>

using namespace std;
using namespace cv;

#define NOCHECK      0
#define CROSSCHECK   1
#define RATIOCHECK   2
#define BOTHCHECK    3


class RobustMatcher
{
private:
	//特征点检测器对象的指针
	Ptr<FeatureDetector> detector;
	//特征描述子提取器对象的指针
	Ptr<DescriptorExtractor> descriptor;
	int normType;
	float ratio;   //第一个和第二个NN之间的最大比率
	bool refineF;  //如果等于true，则会优化基础矩阵
	bool refineM;  //如果等于true，则会优化匹配结果
	double distance;    //到极点的最小距离
	double confidence;  //可信度（概率）
public:

	RobustMatcher(const cv::Ptr<cv::FeatureDetector> &detector,
		const cv::Ptr<cv::DescriptorExtractor> &descriptor = cv::Ptr<cv::DescriptorExtractor>())
		: detector(detector), descriptor(descriptor), normType(cv::NORM_L2),
		ratio(0.8f), refineF(true), refineM(true), confidence(0.98), distance(1.0)
	{

		//这里使用关联描述子
		if (!this->descriptor) {
			this->descriptor = this->detector;
		}
    }


	// Set the feature detector
	void setFeatureDetector(const cv::Ptr<cv::FeatureDetector>& detect) {

		this->detector = detect;
	}

	// Set descriptor extractor
	void setDescriptorExtractor(const cv::Ptr<cv::DescriptorExtractor>& desc) {

		this->descriptor = desc;
	}

	// Set the norm to be used for matching
	void setNormType(int norm) {

		normType = norm;
	}

	// Set the minimum distance to epipolar in RANSAC
	void setMinDistanceToEpipolar(double d) {

		distance = d;
	}

	// Set confidence level in RANSAC
	void setConfidenceLevel(double c) {

		confidence = c;
	}

	// Set the NN ratio
	void setRatio(float r) {

		ratio = r;
	}

	// if you want the F matrix to be recalculated
	void refineFundamental(bool flag) {

		refineF = flag;
	}

	// if you want the matches to be refined using F
	void refineMatches(bool flag) {

		refineM = flag;
	}



	// Clear matches for which NN ratio is > than threshold
	// return the number of removed points 
	// (corresponding entries being cleared, i.e. size will be 0)
	int ratioTest(const std::vector<std::vector<cv::DMatch> >& inputMatches,
		std::vector<cv::DMatch>& outputMatches) {

		int removed = 0;

		// for all matches
		for (std::vector<std::vector<cv::DMatch> >::const_iterator matchIterator = inputMatches.begin();
			matchIterator != inputMatches.end(); ++matchIterator) {

			//   first best match/second best match
			if ((matchIterator->size() > 1) && // if 2 NN has been identified 
				(*matchIterator)[0].distance / (*matchIterator)[1].distance < ratio) {

				// it is an acceptable match
				outputMatches.push_back((*matchIterator)[0]);

			}
			else {

				removed++;
			}
		}

		return removed;
	}

	// Insert symmetrical matches in symMatches vector
	void symmetryTest(const std::vector<cv::DMatch>& matches1,
		const std::vector<cv::DMatch>& matches2,
		std::vector<cv::DMatch>& symMatches) {

		// for all matches image 1 -> image 2
		for (std::vector<cv::DMatch>::const_iterator matchIterator1 = matches1.begin();
			matchIterator1 != matches1.end(); ++matchIterator1) {

			// for all matches image 2 -> image 1
			for (std::vector<cv::DMatch>::const_iterator matchIterator2 = matches2.begin();
				matchIterator2 != matches2.end(); ++matchIterator2) {

				// Match symmetry test
				if (matchIterator1->queryIdx == matchIterator2->trainIdx  &&
					matchIterator2->queryIdx == matchIterator1->trainIdx) {

					// add symmetrical match
					symMatches.push_back(*matchIterator1);
					break; // next match in image 1 -> image 2
				}
			}
		}
	}

	// Apply both ratio and symmetry test
	// (often an over-kill)
	void ratioAndSymmetryTest(const std::vector<std::vector<cv::DMatch> >& matches1,
		const std::vector<std::vector<cv::DMatch> >& matches2,
		std::vector<cv::DMatch>& outputMatches) {

		// Remove matches for which NN ratio is > than threshold

		// clean image 1 -> image 2 matches
		std::vector<cv::DMatch> ratioMatches1;
		int removed = ratioTest(matches1, ratioMatches1);
		std::cout << "Number of matched points 1->2 (ratio test) : " << ratioMatches1.size() << std::endl;
		// clean image 2 -> image 1 matches
		std::vector<cv::DMatch> ratioMatches2;
		removed = ratioTest(matches2, ratioMatches2);
		std::cout << "Number of matched points 1->2 (ratio test) : " << ratioMatches2.size() << std::endl;

		// Remove non-symmetrical matches
		symmetryTest(ratioMatches1, ratioMatches2, outputMatches);

		std::cout << "Number of matched points (symmetry test): " << outputMatches.size() << std::endl;
	}

	// 用RANSAC算法获取优质匹配项
	// 返回基础矩阵和匹配项
	cv::Mat ransacTest(const std::vector<cv::DMatch>& matches,
		std::vector<cv::KeyPoint>& keypoints1,
		std::vector<cv::KeyPoint>& keypoints2,
		std::vector<cv::DMatch>& outMatches) {

		//把关键点转变为Point2f类型	
		std::vector<cv::Point2f> points1, points2;

		for (std::vector<cv::DMatch>::const_iterator it = matches.begin();
			it != matches.end(); ++it) {

			//获取左侧关键点的位置
			points1.push_back(keypoints1[it->queryIdx].pt);
			//获取右侧关键点的位置
			points2.push_back(keypoints2[it->trainIdx].pt);
		}

		// 用 RANSAC 计算F矩阵
		std::vector<uchar> inliers(points1.size(), 0);
		cv::Mat fundamental = cv::findFundamentalMat(
			points1, points2, //匹配像素点
			inliers,         //匹配状态(inlier or outlier)  
			cv::FM_RANSAC,   // RANSAC 算法
			distance,        // 到对极线的距离
			confidence);     // 置信度

		 //取下剩下的 (inliers) 匹配项
		std::vector<uchar>::const_iterator itIn = inliers.begin();
		std::vector<cv::DMatch>::const_iterator itM = matches.begin();
		// 遍历所有的匹配项
		for (; itIn != inliers.end(); ++itIn, ++itM) {

			if (*itIn) { // it is a valid match

				outMatches.push_back(*itM);
			}
		}

		if (refineF || refineM) {
			// The F matrix will be recomputed with all accepted matches

			// Convert keypoints into Point2f for final F computation	
			points1.clear();
			points2.clear();

			for (std::vector<cv::DMatch>::const_iterator it = outMatches.begin();
				it != outMatches.end(); ++it) {

				// Get the position of left keypoints
				points1.push_back(keypoints1[it->queryIdx].pt);
				// Get the position of right keypoints
				points2.push_back(keypoints2[it->trainIdx].pt);
			}

			// Compute 8-point F from all accepted matches
			fundamental = cv::findFundamentalMat(
				points1, points2, // matching points
				cv::FM_8POINT); // 8-point method

			if (refineM) {

				std::vector<cv::Point2f> newPoints1, newPoints2;
				// refine the matches
				correctMatches(fundamental,             // F matrix
					points1, points2,        // original position
					newPoints1, newPoints2); // new position
				for (int i = 0; i< points1.size(); i++) {

					std::cout << "(" << keypoints1[outMatches[i].queryIdx].pt.x
						<< "," << keypoints1[outMatches[i].queryIdx].pt.y
						<< ") -> ";
					std::cout << "(" << newPoints1[i].x
						<< "," << newPoints1[i].y << std::endl;
					std::cout << "(" << keypoints2[outMatches[i].trainIdx].pt.x
						<< "," << keypoints2[outMatches[i].trainIdx].pt.y
						<< ") -> ";
					std::cout << "(" << newPoints2[i].x
						<< "," << newPoints2[i].y << std::endl;

					keypoints1[outMatches[i].queryIdx].pt.x = newPoints1[i].x;
					keypoints1[outMatches[i].queryIdx].pt.y = newPoints1[i].y;
					keypoints2[outMatches[i].trainIdx].pt.x = newPoints2[i].x;
					keypoints2[outMatches[i].trainIdx].pt.y = newPoints2[i].y;
				}
			}
		}


		return fundamental;
	}








	//用RANSAC算法匹配特征点
	//返回基础矩阵和输出的匹配项
	//这是一个简单的展示，和书上的一样，下面有个比较复杂的
	Mat matchBook(Mat &image1, Mat &image2,  //输入图像
		vector<DMatch> &matches,         //输出匹配项
		vector<KeyPoint> &keypoints1,    //输出关键点
		vector<KeyPoint> &keypoints2)
	{
		//1.检测特征点
		detector->detect(image1, keypoints1);
		detector->detect(image2, keypoints2);
		//2.提取特征描述子
		Mat descriptors1, descriptors2;
		descriptor->compute(image1, keypoints1, descriptors1);
		descriptor->compute(image2, keypoints2, descriptors2);
		//3.匹配两幅图像描述子
		//（用于部分检测方法）
		//构造匹配类的实例（带交叉检查）
		BFMatcher matcher(normType,  //差距衡量
			true);                   //交叉检查标志
		//匹配描述子
		vector<DMatch> outputMatches;
		matcher.match(descriptors1, descriptors2, outputMatches);
		//4.用RANSAC算法验证匹配项
		Mat fundamental = ransacTest(outputMatches, keypoints1, keypoints2, matches);
		//返回基础矩阵
		return fundamental;
	}


	//用RANSAC算法匹配特征点
	//返回基础矩阵和输出的匹配项
	cv::Mat match(cv::Mat& image1, cv::Mat& image2, // 输入图像
		std::vector<cv::DMatch>& matches, // 输出匹配项和关键点
		std::vector<cv::KeyPoint>& keypoints1, std::vector<cv::KeyPoint>& keypoints2,
		int check = CROSSCHECK) {  // check type (symmetry or ratio or none or both)

		//检测特征点
		detector->detect(image1, keypoints1);
		detector->detect(image2, keypoints2);

		std::cout << "Number of feature points (1): " << keypoints1.size() << std::endl;
		std::cout << "Number of feature points (2): " << keypoints2.size() << std::endl;

		// 提取特征描述子
		cv::Mat descriptors1, descriptors2;
		descriptor->compute(image1, keypoints1, descriptors1);
		descriptor->compute(image2, keypoints2, descriptors2);

		std::cout << "descriptor matrix size: " << descriptors1.rows << " by " << descriptors1.cols << std::endl;

		// 3. Match the two image descriptors
		//    (optionaly apply some checking method)

		// Construction of the matcher with crosscheck 
		cv::BFMatcher matcher(normType,            //distance measure
			check == CROSSCHECK);  // crosscheck flag

								   // vectors of matches
		std::vector<std::vector<cv::DMatch> > matches1;
		std::vector<std::vector<cv::DMatch> > matches2;
		std::vector<cv::DMatch> outputMatches;

		// call knnMatch if ratio check is required
		if (check == RATIOCHECK || check == BOTHCHECK) {
			// from image 1 to image 2
			// based on k nearest neighbours (with k=2)
			matcher.knnMatch(descriptors1, descriptors2,
				matches1, // vector of matches (up to 2 per entry) 
				2);		  // return 2 nearest neighbours

			std::cout << "Number of matched points 1->2: " << matches1.size() << std::endl;

			if (check == BOTHCHECK) {
				// from image 2 to image 1
				// based on k nearest neighbours (with k=2)
				matcher.knnMatch(descriptors2, descriptors1,
					matches2, // vector of matches (up to 2 per entry) 
					2);		  // return 2 nearest neighbours

				std::cout << "Number of matched points 2->1: " << matches2.size() << std::endl;
			}

		}

		// select check method
		switch (check) {

		case CROSSCHECK:
			matcher.match(descriptors1, descriptors2, outputMatches);
			std::cout << "Number of matched points 1->2 (after cross-check): " << outputMatches.size() << std::endl;
			break;
		case RATIOCHECK:
			ratioTest(matches1, outputMatches);
			std::cout << "Number of matched points 1->2 (after ratio test): " << outputMatches.size() << std::endl;
			break;
		case BOTHCHECK:
			ratioAndSymmetryTest(matches1, matches2, outputMatches);
			std::cout << "Number of matched points 1->2 (after ratio and cross-check): " << outputMatches.size() << std::endl;
			break;
		case NOCHECK:
		default:
			matcher.match(descriptors1, descriptors2, outputMatches);
			std::cout << "Number of matched points 1->2: " << outputMatches.size() << std::endl;
			break;
		}

		// 4. Validate matches using RANSAC
		cv::Mat fundamental = ransacTest(outputMatches, keypoints1, keypoints2, matches);
		std::cout << "Number of matched points (after RANSAC): " << matches.size() << std::endl;

		// return the found fundamental matrix
		return fundamental;
	}
};

#endif // ！define MATCHER




#pragma once
#if !defined TMATCHER
#define TMATCHER

#define VERBOSE 1

#include <iostream>
#include <vector>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/features2d.hpp>

using namespace std;
using namespace cv;

class TargetMatcher
{
private:
	//特征点检测器对象的指针
	Ptr<FeatureDetector> detector;
	//特征描述子提取器对象的指针
	Ptr<DescriptorExtractor> descriptor;
	//目标图像
	Mat target;
	//比较描述子容器
	int normType;
	//最小重投影误差
	double distance;
	//金字塔形图像的数量
	int numberOfLevels;
	//层级之间的范围
	double scaleFactor;
	//目标图像构建的金字塔以及它的关键点
	vector<Mat> pyramid;
	vector<vector<KeyPoint> > pyrKeypoints;
	vector<Mat> pyrDescriptors;

	// 创建目标图像的金字塔
	void createPyramid() {

		// 创建目标图像的金字塔
		pyramid.clear();
		cv::Mat layer(target);
		for (int i = 0; i < numberOfLevels; i++) { // 逐层缩小
			pyramid.push_back(target.clone());
			resize(target, target, cv::Size(), scaleFactor, scaleFactor);
		}

		pyrKeypoints.clear();
		pyrDescriptors.clear();
		// 逐层检测关键点和描述子
		for (int i = 0; i < numberOfLevels; i++) {
			// 在第i层检测目标关键点
			pyrKeypoints.push_back(std::vector<cv::KeyPoint>());
			detector->detect(pyramid[i], pyrKeypoints[i]);
			if (VERBOSE)
				std::cout << "Interest points: target=" << pyrKeypoints[i].size() << std::endl;
			//在第i层计算描述子
			pyrDescriptors.push_back(cv::Mat());
			descriptor->compute(pyramid[i], pyrKeypoints[i], pyrDescriptors[i]);
		}
	}
public:
	TargetMatcher(const cv::Ptr<cv::FeatureDetector> &detector,
		const cv::Ptr<cv::DescriptorExtractor> &descriptor = cv::Ptr<cv::DescriptorExtractor>(),
		int numberOfLevels = 8, double scaleFactor = 0.9)
		: detector(detector), descriptor(descriptor), normType(cv::NORM_L2), distance(1.0),
		numberOfLevels(numberOfLevels), scaleFactor(scaleFactor) {

		// in this case use the associated descriptor
		if (!this->descriptor) {
			this->descriptor = this->detector;
		}
	}

	// Set the norm to be used for matching
	void setNormType(int norm) {

		normType = norm;
	}

	// Set the minimum reprojection distance
	void setReprojectionDistance(double d) {

		distance = d;
	}

	//设置目标图像
	void setTarget(const Mat t)
	{
		if (VERBOSE)
			cv::imshow("Target", t);
		target = t;
		createPyramid();
	}



	// Identify good matches using RANSAC
	// Return homography matrix and output matches
	//下面的是更好的
	cv::Mat ransacTest(const std::vector<cv::DMatch>& matches,
		std::vector<cv::KeyPoint>& keypoints1,
		std::vector<cv::KeyPoint>& keypoints2,
		std::vector<cv::DMatch>& outMatches) {

		// Convert keypoints into Point2f	
		std::vector<cv::Point2f> points1, points2;
		outMatches.clear();
		for (std::vector<cv::DMatch>::const_iterator it = matches.begin();
			it != matches.end(); ++it) {

			// Get the position of left keypoints
			points1.push_back(keypoints1[it->queryIdx].pt);
			// Get the position of right keypoints
			points2.push_back(keypoints2[it->trainIdx].pt);
		}

		// Find the homography between image 1 and image 2
		std::vector<uchar> inliers(points1.size(), 0);
		cv::Mat homography = cv::findHomography(
			points1, points2, // corresponding points
			inliers,         // match status (inlier or outlier)  
			cv::RHO,	     // RHO method
			distance);       // max distance to reprojection point

							 // extract the surviving (inliers) matches
		std::vector<uchar>::const_iterator itIn = inliers.begin();
		std::vector<cv::DMatch>::const_iterator itM = matches.begin();
		// for all matches
		for (; itIn != inliers.end(); ++itIn, ++itM) {

			if (*itIn) { // it is a valid match

				outMatches.push_back(*itM);
			}
		}

		return homography;
	}

	// 检测预先定义的平面目标
	// 返回单应矩阵和检测到的目标的4个角点
	cv::Mat detectTarget(const cv::Mat& image,
		// 目标角点的坐标（顺时针方向）
		std::vector<cv::Point2f>& detectedCorners) {

		// 1. 检测图像的关键点
		std::vector<cv::KeyPoint> keypoints;
		detector->detect(image, keypoints);
		if (VERBOSE)
			std::cout << "Interest points: image=" << keypoints.size() << std::endl;
		// 计算描述子
		cv::Mat descriptors;
		descriptor->compute(image, keypoints, descriptors);
		std::vector<cv::DMatch> matches;

		cv::Mat bestHomography;
		cv::Size bestSize;
		int maxInliers = 0;
		cv::Mat homography;

		// 构建匹配器
		cv::BFMatcher matcher(normType);

		// 2. 对金字塔的每层， 鲁棒匹配单应矩阵
		for (int i = 0; i < numberOfLevels; i++) {
			// 在目标和图像之间发现RANSAC单应矩阵
			matches.clear();

			// 匹配描述子
			matcher.match(pyrDescriptors[i], descriptors, matches);
			if (VERBOSE)
				std::cout << "Number of matches (level " << i << ")=" << matches.size() << std::endl;
			// 用RANSAC验证匹配项
			std::vector<cv::DMatch> inliers;
			homography = ransacTest(matches, pyrKeypoints[i], keypoints, inliers);
			if (VERBOSE)
				std::cout << "Number of inliers=" << inliers.size() << std::endl;

			if (inliers.size() > maxInliers) { // 有更好的 H
				maxInliers = inliers.size();
				bestHomography = homography;
				bestSize = pyramid[i].size();
			}

			if (VERBOSE) {
				cv::Mat imageMatches;
				cv::drawMatches(target, pyrKeypoints[i],  // 1st image and its keypoints
					image, keypoints,  // 2nd image and its keypoints
					inliers,			// the matches
					imageMatches,		// the image produced
					cv::Scalar(255, 255, 255),  // color of the lines
					cv::Scalar(255, 255, 255),  // color of the keypoints
					std::vector<char>(),
					2);
				cv::imshow("Target matches", imageMatches);
				cv::waitKey();
			}
		}

		// 3. 用最佳单应矩阵找出角点坐标
		if (maxInliers > 8) { // 估计值有效

			//最佳尺寸的目标角点
			std::vector<cv::Point2f> corners;
			corners.push_back(cv::Point2f(0, 0));
			corners.push_back(cv::Point2f(bestSize.width - 1, 0));
			corners.push_back(cv::Point2f(bestSize.width - 1, bestSize.height - 1));
			corners.push_back(cv::Point2f(0, bestSize.height - 1));

			// 重新投影目标角点
			cv::perspectiveTransform(corners, detectedCorners, bestHomography);
		}

		if (VERBOSE)
			std::cout << "Best number of inliers=" << maxInliers << std::endl;
		return bestHomography;
	}

};

#endif // !defined TMATCHER

```

### Chapter_11  :   三维重建

> 1.相机标定
> 2.相机姿态还原
> 3.用标定相机实现三维重建
> 4.计算立体图像的深度

<!--more-->

```c++

//#include <iostream>
//#include <iomanip>
//#include <vector>
//
//#include <opencv2/core.hpp>
//#include <opencv2/imgproc.hpp>
//#include <opencv2/highgui.hpp>
//#include <opencv2/features2d.hpp>
//
//#include <opencv2/viz.hpp>
//#include <opencv2/calib3d.hpp>
//
//#include "CameraCalibrator.h"
//
//int main()
//{
//	////1.相机标定
//	//cv::Mat image;
//	//std::vector<std::string> filelist;
//
//	//// generate list of chessboard image filename
//	//// named chessboard01 to chessboard27 in chessboard sub-dir
//	//for (int i = 1; i <= 27; i++) {
//
//	//	std::stringstream str;
//	//	str << "images/chessboards/chessboard" << std::setw(2) << std::setfill('0') << i << ".jpg";
//	//	std::cout << str.str() << std::endl;
//
//	//	filelist.push_back(str.str());
//	//	image = cv::imread(str.str(), 0);
//
//	//	// cv::imshow("Board Image",image);	
//	//	// cv::waitKey(100);
//	//}
//
//	//// Create calibrator object
//	//CameraCalibrator cameraCalibrator;
//	//// add the corners from the chessboard
//	//cv::Size boardSize(7, 5);
//	//cameraCalibrator.addChessboardPoints(
//	//	filelist,	// filenames of chessboard image
//	//	boardSize, "Detected points");	// size of chessboard
//
//	//									// calibrate the camera
//	//cameraCalibrator.setCalibrationFlag(true, true);
//	//cameraCalibrator.calibrate(image.size());
//
//	//// Exampple of Image Undistortion
//	//image = cv::imread(filelist[14], 0);
//	//cv::Size newSize(static_cast<int>(image.cols*1.5), static_cast<int>(image.rows*1.5));
//	//cv::Mat uImage = cameraCalibrator.remap(image, newSize);
//
//	//// display camera matrix
//	//cv::Mat cameraMatrix = cameraCalibrator.getCameraMatrix();
//	//std::cout << " Camera intrinsic: " << cameraMatrix.rows << "x" << cameraMatrix.cols << std::endl;
//	//std::cout << cameraMatrix.at<double>(0, 0) << " " << cameraMatrix.at<double>(0, 1) << " " << cameraMatrix.at<double>(0, 2) << std::endl;
//	//std::cout << cameraMatrix.at<double>(1, 0) << " " << cameraMatrix.at<double>(1, 1) << " " << cameraMatrix.at<double>(1, 2) << std::endl;
//	//std::cout << cameraMatrix.at<double>(2, 0) << " " << cameraMatrix.at<double>(2, 1) << " " << cameraMatrix.at<double>(2, 2) << std::endl;
//
//	//cv::namedWindow("Original Image");
//	//cv::imshow("Original Image", image);
//	//cv::namedWindow("Undistorted Image");
//	//cv::imshow("Undistorted Image", uImage);
//
//	//// Store everything in a xml file
//	//cv::FileStorage fs("calib.xml", cv::FileStorage::WRITE);
//	//fs << "Intrinsic" << cameraMatrix;
//	//fs << "Distortion" << cameraCalibrator.getDistCoeffs();
//
//
//
//
//	// Read the camera calibration parameters
//	cv::Mat cameraMatrix;
//	cv::Mat cameraDistCoeffs;
//	cv::FileStorage fs("calib.xml", cv::FileStorage::READ);
//	fs["Intrinsic"] >> cameraMatrix;
//	fs["Distortion"] >> cameraDistCoeffs;
//	std::cout << " Camera intrinsic: " << cameraMatrix.rows << "x" << cameraMatrix.cols << std::endl;
//	std::cout << cameraMatrix.at<double>(0, 0) << " " << cameraMatrix.at<double>(0, 1) << " " << cameraMatrix.at<double>(0, 2) << std::endl;
//	std::cout << cameraMatrix.at<double>(1, 0) << " " << cameraMatrix.at<double>(1, 1) << " " << cameraMatrix.at<double>(1, 2) << std::endl;
//	std::cout << cameraMatrix.at<double>(2, 0) << " " << cameraMatrix.at<double>(2, 1) << " " << cameraMatrix.at<double>(2, 2) << std::endl << std::endl;
//	cv::Matx33d cMatrix(cameraMatrix);
//
//	// Input image points
//	std::vector<cv::Point2f> imagePoints;
//	imagePoints.push_back(cv::Point2f(136, 113));
//	imagePoints.push_back(cv::Point2f(379, 114));
//	imagePoints.push_back(cv::Point2f(379, 150));
//	imagePoints.push_back(cv::Point2f(138, 135));
//	imagePoints.push_back(cv::Point2f(143, 146));
//	imagePoints.push_back(cv::Point2f(381, 166));
//	imagePoints.push_back(cv::Point2f(345, 194));
//	imagePoints.push_back(cv::Point2f(103, 161));
//
//	// Input object points
//	std::vector<cv::Point3f> objectPoints;
//	objectPoints.push_back(cv::Point3f(0, 45, 0));
//	objectPoints.push_back(cv::Point3f(242.5, 45, 0));
//	objectPoints.push_back(cv::Point3f(242.5, 21, 0));
//	objectPoints.push_back(cv::Point3f(0, 21, 0));
//	objectPoints.push_back(cv::Point3f(0, 9, -9));
//	objectPoints.push_back(cv::Point3f(242.5, 9, -9));
//	objectPoints.push_back(cv::Point3f(242.5, 9, 44.5));
//	objectPoints.push_back(cv::Point3f(0, 9, 44.5));
//
//	// Read image
//	cv::Mat image = cv::imread("./images/bench2.jpg");
//	// Draw image points
//	for (int i = 0; i < 8; i++) {
//		cv::circle(image, imagePoints[i], 3, cv::Scalar(0, 0, 0), 2);
//	}
//	cv::imshow("An image of a bench", image);
//
//	// Create a viz window
//	cv::viz::Viz3d visualizer("Viz window");
//	visualizer.setBackgroundColor(cv::viz::Color::white());
//
//	/// Construct the scene
//	// Create a virtual camera
//	cv::viz::WCameraPosition cam(cMatrix,  // matrix of intrinsics
//		image,    // image displayed on the plane
//		30.0,     // scale factor
//		cv::viz::Color::black());
//	// Create a virtual bench from cuboids
//	cv::viz::WCube plane1(cv::Point3f(0.0, 45.0, 0.0),
//		cv::Point3f(242.5, 21.0, -9.0),
//		true,  // show wire frame 
//		cv::viz::Color::blue());
//	plane1.setRenderingProperty(cv::viz::LINE_WIDTH, 4.0);
//	cv::viz::WCube plane2(cv::Point3f(0.0, 9.0, -9.0),
//		cv::Point3f(242.5, 0.0, 44.5),
//		true,  // show wire frame 
//		cv::viz::Color::blue());
//	plane2.setRenderingProperty(cv::viz::LINE_WIDTH, 4.0);
//	// Add the virtual objects to the environment
//	visualizer.showWidget("top", plane1);
//	visualizer.showWidget("bottom", plane2);
//	visualizer.showWidget("Camera", cam);
//
//	// Get the camera pose from 3D/2D points
//	cv::Mat rvec, tvec;
//	cv::solvePnP(objectPoints, imagePoints,      // corresponding 3D/2D pts 
//		cameraMatrix, cameraDistCoeffs, // calibration 
//		rvec, tvec);                    // output pose
//	std::cout << " rvec: " << rvec.rows << "x" << rvec.cols << std::endl;
//	std::cout << " tvec: " << tvec.rows << "x" << tvec.cols << std::endl;
//
//	cv::Mat rotation;
//	// convert vector-3 rotation
//	// to a 3x3 rotation matrix
//	cv::Rodrigues(rvec, rotation);
//
//	// Move the bench	
//	cv::Affine3d pose(rotation, tvec);
//	visualizer.setWidgetPose("top", pose);
//	visualizer.setWidgetPose("bottom", pose);
//
//	// visualization loop
//	while (cv::waitKey(100) == -1 && !visualizer.wasStopped())
//	{
//
//		visualizer.spinOnce(1,     // pause 1ms 
//			true); // redraw
//	}
//
//
//
//
//
//
//
//
//	waitKey(0);
//	return 0;
//}






#pragma once
#if !defined CAMERACALIBRATOR_H
#define CAMERACALIBRATOR_H

#include <iostream>
#include <vector>

#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

class CameraCalibrator
{
private:
	//输入点：
	//世界坐标系中的点
	//（每个正方形为一个单位）
	vector<vector<Point3f> > objectPoints;
	//点在图像中的位置（以像素为单位）
	vector<vector<Point2f> > imagePoints;
	//输出矩阵
	Mat cameraMatrix;
	Mat distCoeffs;
	//指定标定方式的标志
	int flag;
	//用于图像不失真
	Mat map1, map2;
	bool mustInitUndistort;

public:
	CameraCalibrator() : flag(0), mustInitUndistort(true){}
	
	//打开棋盘图像，提取角点
	int addChessboardPoints(const vector<string> &filelist, Size &boardSize, string windowName = "");
	// Add scene points and corresponding image points
	void addPoints(const std::vector<cv::Point2f>& imageCorners, const std::vector<cv::Point3f>& objectCorners);
	//标定相机
	double calibrate(const Size imageSize);
	// Set the calibration flag
	void setCalibrationFlag(bool radial8CoeffEnabled = false, bool tangentialParamEnabled = false);
	//去除图像中的畸变（标定后）
	Mat remap(const Mat &image, Size &outputSize = Size(-1, -1));

	// Getters
	cv::Mat getCameraMatrix() { return cameraMatrix; }
	cv::Mat getDistCoeffs() { return distCoeffs; }

};


#endif // !defined CAMERACALIBRATOR_H





#include "CameraCalibrator.h"

//打开棋盘图像，提取角点
int CameraCalibrator::addChessboardPoints(
	const vector<string> &filelist,   //文件名列表
	Size &boardSize,                  //标定面板的大小
	string windowName                 //展示窗口的名字
)
{
	//棋盘上的角点
	vector<Point2f> imageCorners;
	vector<Point3f> objectCorners;
	//场景中的三维点
	//在棋盘坐标系中， 初始化棋盘中的角点
	//角点的三维坐标（X, Y，Z）= （i，j，0）
	for (int i = 0; i < boardSize.height; i++)
	{
		for (int j = 0; j < boardSize.width; j++)
		{
			objectCorners.push_back(Point3f(i, j, 0.0f));
		}
	}
	//图像上的二维点
	Mat image;  //用于存储棋盘图像
	int successes = 0;
	//处理所有视角
	for (int i = 0; i < filelist.size(); i++)
	{
		//打开图像
		image = imread(filelist[i], 0);
		//取得棋盘中的角点
		bool found = findChessboardCorners(image,       //包含棋盘图案的图像
			                               boardSize,   //图案的大小
			                               imageCorners //检测到角点的列表
		                                   );
		//取得角点上的亚像素级精度
		if (found)
		{
			cornerSubPix(image, imageCorners,
				Size(5, 5),
				Size(-1, -1),
				TermCriteria(TermCriteria::MAX_ITER + TermCriteria::EPS,
					30,   //最大迭代次数
					0.1));//最小精度
			//如果棋盘是完好的，就把它加入结果
			if (imageCorners.size() == boardSize.area())
			{
				//加入从同一个视角得到的图像和场景点
				addPoints(imageCorners, objectCorners);
				successes++;
			}
		}
		if (windowName.length() > 0 && imageCorners.size() == boardSize.area())
		{
			//画角点
			drawChessboardCorners(image, boardSize, imageCorners, found);
			imshow(windowName, image);
			waitKey(100);
		}
	}

	return successes;
}


// Add scene points and corresponding image points
void CameraCalibrator::addPoints(const std::vector<cv::Point2f>& imageCorners, const std::vector<cv::Point3f>& objectCorners) {

	// 2D image points from one view
	imagePoints.push_back(imageCorners);
	// corresponding 3D scene points
	objectPoints.push_back(objectCorners);
}

// 标定相机
// 返回重投影误差
double CameraCalibrator::calibrate(const cv::Size imageSize)
{
	// undistorter must be reinitialized
	mustInitUndistort = true;

	//输出旋转量和平移量
	std::vector<cv::Mat> rvecs, tvecs;

	// 开始标定
	return
		calibrateCamera(objectPoints, // 三维点
			imagePoints,  // 图像点
			imageSize,    // 图像尺寸
			cameraMatrix, // 输出相机矩阵
			distCoeffs,   // 输出畸变矩阵
			rvecs, tvecs, // Rs, Ts 
			flag);        // 设置选项,CV_CALIB_USE_INTRINSIC_GUESS);

}

// 去除图像中的畸变(标定后)
cv::Mat CameraCalibrator::remap(const cv::Mat &image, cv::Size &outputSize) {

	cv::Mat undistorted;

	if (outputSize.height == -1)
		outputSize = image.size();

	if (mustInitUndistort) { // 每个标定过程调用一次

		cv::initUndistortRectifyMap(
			cameraMatrix,  // 计算得到的相机矩阵
			distCoeffs,    // 计算得到的畸变矩阵
			cv::Mat(),     // 可选矫正项 (无) 
			cv::Mat(),     // 生成无畸变的相机矩阵
			outputSize,    // 无畸变图像的尺寸
			CV_32FC1,      // 输出图片的类型
			map1, map2);   // x 和 y 映射功能

		mustInitUndistort = false;
	}

	// 映射功能
	cv::remap(image, undistorted, map1, map2,
		cv::INTER_LINEAR); // 插值类型

	return undistorted;
}


// Set the calibration options
// 8radialCoeffEnabled should be true if 8 radial coefficients are required (5 is default)
// tangentialParamEnabled should be true if tangeantial distortion is present
void CameraCalibrator::setCalibrationFlag(bool radial8CoeffEnabled, bool tangentialParamEnabled) {

	// Set the flag used in cv::calibrateCamera()
	flag = 0;
	if (!tangentialParamEnabled) flag += CV_CALIB_ZERO_TANGENT_DIST;
	if (radial8CoeffEnabled) flag += CV_CALIB_RATIONAL_MODEL;
}


```



### Chapter_12 : 处理视频序列

> 1.读取视频处理
> 2.处理视频帧
> 3.提取视频中的眼前物体

<!--more-->

```c++
#include <opencv2/bgsegm.hpp>

#include "VideoProcessor.h"
#include "BGFGSegmentor.h"

void draw(const cv::Mat& img, cv::Mat& out) {

	img.copyTo(out);
	cv::circle(out, cv::Point(100, 100), 5, cv::Scalar(255, 0, 0), 2);
}

void canny(cv::Mat& img, cv::Mat& out) {

	// 转换为灰度
	if (img.channels() == 3)
		cv::cvtColor(img, out, cv::COLOR_BGR2GRAY);
	// 计算Canny边缘
	cv::Canny(out, out, 100, 200);
	// 反转图像
	cv::threshold(out, out, 128, 255, cv::THRESH_BINARY_INV);
}


int main()
{
	////1.读取视频处理
 //   //打开视频文件
	//VideoCapture capture("./images/bike.avi");
	////检查视频是否打开成功
	//if (!capture.isOpened() )
	//	return 1;
	////取得帧速率
	//double rate = capture.get(CV_CAP_PROP_FPS);

	//bool stop(false);
	//Mat frame;  //当前视频帧
	//namedWindow("Extracted Frame");

	////根据帧速率计算帧之间的等待时间， 单位为ms
	//int delay = 1000 / rate;
	////循环遍历视频中的全部帧
	//while (!stop)
	//{
	//	//读取下一帧（如果有）
	//	if (!capture.read(frame) )
	//		break;

	//	imshow("Extracted Frame", frame);

	//	//等待一段时间，或者通过按键停止
	//	if (waitKey(delay) >= 0)
	//		stop = true;
	//	
	//}

	////关闭视频文件
	////不是必须的，因为类的析构函数会调用
	//capture.release();


	////2.处理视频帧
	////打开视频文件
	//VideoCapture capture("./images/bike.avi");
	////检查视频是否打开
	//if (!capture.isOpened())
	//	return 1;
	////获取帧速率
	//double rate = capture.get(CV_CAP_PROP_FPS);
	//cout << "Frame rate: " << rate << "fps" << endl;

	//double stop(false);
	//Mat frame;
	////根据帧速率计算镇之间的等待时间，单位为ms
	//int delay = 1000 / rate;
	//long long i = 0;
	//string b = "bike";
	//string ext = ".bmp";



	////循环遍历视频中的全部帧
	//while (!stop)
	//{
	//	//读取下一帧（如果有）
	//	if (!capture.read(frame))
	//		break;
	//	imshow("Extracted Frame", frame);

	//	string name(b);
	//	ostringstream ss;
	//	ss << setfill('0') << setw(3) << i;
	//	name += ss.str();
	//	i++;
	//	name += ext;

	//	cout << name << endl;

	//	Mat test;

	//	//等待一段时间， 或者通过按键停止
	//	if (waitKey(delay) >= 0)
	//		stop = true;
	//}

	//capture.release();


	//waitKey();


	////现在创建一个自定义类VideoProcessor完整的封装了视频处理任务
	//VideoProcessor processor;
	////打开视频文件
	//processor.setInput("./images/bike.avi");
	////声明显示视频的窗口
	//processor.displayInput("Input Video");
	//processor.displayOutput("Output Video");
	////用原始帧速率播放视频
	//processor.setDelay(1000. / processor.getFrameRate());
	////设置处理帧的回调函数
	//processor.setFrameProcessor(canny);
	////输出视频
	//processor.setOutput("./images/bikeCanny.avi", -1, 15);
	////在当前帧停止处理
	//processor.stopAtFrameNo(51);
	////开始处理
	//processor.run();

	//waitKey();


	//3.提取视频中的眼前物体
	// 打开视频
	cv::VideoCapture capture("./images/bike.avi");
	// 检查是否打开成功
	if (!capture.isOpened())
		return 0;

	// 当前视频帧
	cv::Mat frame;
	// 前景的二值图像
	cv::Mat foreground;
	// 背景图
	cv::Mat background;

	cv::namedWindow("Extracted Foreground");

	// 混合高斯模型类的对象，全部采用默认参数
	cv::Ptr<cv::BackgroundSubtractor> ptrMOG = cv::bgsegm::createBackgroundSubtractorMOG();

	bool stop(false);
	// 遍历视频中的所有帧
	while (!stop)
	{

		// 读取下一帧（如果有）
		if (!capture.read(frame))
			break;

		// 更新背景并返回前景
		ptrMOG->apply(frame, foreground, 0.01);

		// 改进图像效果
		cv::threshold(foreground, foreground, 128, 255, cv::THRESH_BINARY_INV);

		// 显示前景
		cv::imshow("Extracted Foreground", foreground);

		// 产生延时或者按键结束
		if (cv::waitKey(10) >= 0)
			stop = true;
	}
	cv::waitKey();

	//// Create video procesor instance
	//VideoProcessor processor;

	//// Create background/foreground segmentor 
	//BGFGSegmentor segmentor;
	//segmentor.setThreshold(25);

	//// Open video file
	//processor.setInput("./images/bike.avi");

	//// set frame processor
	//processor.setFrameProcessor(&segmentor);

	//// Declare a window to display the video
	//processor.displayOutput("Extracted Foreground");

	//// Play the video at the original frame rate
	//processor.setDelay(1000. / processor.getFrameRate());

	//// Start the process
	//processor.run();

	//cv::waitKey();


	return 0;
}






#pragma once
#if !defined VPROCESSOR
#define VPROCESSOR

#include <iostream>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

//帧处理的接口
class FrameProcessor
{
public:
	//处理方法
	virtual void process(Mat &input, Mat &output) = 0;
};

class VideoProcessor
{
private:
	//Opencv视频捕获对象
	VideoCapture capture;
	//处理每一帧时都会调用的回调函数
	void(*process) (Mat&, Mat&);
	FrameProcessor *frameProcessor;
	//布尔型变量，表示该回调函数是否会被调用
	bool callIt;
	//输入窗口的显示名称
	string windowNameInput;
	//输出窗口的显示名称
	string windowNameOutput;
	//帧之间的延时
	int delay;
	//已经处理的帧数
	long fnumber;
	//达到这个帧数时结束
	long frameToStop;
	//结束处理
	bool stop;

	//用作输入图像的图像文件名向量
	vector<string> images;
	//图像容器的迭代器
	vector<string>::const_iterator itImg;
	//Opencv写视频对象
	VideoWriter writer;
	//输出文件
	string outputFile;
	//输出图像的当前序号
	int currentIndex;
	//输出图像文件名中数字的位数
	int digits;
	//输出图像的扩展名
	string extension;

	//取得下一帧
	//可以是视频文件、摄像机、图像向量
	bool readNextFrame(Mat &frame)
	{
		if (images.size()== 0)
			return capture.read(frame);
		else
		{
			if (itImg != images.end() )
			{
				frame = imread(*itImg);
				itImg++;
				return frame.data != 0;
			}

			return false;
		}
	}
		//写输出的帧
		//可以是视频文件或者图像组
		void writeNextFrame(Mat &frame)
		{
			if (extension.length())  //写入到图像组
			{
				stringstream ss;
				ss << outputFile << setfill('0') << setw(digits) << currentIndex++ << extension;
				imwrite(ss.str(), frame);
			}
			else   //写入到视频文件
			{
				writer.write(frame);
			}
		}
	
public:

	// 构造函数设置默认的值
	VideoProcessor() : callIt(false), delay(-1),fnumber(0), stop(false), digits(0), frameToStop(-1),process(0), frameProcessor(0) {}



	// 设置视频文件的名称
	bool setInput(std::string filename) {

		fnumber = 0;
		// 防止已经有资源和VideoCapture实例关联
		capture.release();
		images.clear();

		//打开视频文件
		return capture.open(filename);
	}

	// 设置相机id
	bool setInput(int id) {

		fnumber = 0;
		// 防止已经有资源和VideoCapture实例关联
		capture.release();
		images.clear();

		// 打开视频文件
		return capture.open(id);
	}

	// 设置输入图像的向量
	bool setInput(const std::vector<std::string>& imgs) {

		fnumber = 0;
		// 防止已经有资源和VideoCapture实例关联
		capture.release();

		// 将这个图像向量作为输入对象
		images = imgs;
		itImg = images.begin();

		return true;
	}

	// 设置输出视频文件
	// 默认情况下会使用与输入视频相同的参数
	bool setOutput(const std::string &filename, int codec = 0, double framerate = 0.0, bool isColor = true) {

		outputFile = filename;
		extension.clear();

		if (framerate == 0.0)
			framerate = getFrameRate(); // 与输入相同

		char c[4];
		//使用与输入相同的编解码器
		if (codec == 0) {
			codec = getCodec(c);
		}

		//打开输出视频
		return writer.open(outputFile, //文件名
			codec, // 所用的编解码器
			framerate,      // 视频的帧速率
			getFrameSize(), // 帧的尺寸
			isColor);       // 彩色视频？
	}

	// 设置输出一系列图像文件
	// 扩展名是 ".jpg",或者".bmp" ...
	bool setOutput(const std::string &filename, // 前缀
		const std::string &ext, // 图像文件的扩展名
		int numberOfDigits = 3,   //数字的位数
		int startIndex = 0) {     // 开始序号

	    // 数字的位数必须是正数
		if (numberOfDigits<0)
			return false;

		// 文件名和常用的扩展名
		outputFile = filename;
		extension = ext;

		// 文件编号方案中数字的位数
		digits = numberOfDigits;
		// 从这个序号开始编码
		currentIndex = startIndex;

		return true;
	}

	// set the callback function that will be called for each frame
	void setFrameProcessor(void(*frameProcessingCallback)(cv::Mat&, cv::Mat&)) {

		// 使回调函数失效
		frameProcessor = 0;
		// 这个就是即将被调用的帧处理实例
		process = frameProcessingCallback;
		callProcess();
	}

	// 设置实现 FrameProcessor接口的实例
	void setFrameProcessor(FrameProcessor* frameProcessorPtr) {

		// 使回调函数失效
		process = 0;
		// 这个就是即将被调用的帧处理实例
		frameProcessor = frameProcessorPtr;
		callProcess();
	}

	// stop streaming at this frame number
	void stopAtFrameNo(long frame) {

		frameToStop = frame;
	}

	//需要调用回调函数
	void callProcess() {

		callIt = true;
	}

	//不需要调用回调函数
	void dontCallProcess() {

		callIt = false;
	}

	//用于显示输入的帧
	void displayInput(std::string wn) {

		windowNameInput = wn;
		cv::namedWindow(windowNameInput);
	}

	//用于显示处理过的帧
	void displayOutput(std::string wn) {

		windowNameOutput = wn;
		cv::namedWindow(windowNameOutput);
	}

	// do not display the processed frames
	void dontDisplay() {

		cv::destroyWindow(windowNameInput);
		cv::destroyWindow(windowNameOutput);
		windowNameInput.clear();
		windowNameOutput.clear();
	}

	// 设置帧之间的延迟
	// 0 表示每一帧都在等待
	// 负数表示不延时
	void setDelay(int d) {

		delay = d;
	}

	// a count is kept of the processed frames
	long getNumberOfProcessedFrames() {

		return fnumber;
	}

	// return the size of the video frame
	cv::Size getFrameSize() {

		if (images.size() == 0) {

			// get size of from the capture device
			int w = static_cast<int>(capture.get(cv::CAP_PROP_FRAME_WIDTH));
			int h = static_cast<int>(capture.get(cv::CAP_PROP_FRAME_HEIGHT));

			return cv::Size(w, h);

		}
		else { // if input is vector of images

			cv::Mat tmp = cv::imread(images[0]);
			if (!tmp.data) return cv::Size(0, 0);
			else return tmp.size();
		}
	}

	// 返回下一帧的编号
	long getFrameNumber() {

		if (images.size() == 0) {

			//从捕获设备获取信息
			long f = static_cast<long>(capture.get(cv::CAP_PROP_POS_FRAMES));
			return f;

		}
		else { // if input is vector of images

			return static_cast<long>(itImg - images.begin());
		}
	}

	// return the position in ms
	double getPositionMS() {

		// undefined for vector of images
		if (images.size() != 0) return 0.0;

		double t = capture.get(cv::CAP_PROP_POS_MSEC);
		return t;
	}

	// return the frame rate
	double getFrameRate() {

		// undefined for vector of images
		if (images.size() != 0) return 0;

		double r = capture.get(cv::CAP_PROP_FPS);
		return r;
	}

	// return the number of frames in video
	long getTotalFrameCount() {

		// for vector of images
		if (images.size() != 0) return images.size();

		long t = capture.get(cv::CAP_PROP_FRAME_COUNT);
		return t;
	}

	// get the codec of input video
	int getCodec(char codec[4]) {

		// undefined for vector of images
		if (images.size() != 0) return -1;

		union {
			int value;
			char code[4];
		} returned;

		returned.value = static_cast<int>(capture.get(cv::CAP_PROP_FOURCC));

		codec[0] = returned.code[0];
		codec[1] = returned.code[1];
		codec[2] = returned.code[2];
		codec[3] = returned.code[3];

		return returned.value;
	}

	// go to this frame number
	bool setFrameNumber(long pos) {

		// for vector of images
		if (images.size() != 0) {

			// move to position in vector
			itImg = images.begin() + pos;
			// is it a valid position?
			if (pos < images.size())
				return true;
			else
				return false;

		}
		else { // if input is a capture device

			return capture.set(cv::CAP_PROP_POS_FRAMES, pos);
		}
	}

	// go to this position
	bool setPositionMS(double pos) {

		// not defined in vector of images
		if (images.size() != 0)
			return false;
		else
			return capture.set(cv::CAP_PROP_POS_MSEC, pos);
	}

	// go to this position expressed in fraction of total film length
	bool setRelativePosition(double pos) {

		// for vector of images
		if (images.size() != 0) {

			// move to position in vector
			long posI = static_cast<long>(pos*images.size() + 0.5);
			itImg = images.begin() + posI;
			// is it a valid position?
			if (posI < images.size())
				return true;
			else
				return false;

		}
		else { // if input is a capture device

			return capture.set(cv::CAP_PROP_POS_AVI_RATIO, pos);
		}
	}

	//结束处理
	void stopIt() {

		stop = true;
	}

	// 处理过程是否已经停止？
	bool isStopped() {

		return stop;
	}

	// 捕获设备是否已经打开？
	bool isOpened() {

		return capture.isOpened() || !images.empty();
	}

	// 抓取并处理序列中的帧
	void run() {

		//当前帧
		cv::Mat frame;
		//输出帧
		cv::Mat output;

		//如果没有设置捕获设备
		if (!isOpened())
			return;

		stop = false;

		while (!isStopped()) {

			// 读取下一帧（如果有）
			if (!readNextFrame(frame))
				break;

			// 显示输入的帧
			if (windowNameInput.length() != 0)
				cv::imshow(windowNameInput, frame);

			// 调用处理函数
			if (callIt) {

				// 处理帧
				if (process)
					process(frame, output);
				else if (frameProcessor)
					frameProcessor->process(frame, output);
				// 递增帧数
				fnumber++;

			}
			else {
				//没有处理
				output = frame;
			}

			// 写入输出序列
			if (outputFile.length() != 0)
				writeNextFrame(output);

			// 显示输出的帧
			if (windowNameOutput.length() != 0)
				cv::imshow(windowNameOutput, output);

			// 产生延迟
			if (delay >= 0 && cv::waitKey(delay) >= 0)
				stopIt();

			// 检查是否需要结束
			if (frameToStop >= 0 && getFrameNumber() == frameToStop)
				stopIt();
		}
	}
};



#endif // !defined VPROCESSOR






#pragma once
#if !defined BGFGSeg
#define BGFGSeg

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "videoprocessor.h"

class BGFGSegmentor : public FrameProcessor {

	cv::Mat gray;			// 当前灰度图像
	cv::Mat background;		// 累积的背景
	cv::Mat backImage;		// 当前背景图像
	cv::Mat foreground;		// 前景图像
	double learningRate;    // 累积背景时使用的学习效率
	int threshold;			// 提取前景的阈值

public:

	BGFGSegmentor() : threshold(10), learningRate(0.01) {}

	// Set the threshold used to declare a foreground
	void setThreshold(int t) {

		threshold = t;
	}

	// Set the learning rate
	void setLearningRate(double r) {

		learningRate = r;
	}

	// 处理方法
	void process(cv::Mat &frame, cv::Mat &output) {

		// 转换为灰度图
		cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);

		// 采用第一针初始化背景
		if (background.empty())
			gray.convertTo(background, CV_32F);

		// 将背景转化为 8U类型
		background.convertTo(backImage, CV_8U);

		// 计算图像与背景之间的差异
		cv::absdiff(backImage, gray, foreground);

		// 在前景图像上应用
		cv::threshold(foreground, output, threshold, 255, cv::THRESH_BINARY_INV);

		// 积累背景
		cv::accumulateWeighted(gray, background,
			// alpha*gray + (1-alpha)*background
			learningRate,  // 学习速率
			output);       // 掩码
	}
};

#endif


```

### Chapter_13  :   跟踪运动物体

> 1.跟踪视频中的特征点
> 2.估算光流
> 3.跟踪视频中的物体

<!--more-->

```c++
#include "FeatureTracker.h"
#include "VideoProcessor.h"
#include "VisualTracker.h"

//绘制光流向量图
void drawOpticalFlow(const Mat& oflow, //光流
	                 Mat &flowImage,   //绘制的图像
	                 int  stride,      //显示向量的步长
	                 float scale,      //放大因子
	                 const Scalar& color)  //显示向量的颜色
{
	//必要时创建图像
	if (flowImage.size() != oflow.size())
	{
		flowImage.create(oflow.size(), CV_8UC3);
		flowImage = Vec3i(255, 255, 255);
	}
	//对所有向量， 以stride作为步长
	for (int y = 0; y < oflow.rows; y += stride)
	{
		for (int x = 0; x < oflow.cols; x += stride) 
		{
			// 获取向量	
			Point2f vector = oflow.at< Point2f>(y, x);
			// 画线条
			line(flowImage, Point(x, y),
				Point(static_cast<int>(x + scale*vector.x + 0.5),
					static_cast<int>(y + scale*vector.y + 0.5)), color);
			// 画顶端圆圈	
			circle(flowImage, Point(static_cast<int>(x + scale*vector.x + 0.5),
				static_cast<int>(y + scale*vector.y + 0.5)), 1, color, -1);
		}
   }
}


int main()
{
	////1.跟踪视频中的特征点
	////创建视频处理类的实例
	//VideoProcessor processor;
	////创建特征跟踪类的实例
	//FeatureTracker tracker;
	////打开视频文件
	//processor.setInput("./images/bike.avi");
	////设置帧处理类
	//processor.setFrameProcessor(&tracker);
	////声明显示视频的窗口
	//processor.displayOutput("Tracked Feature");
	////以原始帧速率播放视频
	//processor.setDelay(1000. / processor.getFrameRate());

	//processor.stopAtFrameNo(90);
	////开始处理
	//processor.run();
	//
	//waitKey();


	////2.估算光流
	//Mat frame1 = imread("./images/goose/goose230.bmp", 0);
	//Mat frame2 = imread("./images/goose/goose237.bmp", 0);

	//imshow("frame1", frame1);
	//imshow("frame2", frame2);

	//// Combined display
	//Mat combined(frame1.rows, frame1.cols + frame2.cols, CV_8U);
	//frame1.copyTo(combined.colRange(0, frame1.cols));
	//frame2.copyTo(combined.colRange(frame1.cols, frame1.cols + frame2.cols));
	//imshow("Frames", combined);

	//// 创建光流算法
	//Ptr<DualTVL1OpticalFlow> tvl1 = createOptFlow_DualTVL1();

	//cout << "regularization coeeficient: " << tvl1->getLambda() << endl; // the smaller the soomther
	//cout << "Number of scales: " << tvl1->getScalesNumber() << endl; // number of scales
	//cout << "Scale step: " << tvl1->getScaleStep() << endl; // size between scales
	//cout << "Number of warpings: " << tvl1->getWarpingsNumber() << endl; // size between scales
	//cout << "Stopping criteria: " << tvl1->getEpsilon() << " and " << tvl1->getOuterIterations() << endl; // size between scales
	//														// compute the optical flow between 2 frames
	//Mat oflow; // 二维光流向量的图像
	//// 计算frame1和frame2之间的光流
	//tvl1->calc(frame1, frame2, oflow);

	//// 绘制光流
	//Mat flowImage;
	//drawOpticalFlow(oflow,     // 输入光流向量 
	//	flowImage, // 生成的图像
	//	8,         // 每隔8个像素显示一个向量
	//	2,         // 长度延长2倍
	//	Scalar(0, 0, 0)); // 向量颜色

	//imshow("Optical Flow", flowImage);

	//// 计算两个帧之间更加光滑的光流
	//tvl1->setLambda(0.075);
	//tvl1->calc(frame1, frame2, oflow);

	//// Draw the optical flow image
	//Mat flowImage2;
	//drawOpticalFlow(oflow,     // input flow vectors 
	//	flowImage2, // image to be generated
	//	8,         // display vectors every 8 pixels
	//	2,         // multiply size of vectors by 2
	//	Scalar(0, 0, 0)); // vector color

	//imshow("Smoother Optical Flow", flowImage2);
 //   waitKey();


   
	//3.跟踪视频中的物体
	// 创建视频处理器实例
	VideoProcessor processor;

	// 生成文件名
	std::vector<std::string> imgs;
	std::string prefix = "images/goose/goose";
	std::string ext = ".bmp";

	// 添加用于跟踪的图像名称
	for (long i = 130; i < 317; i++) {

		string name(prefix);
		ostringstream ss;
		ss << setfill('0') << setw(3) << i;
		name += ss.str();
		name += ext;

		cout << name << endl;
		imgs.push_back(name);
	}

	// 创建特征提取器实例
	Ptr<TrackerMedianFlow> ptr = TrackerMedianFlow::create();
	VisualTracker tracker(ptr);
	// VisualTracker tracker(TrackerKCF::createTracker());

	// 打开视频文件
	processor.setInput(imgs);

	// 设置帧处理器
	processor.setFrameProcessor(&tracker);

	// 声明显示视频的窗口
	processor.displayOutput("Tracked object");

	// 定义显示的帧速率
	processor.setDelay(50);

	// 指定初始目标位置
	cv::Rect bb(290, 100, 65, 40);
	tracker.setBoundingBox(bb);

	// 开始跟踪
	processor.run();

	cv::waitKey();

	// 中值流量跟踪算法
	Mat image1 = imread("./images/goose/goose130.bmp", ImreadModes::IMREAD_GRAYSCALE);

	// 定义一个10 * 10 的网格
	vector<Point2f> grid;
	for (int i = 0; i < 10; i++) {
		for (int j = 0; j < 10; j++) {
			Point2f p(bb.x + i*bb.width / 10., bb.y + j*bb.height / 10);
			grid.push_back(p);
		}
	}

	// track in next image
	Mat image2 = imread("./images/goose/goose131.bmp", ImreadModes::IMREAD_GRAYSCALE);
	vector<Point2f> newPoints;
	vector<uchar> status; // status of tracked features
	vector<float> err;    // error in tracking

							   // track the points
	calcOpticalFlowPyrLK(image1, image2, // 2 consecutive images
		grid,      // input point position in first image
		newPoints, // output point postion in the second image
		status,    // tracking success
		err);      // tracking error

				   // Draw the points
	for (Point2f p : grid)
	{

	   circle(image1, p, 1, Scalar(255, 255, 255), -1);
	}
	imshow("Initial points", image1);

	for (Point2f p : newPoints) {

		circle(image2, p, 1, Scalar(255, 255, 255), -1);
	}
	imshow("Tracked points", image2);

	waitKey();

	return 0;
}




#pragma once
#if !defined FTRACKER
#define FTRACKER

#include <iostream>
#include <string>
#include <vector>

#include <opencv2/opencv.hpp>

#include "VideoProcessor.h"


class FeatureTracker :public FrameProcessor
{
private:
	//当前的灰度图像
	Mat gray;
	//上一个灰度图像
	Mat gray_prev;
	//被跟踪的特征，从0到1
	vector<Point2f> points[2];
	//被跟踪特征点的初始位置
	vector<Point2f> initial;
	//被监测的特征
	vector<Point2f> features;
	//检测特征点的最大个数
	int max_count;
	//检测特征点的质量等级
	double qlevel;
	//两个特征点之间的最小差距
	double minDist;
	//被跟踪特征的状态
	vector<uchar> status;
	//跟踪中出现的误差
	vector<float> err;
public:

	FeatureTracker():max_count(500), qlevel(0.01), minDist(10.){}

	//处理方法
	void process(Mat &frame, Mat &output)
	{
		//装换成灰度图
		cvtColor(frame, gray, CV_BGR2GRAY);
		frame.copyTo(output);
		//1.如果必须添加新的特征点
		if (addNewPoints())
		{
			//检测特征点
			detectFeaturePoints();
			//在当前跟踪列表中添加检测的特征点
			points[0].insert(points[0].end(), features.begin(), features.end());
			initial.insert(initial.end(), features.begin(), features.end());
		}
		//对于序列中的第一幅图像
		if (gray_prev.empty())
			gray.copyTo(gray_prev);
		//2.跟踪特征
		calcOpticalFlowPyrLK(gray_prev, gray, //两个连续图像
			points[0],       //输入第一幅图像的特征点位置
			points[1],       //输入第二幅图像的特征点位置
			status,          //跟踪成功
			err);              //跟踪误差
		//3.循环检查被跟踪的特征点，剔除部分特征点
		int k = 0;
		for (int i = 0; i < points[1].size(); i++)
		{
			//是否保留这个特征点
			if (acceptTrackedPoint(i))
			{
				//在向量中保留这个特征点
				initial[k] = initial[i];
				points[1][k++] = points[1][i];
			}
		}
		//删除跟踪失败的特征点
		points[1].resize(k);
		initial.resize(k);
		//4.处理已经认可的被跟踪特征点
		handleTrackedPoints(frame, output);
		//5.让当前特征点和图像变成前一个
		swap(points[1], points[0]);
		swap(gray_prev, gray);
	}

	//特征点检测方法
	void detectFeaturePoints()
	{
		//检测特征点
		goodFeaturesToTrack(gray,  //图像
			features,  //输出检测到的特征点
			max_count, //特征点的最大数量
			qlevel,    //质量等级
			minDist);   //特征点之间的最小差距
	}

	//判断是否需要添加新的特征点
	bool addNewPoints()
	{
		//如果特征点数量太少
		return points[0].size() <= 10;
	}

	//判断需要保留的特征点
	bool acceptTrackedPoint(int i)
	{
		return status[i] &&
			//如果特征点已将移动
			(abs(points[0][i].x - points[1][i].x) +
			(abs(points[0][i].y - points[1][i].y)) > 2);
	}

	//处理当前跟踪的特征点
	void handleTrackedPoints(Mat &frame, Mat &output)
	{
		//遍历所有特征点
		for (int i = 0; i < points[1].size(); i++)
		{
			//画线和圆
			line(output, initial[i], points[1][i], Scalar(255, 255, 255));
			circle(output, points[1][i], 3, Scalar(255, 255, 255), -1);
		}		
	
	}
};

#endif // !defined FTRACKER




#pragma once
#if !defined VTRACKER
#define VTRACKER

#include <iostream>

#include <opencv2/opencv.hpp>
#include <opencv2/tracking/tracker.hpp>

#include "VideoProcessor.h"

using namespace std;
using namespace cv;


class VisualTracker : public FrameProcessor
{
private:
	Ptr<Tracker> tracker;
	Rect2d box;
	bool reset;
public:
	//构造函数指定选用的跟踪器
	VisualTracker(Ptr<Tracker> tracker) :reset(true), tracker(tracker) {}
	
	//设置矩形，以启动跟踪过程
	void setBoundingBox(const Rect2d& bb) {

		box = bb;
		reset = true;
	}
	//回调函数
	void process(Mat &frame, Mat &output)
	{
		if (reset)
		{ 
			// 新跟踪会话
			reset = false;
			tracker->init(frame, box);
		}
		else 
		{ 
			// 更新目标位置
			tracker->update(frame, box);
		}

		// 在当前帧中绘制矩形
		frame.copyTo(output);
		rectangle(output, box, Scalar(255, 255, 255), 2);
	}
};

#endif // !defined VTRACKER

```

### Chapter_14  :    实用案列

> 1.用最邻近局部二值模式实现人脸识别
> 2.通过级联Haar特征实现物体和人脸定位
> 3.用支持向量机和方向梯度直方图实现物体与行人检测

<!--more-->

```c++
#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/face.hpp>
#include <iomanip>
#include <opencv2/objdetect.hpp>
#include <opencv2/ml.hpp>

using namespace std;
using namespace cv;


//1.用最邻近局部二值模式实现人脸识别
//计算灰度图像的局部二值模式
void lbp(const Mat &image, Mat &result)
{
	assert(image.channels() == 1); // input image must be gray scale

	result.create(image.size(), CV_8U); // 必要时分配空间
										
     // 逐行处理，除了第一行和最后一行
	for (int j = 1; j<image.rows - 1; j++)
	{
		// 输入行的指针
		const uchar* previous = image.ptr<const uchar>(j - 1);
		const uchar* current = image.ptr<const uchar>(j);	 
		const uchar* next = image.ptr<const uchar>(j + 1); 

		uchar* output = result.ptr<uchar>(j);	// 输出行

		for (int i = 1; i<image.cols - 1; i++) {

			// 构建局部二值模式
			*output = previous[i - 1] > current[i] ? 1 : 0;
			*output |= previous[i] > current[i] ? 2 : 0;
			*output |= previous[i + 1] > current[i] ? 4 : 0;

			*output |= current[i - 1] > current[i] ? 8 : 0;
			*output |= current[i + 1] > current[i] ? 16 : 0;

			*output |= next[i - 1] > current[i] ? 32 : 0;
			*output |= next[i] > current[i] ? 64 : 0;
			*output |= next[i + 1] > current[i] ? 128 : 0;

			output++; // 下一个像素
		}
	}

	// 未处理的设置为0
	result.row(0).setTo(Scalar(0));
	result.row(result.rows - 1).setTo(Scalar(0));
	result.col(0).setTo(Scalar(0));
	result.col(result.cols - 1).setTo(Scalar(0));
}


//3.用支持向量机和方向梯度直方图实现物体与行人检测
// 画出一个单元格的HOG
void drawHOG(vector<float>::const_iterator hog, // HOG迭代器
	         int numberOfBins,        // HOG中的箱子数量
	         Mat &image,              // 单元格的图像
	         float scale = 1.0)       // 长度缩放比例
{       

	const float PI = 3.1415927;
	float binStep = PI / numberOfBins;
	float maxLength = image.rows;
	float cx = image.cols / 2.;
	float cy = image.rows / 2.;

	// 逐个箱子
	for (int bin = 0; bin < numberOfBins; bin++) {

		// 箱子方向
		float angle = bin*binStep;
		float dirX = cos(angle);
		float dirY = sin(angle);
		// 线条长度， 与箱子大小成正比
		float length = 0.5*maxLength* *(hog + bin);

		// 画线条
		float x1 = cx - dirX * length * scale;
		float y1 = cy - dirY * length * scale;
		float x2 = cx + dirX * length * scale;
		float y2 = cy + dirY * length * scale;
		line(image, Point(x1, y1), Point(x2, y2), CV_RGB(255, 255, 255), 1);
	}
}

// 在图像上绘制 HOG
void drawHOGDescriptors(const Mat &image,  // 输入图像
	                    Mat &hogImage,     // 结果HOG图像
	                    Size cellSize,     // 每个单元格的大小（忽略区块）
	                    int nBins)         // 箱子数量
{                             

	// 区块大小等于图像大小
	HOGDescriptor hog(Size( (image.cols / cellSize.width) * cellSize.width,(image.rows / cellSize.height) * cellSize.height),
		              Size( (image.cols / cellSize.width) * cellSize.width,(image.rows / cellSize.height) * cellSize.height),
		              cellSize,    // 区块步长（这里只有一个区块）
		              cellSize,    // 单元格大小
		              nBins);      // 箱子数量

	// 计算 HOG
	vector<float> descriptors;
	hog.compute(image, descriptors);

	float scale = 2.0 / * max_element(descriptors.begin(), descriptors.end());

	hogImage.create(image.rows, image.cols, CV_8U);

	vector<float>::const_iterator itDesc = descriptors.begin();

	for (int i = 0; i < image.rows / cellSize.height; i++) {
		for (int j = 0; j < image.cols / cellSize.width; j++) {
			// 画出每个单元格
			hogImage(Rect(j*cellSize.width, i*cellSize.height, cellSize.width, cellSize.height));
			drawHOG(itDesc, nBins,
				    hogImage(Rect(j*cellSize.width, i*cellSize.height,cellSize.width, cellSize.height)), scale);
			itDesc += nBins;
		}
	}
}



int main()
{

	////1.用最邻近局部二值模式实现人脸识别
	//Mat image = imread("./images/girl.jpg", IMREAD_GRAYSCALE);

	//imshow("Original image", image);

	//Mat lbpImage;
	//lbp(image, lbpImage);

	//imshow("LBP image", lbpImage);

	//Ptr<face::FaceRecognizer> recognizer =  face::LBPHFaceRecognizer::create(1,   //LBP模式的半径
	//	8,     //使用邻近像素的数量
	//	8, 8,  //网格大小
	//	200.); //最邻近的距离阈值

	////参考图像和标签的向量 
	//vector<Mat> referenceImages;
	//vector<int> labels;
	////打开参考图像
	//referenceImages.push_back(imread("./images/face0_1.png", IMREAD_GRAYSCALE));
	//labels.push_back(0); // 编号为0的人
	//referenceImages.push_back(imread("./images/face0_2.png", IMREAD_GRAYSCALE));
	//labels.push_back(0); // 编号为0的人
	//referenceImages.push_back(imread("./images/face0_2.png", IMREAD_GRAYSCALE));
	//labels.push_back(0); // 编号为0的人
	//referenceImages.push_back(imread("./images/face1_1.png", IMREAD_GRAYSCALE));
	//labels.push_back(1); // 编号为1的人
	//referenceImages.push_back(imread("./images/face1_2.png", IMREAD_GRAYSCALE));
	//labels.push_back(1); // 编号为1的人
	//referenceImages.push_back(imread("./images/face1_2.png", IMREAD_GRAYSCALE));
	//labels.push_back(1); // 编号为1的人


	///*Mat i1 = imread("./images/face0_1.png", IMREAD_GRAYSCALE);
	//Mat i2 = imread("./images/face0_2.png", IMREAD_GRAYSCALE);
	//imshow("i1", i1);
	//imshow("i2", i2);*/

	//// the 6 positive samples
	//Mat faceImages(2 * referenceImages[0].rows, 3 * referenceImages[0].cols, CV_8U);
	//for (int i = 0; i < 2; i++)
	//	for (int j = 0; j < 3; j++) {

	//		referenceImages[i * 2 + j].copyTo(faceImages(Rect(j*referenceImages[i * 2 + j].cols, i*referenceImages[i * 2 + j].rows, referenceImages[i * 2 + j].cols, referenceImages[i * 2 + j].rows)));
	//	}

	//resize(faceImages, faceImages, Size(), 0.5, 0.5);
	//imshow("Reference faces", faceImages);

	////通过计算LBPH进行训练
	//recognizer->train(referenceImages, labels);

	//int predictedLabel = -1;
	//double confidence = 0.0;

	//// Extract a face image
	//Mat inputImage;
	//resize(image(Rect(160, 75, 90, 90)), inputImage, Size(256, 256));
	//imshow("Input image", inputImage);

	//// 识别图像对应的编号
	//recognizer->predict(inputImage,     // 人脸图像 
	//	predictedLabel, // 识别结果
	//	confidence);    // 置信度

	//cout << "Image label= " << predictedLabel << " (" << confidence << ")" << endl;



	////2.通过级联Haar特征实现物体和人脸定位
	//// open the positive sample images
	//std::vector<cv::Mat> referenceImages;
	//referenceImages.push_back(cv::imread("./images/stopSamples/stop00.png"));
	//referenceImages.push_back(cv::imread("./images/stopSamples/stop01.png"));
	//referenceImages.push_back(cv::imread("./images/stopSamples/stop02.png"));
	//referenceImages.push_back(cv::imread("./images/stopSamples/stop03.png"));
	//referenceImages.push_back(cv::imread("./images/stopSamples/stop04.png"));
	//referenceImages.push_back(cv::imread("./images/stopSamples/stop05.png"));
	//referenceImages.push_back(cv::imread("./images/stopSamples/stop06.png"));
	//referenceImages.push_back(cv::imread("./images/stopSamples/stop07.png"));

	//// create a composite image
	//cv::Mat positveImages(2 * referenceImages[0].rows, 4 * referenceImages[0].cols, CV_8UC3);
	//for (int i = 0; i < 2; i++)
	//	for (int j = 0; j < 4; j++) {

	//		referenceImages[i * 2 + j].copyTo(positveImages(cv::Rect(j*referenceImages[i * 2 + j].cols, i*referenceImages[i * 2 + j].rows, referenceImages[i * 2 + j].cols, referenceImages[i * 2 + j].rows)));
	//	}

	//cv::imshow("Positive samples", positveImages);

	//cv::Mat negative = cv::imread("./images/stopSamples/bg01.jpg");
	//cv::resize(negative, negative, cv::Size(), 0.33, 0.33);
	//cv::imshow("One negative sample", negative);

	//cv::Mat inputImage = cv::imread("./images/stopSamples/stop9.jpg");
	//cv::resize(inputImage, inputImage, cv::Size(), 0.5, 0.5);

	//cv::CascadeClassifier cascade;
	//if (!cascade.load("./images/stopSamples/classifier/cascade.xml")) {
	//	std::cout << "Error when loading the cascade classfier!" << std::endl;
	//	return -1;
	//}

	//// predict the label of this image
	//std::vector<cv::Rect> detections;

	//cascade.detectMultiScale(inputImage, // 输入图像
	//	detections, // 检测结果
	//	1.1,        // 缩小比例
	//	1,          // 所需近邻数量
	//	0,          // 标志位（不用）
	//	cv::Size(48, 48),    // 检测对象的最小尺寸
	//	cv::Size(128, 128)); // 检测对象的最大尺寸

	//std::cout << "detections= " << detections.size() << std::endl;
	//for (int i = 0; i < detections.size(); i++)
	//	cv::rectangle(inputImage, detections[i], cv::Scalar(255, 255, 255), 2);

	//cv::imshow("Stop sign detection", inputImage);

	//// Detecting faces
	//cv::Mat picture = cv::imread("./images/girl.jpg");
	//cv::CascadeClassifier faceCascade;
	//if (!faceCascade.load("D:\\Project_OpenCV\\Cmake\\install\\etc\\haarcascades\\haarcascade_frontalface_default.xml")) {
	//	std::cout << "Error when loading the face cascade classfier!" << std::endl;
	//	return -1;
	//}

	//faceCascade.detectMultiScale(picture, // input image 
	//	detections, // detection results
	//	1.1,        // scale reduction factor
	//	3,          // number of required neighbor detections
	//	0,          // flags (not used)
	//	cv::Size(48, 48),    // minimum object size to be detected
	//	cv::Size(128, 128)); // maximum object size to be detected

	//std::cout << "detections= " << detections.size() << std::endl;
	//// draw detections on image
	//for (int i = 0; i < detections.size(); i++)
	//	cv::rectangle(picture, detections[i], cv::Scalar(255, 255, 255), 2);

	//// Detecting eyes
	//cv::CascadeClassifier eyeCascade;
	//if (!eyeCascade.load("D:\\Project_OpenCV\\Cmake\\install\\etc\\haarcascades\\haarcascade_eye.xml")) {
	//	std::cout << "Error when loading the eye cascade classfier!" << std::endl;
	//	return -1;
	//}

	//eyeCascade.detectMultiScale(picture, // input image 
	//	detections, // detection results
	//	1.1,        // scale reduction factor
	//	3,          // number of required neighbor detections
	//	0,          // flags (not used)
	//	cv::Size(24, 24),    // minimum object size to be detected
	//	cv::Size(64, 64)); // maximum object size to be detected

	//std::cout << "detections= " << detections.size() << std::endl;
	//// draw detections on image
	//for (int i = 0; i < detections.size(); i++)
	//	cv::rectangle(picture, detections[i], cv::Scalar(0, 0, 0), 2);

	//cv::imshow("Detection results", picture);


	//3.用支持向量机和方向梯度直方图实现物体与行人检测
    Mat image = imread("./images/girl.jpg", IMREAD_GRAYSCALE);
    imshow("Original image", image);

    HOGDescriptor hog(Size((image.cols / 16) * 16, (image.rows / 16) * 16), // 窗口大小
	                  Size(16, 16),    // 区块大小
	                  Size(16, 16),    // 区块步长
	                  Size(4, 4),      // 单元格大小
                      9);              // 箱子数量

    vector<float> descriptors;

    // 调用 drawHOGDescriptors函数，在图像上绘制HOG
    Mat hogImage = image.clone();
    drawHOGDescriptors(image, hogImage, Size(16, 16), 9); 
	imshow("HOG image", hogImage);

    // generate the filename
    vector<string> imgs;
    string prefix = "./images/stopSamples/stop";
    string ext = ".png";

    // loading 8 positive samples
    vector<Mat> positives;

    for (long i = 0; i < 8; i++) 
	{

	     string name(prefix);
	     ostringstream ss; 
		 ss << setfill('0') << setw(2) << i; 
		 name += ss.str();
	     name += ext;

	     positives.push_back(imread(name, IMREAD_GRAYSCALE));
    }

    // the first 8 positive samples
    Mat posSamples(2 * positives[0].rows, 4 * positives[0].cols, CV_8U);
    for (int i = 0; i < 2; i++)
	  for (int j = 0; j < 4; j++)
	  {

		positives[i * 4 + j].copyTo(posSamples(Rect(j*positives[i * 4 + j].cols, i*positives[i * 4 + j].rows, positives[i * 4 + j].cols, positives[i * 4 + j].rows)));

	  }

    imshow("Positive samples", posSamples);


    // loading 8 negative samples
    string nprefix = "./images/stopSamples/neg";
    vector<Mat> negatives;

    for (long i = 0; i < 8; i++) 
	{

	     string name(nprefix);
	     ostringstream ss;
		 ss << setfill('0') << setw(2) << i;
		 name += ss.str();
	     name += ext;

	     negatives.push_back(imread(name, IMREAD_GRAYSCALE));
    }

    // the first 8 negative samples
    Mat negSamples(2 * negatives[0].rows, 4 * negatives[0].cols, CV_8U);
    for (int i = 0; i < 2; i++)
	for (int j = 0; j < 4; j++)
	{

		negatives[i * 4 + j].copyTo(negSamples(cv::Rect(j*negatives[i * 4 + j].cols, i*negatives[i * 4 + j].rows, negatives[i * 4 + j].cols, negatives[i * 4 + j].rows)));
	}

    imshow("Negative samples", negSamples);

    // The HOG descriptor for stop sign detection
    HOGDescriptor hogDesc(positives[0].size(), // 窗口大小
	                      Size(8, 8),    // 区块大小
	                      Size(4, 4),    // 区块步长
	                      Size(4, 4),    // 单元格大小
	                      9);            //  箱子数量

    // 计算第一个描述子
    vector<float> desc;
    hogDesc.compute(positives[0], desc);

    cout << "Positive sample size: " << positives[0].rows << "x" << positives[0].cols << endl;
    cout << "HOG descriptor size: " << desc.size() << endl;

    // 样本描述子矩阵
    int featureSize = desc.size();
    int numberOfSamples = positives.size() + negatives.size();
    // 创建存储样本HOG的矩阵
    Mat samples(numberOfSamples, featureSize, CV_32FC1);

    // 用第一个描述子填第一行
    for (int i = 0; i < featureSize; i++)
	    samples.ptr<float>(0)[i] = desc[i];

    // 计算正样本的描述子
    for (int j = 1; j < positives.size(); j++) 
	{
	   hogDesc.compute(positives[j], desc);
	   // 用当前描述子填下一行
	   for (int i = 0; i < featureSize; i++)
		    samples.ptr<float>(j)[i] = desc[i];
     }

     // 计算负样本的描述子
     for (int j = 0; j < negatives.size(); j++) 
	 {
	     hogDesc.compute(negatives[j], desc);
	     // 用当前描述子填下一行
	     for (int i = 0; i < featureSize; i++)
		    samples.ptr<float>(j + positives.size())[i] = desc[i];
      }

     // 创建标签
     Mat labels(numberOfSamples, 1, CV_32SC1);
     // 正样本的标签
     labels.rowRange(0, positives.size()) = 1.0;
     // 负样本的标签
     labels.rowRange(positives.size(), numberOfSamples) = -1.0;

    // 创建SVM分类器
    Ptr<ml::SVM> svm = ml::SVM::create();
    svm->setType(ml::SVM::C_SVC);
    svm->setKernel(ml::SVM::LINEAR);

    // 准备训练数据
    Ptr<ml::TrainData> trainingData =
    ml::TrainData::create(samples, ml::SampleTypes::ROW_SAMPLE, labels);

    // SVM 训练
    svm->train(trainingData);

    Mat queries(4, featureSize, CV_32FC1);

    // 每行填入查询描述子
    hogDesc.compute(imread("./images/stopSamples/stop08.png", IMREAD_GRAYSCALE), desc);
    for (int i = 0; i < featureSize; i++)
	     queries.ptr<float>(0)[i] = desc[i];
         hogDesc.compute(imread("./images/stopSamples/stop09.png", IMREAD_GRAYSCALE), desc);
    for (int i = 0; i < featureSize; i++)
	     queries.ptr<float>(1)[i] = desc[i];
         hogDesc.compute(imread("./images/stopSamples/neg08.png", IMREAD_GRAYSCALE), desc);
    for (int i = 0; i < featureSize; i++)
	    queries.ptr<float>(2)[i] = desc[i];
        hogDesc.compute(imread("./images/stopSamples/neg09.png", IMREAD_GRAYSCALE), desc);
    for (int i = 0; i < featureSize; i++)
	    queries.ptr<float>(3)[i] = desc[i];
    Mat predictions;

    // 测试分类器
    svm->predict(queries, predictions);

    for (int i = 0; i < 4; i++)
	    cout << "query: " << i << ": " << ((predictions.at<float>(i) < 0.0) ? "Negative" : "Positive") << endl;

   // 行人检测
   Mat myImage = imread("./images/person.jpg", IMREAD_GRAYSCALE);

   // 创建检测器
   vector<Rect> peoples;
   HOGDescriptor peopleHog;
   peopleHog.setSVMDetector(HOGDescriptor::getDefaultPeopleDetector());
   // 检测图像中的行人
   peopleHog.detectMultiScale(myImage, // 输入图像
	                          peoples, // 输出矩形列表 
	                          0,       // 判断检测结果是否有效的阈值
	                          Size(4, 4),       // 窗口步长
	                          Size(32, 32),     // 填充图像
	                          1.1,              // 缩放比列
	                          2);               // 分组阈值

   // draw detections on image
   cout << "Number of peoples detected: " << peoples.size() << endl;
   for (int i = 0; i < peoples.size(); i++)
	  rectangle(myImage, peoples[i], Scalar(255, 255, 255), 2);

   imshow("People detection", myImage);


	waitKey(0);
	return 0;
}
```


