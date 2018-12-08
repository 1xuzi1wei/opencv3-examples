#include "pch.h"
#include <opencv2/core/core.hpp>
#include <opencv2/opencv.hpp>
#include "handwriting_recognition.h"
#include "mnist.h"
#include <string>
#include <iostream>

/*
用KNN算法实现手写数字识别
*/

using namespace std;
using namespace cv;

//计时器
double cost_time;
clock_t start_time;
clock_t end_time;


int reverseInt(int i) {
	unsigned char ch1, ch2, ch3, ch4;
	ch1 = i & 255;
	ch2 = (i >> 8) & 255;
	ch3 = (i >> 16) & 255;
	ch4 = (i >> 24) & 255;
	return((int)ch1 << 24) + ((int)ch2 << 16) + ((int)ch3 << 8) + ch4;
}

Mat read_mnist_image(const string fileName) {
	int magic_number = 0;
	int number_of_images = 0;
	int n_rows = 0;
	int n_cols = 0;
	Mat train_images;

	ifstream file(fileName, ios::binary);

	if (file.is_open()) {

		cout << "成功打开图像集....\n";
		file.read((char*)&magic_number, sizeof(magic_number));
		file.read((char*)&number_of_images, sizeof(number_of_images));
		file.read((char*)&n_rows, sizeof(n_rows));
		file.read((char*)&n_cols, sizeof(n_cols));
		cout << magic_number << " " << number_of_images << " " << n_rows << " " << n_cols << endl;

		//小端储存转换
		magic_number = reverseInt(magic_number);
		number_of_images = reverseInt(number_of_images);
		n_rows = reverseInt(n_rows);
		n_cols = reverseInt(n_cols);
		cout << "magic number = " << magic_number << endl;
		cout << "number of images = " << number_of_images << endl;
		cout << "rows = " << n_rows << endl;
		cout << "cols = " << n_cols << endl;

		cout << "开始读取Image数据......\n";
		start_time = clock();
		train_images = Mat::zeros(number_of_images, n_rows * n_cols, CV_32FC1);
		for (int i = 0; i < number_of_images; i++) {
			int index = 0;
			for (int r = 0; r < n_rows; r++) {
				for (int c = 0; c < n_cols; c++) {
					unsigned char temp = 0;
					file.read((char*)&temp, sizeof(temp));
					index = r * n_cols + c;
					train_images.at<uchar>(i, index) = (int)temp;
				}
			}
		}
		end_time = clock();
		cost_time = (end_time - start_time);
		cout << "读取Image数据完毕......" << cost_time << "s\n";

		waitKey(0);
	}
	file.close();
	return train_images;
}

Mat read_mnist_label(const string fileName) {
	int magic_number;
	int number_of_items;
	Mat labels;
	ifstream file(fileName, ios::binary);

	if (file.is_open())
	{
		cout << "成功打开Label集 ... \n";
		file.read((char*)&magic_number, sizeof(magic_number));
		file.read((char*)&number_of_items, sizeof(number_of_items));
		cout << magic_number << " " << number_of_items << endl;

		magic_number = reverseInt(magic_number);
		number_of_items = reverseInt(number_of_items);
		cout << "magic number = " << magic_number << endl;
		cout << "number of items = " << number_of_items << endl;

		cout << "开始读取Label数据......\n";
		start_time = clock();
		labels = Mat::zeros(number_of_items, 1, CV_32SC1);
		for (int i = 0; i < number_of_items; i++) {
			unsigned char temp = 0;
			file.read((char*)&temp, sizeof(temp));
			labels.at<uchar>(i, 0) = (int)temp;
		}
		end_time = clock();
		cost_time = (end_time - start_time);
		cout << "读取Image数据完毕......" << cost_time << "s\n";
		waitKey(0);
	}
	file.close();
	return labels;
}


//用knn方法训练
void knnTrain(Mat trainData, Mat labels) {
	int k = 5;
	Ptr<ml::KNearest> knn = ml::KNearest::create();
	knn->setDefaultK(k);
	knn->setIsClassifier(true);
	Ptr<ml::TrainData> tdata = ml::TrainData::create(trainData, ml::ROW_SAMPLE, labels);
	knn->train(tdata);
	knn->save("knnTest.html");
}

//用测试集进行测试
void testMnist(Mat testData, Mat tLabels) {
	float correct = 0.0;
	Ptr<ml::KNearest> knn = Algorithm::load<ml::KNearest>("knnTest.html");
	int total = testData.rows;
	for (int i = 1; i < total; i++) {
		int actual = tLabels.at<int>(i);
		Mat sample = testData.row(i);
		float r = knn->predict(sample);
		int predict = static_cast<int>(r);
		if (predict == actual) {
			correct++;
		}
	}
	printf("\n recognize rate : %.2f \n", correct / total);
}

//设置测试集跟训练集的文件路径
string trainImage = "mnist\\train-images-idx3-ubyte.gz";
string trainLabel = "mnist\\train-labels-idx1-ubyte.gz";
string testImage = "mnist\t10k-images-idx3-ubyte.gz";
string testLabel = "mnist\t10k-labels-idx1-ubyte.gz";

int main()
{

	//建立训练集和测试集
	Mat trainData = read_mnist_image(trainImage);
	Mat labels = read_mnist_label(trainLabel);
	Mat testData = read_mnist_image(testImage);
	Mat tLabels = read_mnist_label(testLabel);

	//输出训练集和测试集中样本跟对应的标签的行跟列
	cout << trainData.rows << " " << trainData.cols << endl;
	cout << labels.rows << " " << labels.cols << endl;
	cout << testData.rows << " " << testData.cols << endl;
	cout << tLabels.rows << " " << tLabels.cols << endl;

	//利用knn算法训练
	knnTrain(trainData, labels);

	//用测试集进行测试
	testMnist(testData, tLabels);

	waitKey(0);
	return 0;
}

