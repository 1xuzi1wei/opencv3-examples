#include "pch.h"
#include <opencv2/core/core.hpp>
#include <opencv2/opencv.hpp>
#include "handwriting_recognition.h"
#include <string>
#include <iostream>

/*
用KNN算法实现手写数字识别
*/

using namespace std;
using namespace cv;
using namespace cv::ml;

int main()
{
	Mat trainData, trainLabels, testData, tLabels;

	string fileName = "digits.png";
	//建立训练集和测试集
	Mat img = imread(fileName);
	Mat gray;
	cvtColor(img, gray, CV_BGR2GRAY);
	int b = 20;
	int m = gray.rows / b;// m = 50;
	int n = gray.cols / b;// n = 100;
	Mat data;
	Mat labels;
	//Mat labels= Mat::zeros(5000, 10, CV_32F);//当ANN才使用
	for (int i = 0; i < n; i++) {
		int offsetCol = i * b;
		for (int j = 0; j < m; j++) {
			int offsetRow = j * b;
			Mat tmp;
			gray(Range(offsetRow, offsetRow + b), Range(offsetCol, offsetCol + b)).copyTo(tmp);
			data.push_back(tmp.reshape(0, 1));
			labels.push_back((int)j / 5);
			//labels.at<int>(m * i + j, (int)j / 5) = 1; //当ANN才使用
		}
	}
	data.convertTo(data, CV_32F);
	labels.convertTo(labels, CV_32F);
	data = data / 255;

	int sampleNum = data.rows;
	int trainNum = 3000;
	int testNum = sampleNum - trainNum;
	trainData = data(Range(0, trainNum), Range::all());
	trainLabels = labels(Range(0, trainNum), Range::all());
	testData = data(Range(trainNum, sampleNum), Range::all());
	tLabels = labels(Range(trainNum, sampleNum), Range::all());

	//cout << labels.cols << endl;
	/*
	for (int i = 600; i < 620; i++) {
		for (int j = 0; j < 10; j++) {
			cout << labels.at<int>(i, j) << ",";
		}
		cout << endl;
	}
	*/
	/*
	//利用knn算法训练
	int k = 5;
	Ptr<ml::KNearest> knn = ml::KNearest::create();
	knn->setDefaultK(k);
	knn->setIsClassifier(true);
	Ptr<ml::TrainData> tdata = ml::TrainData::create(trainData, ml::ROW_SAMPLE, trainLabels);
	knn->train(tdata);
	knn->save("KnnTest.text");
	*/


	//利用svm算法训练
	//创建svm分类器并且配置分类器属性
	Ptr<SVM> svm = SVM::create();
	svm->setType(SVM::C_SVC);
	svm->setKernel(SVM::RBF);
	svm->setGamma(0.075);
	svm->setDegree(0);
	svm->setC(25);
	svm->setCoef0(0);
	svm->setNu(0);
	svm->setP(0);
	svm->setTermCriteria(TermCriteria(TermCriteria::MAX_ITER, 100, FLT_EPSILON));
	//训练开始
	svm->train(trainData, ROW_SAMPLE, trainLabels);
	//svm->trainAuto(trainData, ROW_SAMPLE, trainLabels, 10, SVM::getDefaultGridPtr(SVM::C), SVM::getDefaultGridPtr(SVM::GAMMA), SVM::getDefaultGridPtr(SVM::P), SVM::getDefaultGridPtr(SVM::NU), SVM::getDefaultGridPtr(SVM::COEF), SVM::getDefaultGridPtr(SVM::DEGREE), false);
	svm->save("svmTest.text");


	/*
	//用人工神经网络训练
	Ptr<ANN_MLP> ann = ANN_MLP::create();
	Mat layerSizes = (Mat_<int>(1,3)<<400,5,10);
	ann->setLayerSizes(layerSizes);
	ann->setTrainMethod(ANN_MLP::BACKPROP, 0.001, 0.1);
	ann->setActivationFunction(ANN_MLP::SIGMOID_SYM, 1.0, 1.0);
	ann->setTermCriteria(TermCriteria(TermCriteria::MAX_ITER | TermCriteria::EPS, 100, 0.001));
	Ptr<TrainData> tdata = TrainData::create(trainData, ROW_SAMPLE, trainLabels);
	ann->train(tdata);
	ann->save("svmTest.text");
	*/

	//用测试集进行测试
	//Ptr<ml::KNearest> knN = KNearest::load("KnnTest.text");
	Ptr<SVM> svM = SVM::load("svmTest.text");
	double test_hr = 0;
	for (int i = 0; i < testNum; i++) {
		//int actual = tLabels.at<int>(i);
		Mat sample = testData.row(i);
		//float r = knN->predict(sample);
		float r = svM->predict(sample);
		r = std::abs(r - tLabels.at<int>(i)) <= FLT_EPSILON ? 1.f : 0.f;
		test_hr += r;
	}
	test_hr /= testNum;
	printf("accuracy: test = %.1f%%\n", test_hr*100.);

	waitKey(0);
	return 0;
}

