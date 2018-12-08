#include "pch.h"
#include <opencv2/core/core.hpp>
#include <opencv2/opencv.hpp>
#include "handwriting_recognition.h"
#include <string>
#include <iostream>

/*
��KNN�㷨ʵ����д����ʶ��
*/

using namespace std;
using namespace cv;
using namespace cv::ml;

Mat deal_camera(Mat srcImage);
int result;
vector<Mat> ROI;//���ڴ��ͼ�пٳ�����������
vector<Rect> ROIposition;//ROI��ͼ���е�λ��
vector<vector<Point> > contours;//�����������������ڴ�������ĵ������������
vector<Vec4i> hierarchy;//���ָ������
int weigth;//���
int height;//�߶�
Mat _roi;
Rect rect;
Point positiosn;
Ptr<ml::KNearest> knn;
int brx; //���½ǵĺ�����
int bry; //���½ǵ�������
int tlx; //���Ͻǵĺ�����
int tly; //���Ͻǵ�������


int main(int argc, char** argv)
{

	//opening the camera
	namedWindow("original", WINDOW_AUTOSIZE);
	VideoCapture cap;

	if (argc == 1) {
		cap.open(0);
	}
	else {
		cap.open(argv[1]);
	}
	if (!cap.isOpened()) {
		std::cerr << "Couldn't open capture." << std::endl;
	}

	//train data
	Mat trainData, trainLabels, testData, tLabels;

	string fileName = "digits.png";
	//����ѵ�����Ͳ��Լ�
	Mat img = imread(fileName);
	Mat gray;
	cvtColor(img, gray, CV_BGR2GRAY);
	int b = 20;
	int m = gray.rows / b;// m = 50;
	int n = gray.cols / b;// n = 100;
	Mat data;
	Mat labels;
	//Mat labels= Mat::zeros(5000, 10, CV_32F);//��ANN��ʹ��
	for (int i = 0; i < n; i++) {
		int offsetCol = i * b;
		for (int j = 0; j < m; j++) {
			int offsetRow = j * b;
			Mat tmp;
			gray(Range(offsetRow, offsetRow + b), Range(offsetCol, offsetCol + b)).copyTo(tmp);
			data.push_back(tmp.reshape(0, 1));
			labels.push_back((int)j / 5);
			//labels.at<int>(m * i + j, (int)j / 5) = 1; //��ANN��ʹ��
		}
	}
	data.convertTo(data, CV_32F);
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

	//����knn�㷨ѵ��
	int k = 5;
	knn = ml::KNearest::create();
	knn->setDefaultK(k);
	knn->setIsClassifier(true);
	Ptr<ml::TrainData> tdata = ml::TrainData::create(trainData, ml::ROW_SAMPLE, trainLabels);
	knn->train(tdata);
	knn->save("KnnTest.text");


	/*
	//����svm�㷨ѵ��
	//����svm�������������÷���������
	Ptr<SVM> svm = SVM::create();
	svm->setType(SVM::C_SVC);
	svm->setKernel(SVM::RBF);
	svm->setGamma(0.075);
	svm->setDegree(0);
	svm->setC(25);
	svm->setCoef0(0);
	svm->setNu(0);
	svm->setP(0);
	svm->setTermCriteria(TermCriteria(TermCriteria::MAX_ITER, 100,FLT_EPSILON ));
	//ѵ����ʼ
	svm->train(trainData, ROW_SAMPLE, trainLabels);
	//svm->trainAuto(trainData, ROW_SAMPLE, trainLabels, 10, SVM::getDefaultGridPtr(SVM::C), SVM::getDefaultGridPtr(SVM::GAMMA), SVM::getDefaultGridPtr(SVM::P), SVM::getDefaultGridPtr(SVM::NU), SVM::getDefaultGridPtr(SVM::COEF), SVM::getDefaultGridPtr(SVM::DEGREE), false);
	svm->save("svmTest.text");
	*/

	/*
	//���˹�������ѵ��
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

	//�ò��Լ����в���
	Ptr<ml::KNearest> knN = Algorithm::load<KNearest>("KnnTest.text");
	//Ptr<SVM> svM = SVM::load("svmTest.text");
	double test_hr = 0;
	for (int i = 0; i < testNum; i++) {
		//int actual = tLabels.at<int>(i);
		Mat sample = testData.row(i);
		float r = knN->predict(sample);
		//float r = svM->predict(sample);
		r = std::abs(r - tLabels.at<int>(i)) <= FLT_EPSILON ? 1.f : 0.f;
		test_hr += r;
	}
	test_hr /= testNum;
	printf("accuracy: test = %.1f%%\n", test_hr*100.);

	for (;;)
	{
		Mat frame;
		cap >> frame;
		frame = deal_camera(frame);
		imshow("original", frame);
		if (waitKey(30) >= 0)
			break;
	}

	waitKey(0);
	return 0;
}

Mat deal_camera(Mat srcImage) {
	Mat dstImage, grayImage, Image, blurImage, morphImage;
	srcImage.copyTo(dstImage);
	cvtColor(dstImage, grayImage, COLOR_BGR2GRAY);
	GaussianBlur(grayImage, blurImage, Size(3, 3), 3, 3);
	threshold(blurImage, Image, 120, 255, CV_THRESH_BINARY_INV);
	//Mat kernerl = getStructuringElement(MORPH_RECT, Size(3, 3), Point(-1, -1));
	//morphologyEx(Image, morphImage, MORPH_OPEN, kernerl, Point(-1, -1));
	findContours(Image, contours, hierarchy, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);
	//added for picture
	vector< vector<Point> >::iterator It;
	for (It = contours.begin(); It < contours.end(); It++)
	{   //�����ɰ�Χ���ֵ���С��
		rect = boundingRect(*It);
		weigth = rect.br().x - rect.tl().x;//��
		height = rect.br().y - rect.tl().y;//��
		Mat tmp, predict_mat;
		if ((weigth < height) && (height > 10))//�����޸�
		{  //�������ֵ������ų���һЩ���ܲ������ֵ�ͼ�Σ�Ȼ�����һ�´���
			Mat roi = Image(rect);
			roi.copyTo(_roi);//�������
			rectangle(srcImage, rect, Scalar(0, 255, 0), 1, 8);
			Mat pre;
			resize(_roi, pre, Size(20, 20));
			pre.convertTo(pre, CV_32F);
			pre.copyTo(tmp);
			predict_mat.push_back(tmp.reshape(0, 1));
			//Mat predict_simple = predict_mat.row(i);
			result = (int)knn->predict(predict_mat);
			char output[10] = { 0 };
			sprintf_s(output, "%d", result);
			positiosn = Point(rect.br().x - 7, rect.br().y + 25);
			putText(srcImage, output, positiosn, 1, 1.0, Scalar(0, 0, 0), 1);//����Ļ�ϴ�ӡ��
			//cout << result << endl;
		}
	}
	return srcImage;
}
