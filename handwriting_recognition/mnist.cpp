#include "mnist.h"

//计时器
double cost_time;
clock_t start_time;
clock_t end_time;

//小端存储转换
int reverseInt(int i) {
	unsigned char ch1, ch2, ch3, ch4;
	ch1 = i & 255;
	ch2 = (i >> 8) & 255;
	ch3 = (i >> 16) & 255;
	ch4 = (i >> 24) & 255;
	return((int)ch1 << 24) + ((int)ch2 << 16) + ((int)ch3 << 8) + ch4;
}

//读取image数据集信息
Mat read_mnist_image(const string fileName) {
	int magic_number = 0;
	int number_of_images = 0;
	int n_rows = 0;
	int n_cols = 0;

	ifstream file(fileName, ios::binary);

	if (file.is_open()) {

		cout << "成功打开图像集....\n";
		file.read((char*)&magic_number, sizeof(magic_number));
		file.read((char*)&number_of_images, sizeof(number_of_images));
		file.read((char*)&n_rows, sizeof(n_rows));
		file.read((char*)&n_cols, sizeof(n_cols));
		cout << magic_number << " " << number_of_images << " " << n_rows << " " << n_cols << endl;

		//小端储存转换
		magic_number = reverseInt(magic_number)
		number_of_images = reverseInt(number_of_images);
		n_rows = reverseInt(n_rows);
		n_cols = reverseInt(n_cols);
		cout << "magic number = " << magic_number << endl;
		cout << "number of images = " << number_of_images << endl;
		cout << "rows = " << n_rows << endl;
		cout << "cols = " << n_cols << endl;

		cout << "开始读取Image数据......\n";
		start_time = clock();
		Mat train_images = Mat::zeros(number_of_images, n_rows * n_cols, CV_32FC1);
		for (int i = 0; i < number_of_images; i++) {
			int index = 0;
			for (int r = 0; r < n_rows; r++) {
				for (int c = 0; c < n_cols; c++) {
					unsigned char temp = 0;
					file.read((char*)&temp, sizeof(temp));
					index = r * width + c;
					train_images.at<uchar>(i, index) = (int)temp;
				}
			}
		}
		end_time = clock();
		cost_time = (end_time - start_time) / CLOCKS_PRE_SEC;
		cout << "读取Image数据完毕......" << cost_time << "s\n";

		waitKey(0);
	}
	file.close();
	return train_images;
}

//读取label数据集信息
Mat read_mnist_label(const string fileName) {
	int magic_number;
	int number_of_items;

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
		Mat labels = Mat::zeros(number_of_items, 1, CV_32SC1);
		for (int i = 0; i < number_of_items; i++) {
					unsigned char temp = 0;
					file.read((char*)&temp, sizeof(temp));
					labels.at<uchar>(i, 0) = (int)temp;
		}
		end_time = clock();
		cost_time = (end_time - start_time) / CLOCKS_PRE_SEC;
		cout << "读取Image数据完毕......" << cost_time << "s\n";
		waitKey(0);
	}
	file.close();
	return labels;
}