#include <iostream>
#include <opencv2\core\core.hpp>
#include <opencv2\highgui\highgui.hpp>

using namespace std;
using namespace cv;

int main(int argc, char ** argv) {

	cout << "start" << endl;

	//ouvrir une image
	Mat img = imread("../data/lena.bmp", CV_LOAD_IMAGE_COLOR);

	int taille = img.size().area();

	if (!img.data) {
		cout << "echec" << endl;
	}

	cout << taille << endl;

	//cout << img << endl;
	//afficher l'image
	namedWindow("lena", CV_WINDOW_AUTOSIZE);
	imshow("lena", img);

	waitKey(0);

	cout << "succes2" << endl;

	return 0;
}