#include <iostream>
#include <opencv2\core\core.hpp>
#include <opencv2\highgui\highgui.hpp>
#include <opencv2\imgproc\imgproc.hpp>

using namespace std;
using namespace cv;

int main(int argc, char ** argv) {

	cout << "start" << endl;

	//ouvrir une image
	Mat img = imread("../data/lena.bmp");

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

	//créer masque binaire
	Mat mask(img.size(), CV_8UC1, 255);

	//définir une roi rectangulaire
	Mat roi(mask, Rect(100,100,50,50));

	//appliquer roi à l'image, masque nul
	roi = Scalar(0,0,0);

	/*--------------------------------------
	Priorités
	---------------------------------------*/
	//ligne de front

	//calculer le laplacien de mask, stocké dans une autre image
	Mat laplace(mask.size(), mask.type());

	Laplacian(mask, laplace, mask.depth());

	//ligne de front contenue dans le laplacien
	//calcul des priorités

	//paramètres du patch
	int t_patch = 9;
	int half = t_patch/2;


	for (int y=0 ; y < laplace.size().height ; y++) {
		for (int x=0 ; x < laplace.size().width ; x++) {

			//si ligne de front
			if (laplace.at<uchar>(y,x) != 0) {
				//compter connus et inconnus, appliquer patch
				
				int connus = 0;
				for (int i=0 ; i<t_patch ; i++) {
					for (int j=0 ; j<t_patch ; j++) {
						if (mask.at<uchar>(y-half+i,x-half+j) =! 0) {
							//connu
							connus++;
						}
					}
				}
				//fin de compte
				//mettre à jour priorité


			}

		}
	}

	


	/*
	namedWindow("lena_trou", CV_WINDOW_AUTOSIZE);
	imshow("lena_trou", img);
	waitKey(0);
	*/

	return 0;
}