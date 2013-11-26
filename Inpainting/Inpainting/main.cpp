#include <iostream>
#include <string>
#include <opencv2\core\core.hpp>
#include <opencv2\highgui\highgui.hpp>
#include <opencv2\imgproc\imgproc.hpp>

using namespace std;
using namespace cv;

int main(int argc, char ** argv) {

	cout << "start" << endl;

	string dataDir("../data/");	//chemin vers les données
	string outDir(dataDir + "output/");	//chemin du repertoire de sortie

	//ouvrir une image
	string imgFile("lena.ppm");

	Mat img = imread(dataDir + imgFile);

	if (!img.data) {
		cout << "echec" << endl;
		waitKey(0);
		return 0;
	}

	//afficher l'image
	namedWindow("lena", CV_WINDOW_AUTOSIZE);
	imshow("lena", img);

	waitKey(0);
	
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
	Mat laplace(mask.size(), CV_32FC1);

	Laplacian(mask, laplace, laplace.depth());

	//inspection de qqs pixels
	for (int i = 98 ; i<103 ; i++) {
		for ( int j=98 ; j < 103 ; j++) {
			cout << "pixel : " << i << " , " << j << " " << laplace.at<float>(i,j) << endl;
		}
	}

	/*
	namedWindow("laplace", CV_WINDOW_AUTOSIZE);
	imshow("laplace", laplace);

	
	if (imwrite(outDir + "laplace.pgm",laplace)) {
		cout << "enregistre" << endl;
	}
	*/

	waitKey(0);


	/*
	
	//ligne de front contenue dans le laplacien

	//binariser le laplacien


	//calcul des priorités

	//paramètres du patch
	float t_patch = 9.0;
	int half = t_patch/2;
	float invCard = 1.0/(t_patch*t_patch);


	for (int y=0 ; y < laplace.size().height ; y++) {
		for (int x=0 ; x < laplace.size().width ; x++) {

			//si ligne de front
			if (laplace.at<float>(y,x) != 0) {
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
				laplace.at<float>(y,x) = connus*invCard;

			}

		}
	}
	//fin du calcul des priorités initiales

	cout << laplace.at<float>(101,150) << endl;

	namedWindow("laplace", CV_WINDOW_AUTOSIZE);
	imshow("laplace", laplace);
	
	waitKey(0);
	
	*/

	/*
	namedWindow("lena_trou", CV_WINDOW_AUTOSIZE);
	imshow("lena_trou", img);
	waitKey(0);
	*/
	
	return 0;
}