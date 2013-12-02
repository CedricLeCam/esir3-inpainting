#include <iostream>
#include <string>
#include <vector>
#include <opencv2\core\core.hpp>
#include <opencv2\highgui\highgui.hpp>
#include <opencv2\imgproc\imgproc.hpp>

using namespace std;
using namespace cv;

int main(int argc, char ** argv) {

	cout << "start" << endl;

	string dataDir("../data/");	//chemin vers les donn�es
	string outDir(dataDir + "output/");	//chemin du repertoire de sortie

	//ouvrir une image
	string imgFile("lena.ppm");

	Mat img = imread(dataDir + imgFile);

	if (!img.data) {
		cout << "echec" << endl;
		waitKey(0);
		return 0;
	}

	//proprietes pratiques
	int height = img.size().height;
	int width = img.size().width;

	//afficher l'image
	namedWindow("lena", CV_WINDOW_AUTOSIZE);
	imshow("lena", img);

	waitKey(0);
	
	//cr�er masque binaire
	Mat mask(img.size(), CV_8UC1, 255);

	//d�finir une roi rectangulaire
	Mat roi(mask, Rect(100,100,50,50));

	//appliquer roi � l'image, masque nul
	roi = Scalar(0);

	//afficher le masque
	namedWindow("masque", CV_WINDOW_AUTOSIZE);
	imshow("masque", mask);

	waitKey(0);

	/*--------------------------------------
	Priorit�s
	---------------------------------------*/
	
	//ligne de front

	//calculer le laplacien de mask, stock� dans une autre image
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


	
	
	//ligne de front contenue dans le laplacien

	//binariser le laplacien
	threshold(laplace, laplace, 1.0, 255.0, THRESH_BINARY);

	//inspection de qqs pixels
	for (int i = 98 ; i<103 ; i++) {
		for ( int j=98 ; j < 103 ; j++) {
			cout << "pixel : " << i << " , " << j << " " << laplace.at<float>(i,j) << endl;
		}
	}


	
	//calcul des priorit�s

	//param�tres du patch
	float t_patch = 9.0;
	int half = t_patch/2;
	float invCard = 1.0/(t_patch*t_patch);


	//retenir la derni�re plus haute priorit�
	int y_prior = 0;
	int x_prior = 0;
	float prior = 0.0;


	for (int y=0 ; y < laplace.size().height ; y++) {
		for (int x=0 ; x < laplace.size().width ; x++) {

			//si ligne de front
			if (laplace.at<float>(y,x) == 255.0) {
				//compter connus et inconnus, appliquer patch
				
				int connus = 0;
				for (int i=0 ; i<t_patch ; i++) {
					for (int j=0 ; j<t_patch ; j++) {


						if (mask.at<uchar>(y-half+i,x-half+j) > 0) {
							//connu
							connus++;
						}
						
					}
				}
				
				//fin de compte
				//mettre � jour priorit�
				float loc_prior = ((float)connus) * invCard;
				laplace.at<float>(y,x) = loc_prior;


				//retenir la plus haute priorit�
				if (loc_prior >= prior) {
					prior = loc_prior;
					y_prior = y;
					x_prior = x;
				}

			}

		}
	}
	//fin du calcul des priorit�s initiales

	//inspection de qqs pixels
	for (int i = 98 ; i<103 ; i++) {
		for ( int j=98 ; j < 103 ; j++) {
			cout << "pixel : " << i << " , " << j << " " << laplace.at<float>(i,j) << endl;
		}
	}

	namedWindow("laplace", CV_WINDOW_AUTOSIZE);
	imshow("laplace", laplace);
	
	waitKey(0);
	
	//remplissage sur la plus haute priorite

	//template matching

	//parametres
	float t_patch = 9.0;
	int half = t_patch/2;

	//r�cup�ration des pixels connus du patch autour de la priorit�
	vector<Point> connus;

	for (int j=y_prior-half ; j < y_prior+half+1 ; j++) {
		for (int i = x_prior-half ; i < x_prior+half+1 ; i++) {
			
			if(mask.at<uchar>(j,i) > 0) {
				//connu
				Point p(j,i);
				connus.push_back(p);
			}
		}
	}


	//scan de l'image
	for (int y = half ; y < height-half ; y++) {
		for (int x = half ; x < width-half ; x++) {

			//matching sur les pixels connus


	/*
	namedWindow("lena_trou", CV_WINDOW_AUTOSIZE);
	imshow("lena_trou", img);
	waitKey(0);
	*/
	
	return 0;
}