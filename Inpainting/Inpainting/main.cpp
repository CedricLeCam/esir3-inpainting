#include <iostream>
#include <string>
#include <vector>
#include <cmath>
#include <opencv2\core\core.hpp>
#include <opencv2\highgui\highgui.hpp>
#include <opencv2\imgproc\imgproc.hpp>


using namespace std;
using namespace cv;


//transition de RGB vers RGB normalise
Vec3f normalizeRGB(const Vec3b & pixel)
{
	int b = pixel.val[0];
	int g = pixel.val[1];
	int r = pixel.val[2];
	float n = 1.0/(float)(b+g+r);
	Vec3f res((float)b*n, (float)g*n, (float)r*n);
	return res;
}

//norme de la difference entre deux Vec3f ||v1 - v2||
float diff(const Vec3f & v1, const Vec3f & v2)
{
	return norm(v1-v2);
}

//squared difference
float squareDiff(const Vec3f & v1, const Vec3f & v2)
{
	float d = diff(v1,v2);
	return d*d;
}


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

	//proprietes pratiques
	int height = img.size().height;
	int width = img.size().width;

	//afficher l'image
	namedWindow("lena", CV_WINDOW_AUTOSIZE);
	imshow("lena", img);

	waitKey(0);
	
	//créer masque binaire
	Mat mask(img.size(), CV_8UC1, 255);

	//définir une roi rectangulaire
	Mat roi(mask, Rect(100,100,50,50));
	Mat roiImg(img, Rect(100,100,50,50));

	//appliquer roi à l'image, masque nul
	roi = Scalar(0);
	roiImg = Scalar(0);

	//afficher le masque
	namedWindow("masque", CV_WINDOW_AUTOSIZE);
	imshow("masque", mask);

	waitKey(0);

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


	
	
	//ligne de front contenue dans le laplacien

	//binariser le laplacien
	threshold(laplace, laplace, 1.0, 255.0, THRESH_BINARY);

	//inspection de qqs pixels
	for (int i = 98 ; i<103 ; i++) {
		for ( int j=98 ; j < 103 ; j++) {
			cout << "pixel : " << i << " , " << j << " " << laplace.at<float>(i,j) << endl;
		}
	}


	
	//calcul des priorités

	//paramètres du patch
	float t_patch = 9.0;
	int half = t_patch/2;
	float invCard = 1.0/(t_patch*t_patch);


	//retenir la dernière plus haute priorité
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
				//mettre à jour priorité
				float loc_prior = ((float)connus) * invCard;
				laplace.at<float>(y,x) = loc_prior;


				//retenir la plus haute priorité
				if (loc_prior >= prior) {
					prior = loc_prior;
					y_prior = y;
					x_prior = x;
				}

			}

		}
	}
	//fin du calcul des priorités initiales

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
	t_patch = 9.0;
	half = t_patch/2;

	//identification des pixels dans le patch autour de la priorité
	vector<Point> indices_connus, indices_inconnus;
	vector<Vec3b> valeurs_connus;
	
	//coin superieur gauche du patch
	int y_prior_sg = y_prior-half;
	int x_prior_sg = x_prior-half;

	for (int j=y_prior_sg ; j < y_prior_sg+t_patch ; j++) {
		for (int i = x_prior_sg ; i < x_prior_sg+t_patch ; i++) {
			
			if(mask.at<uchar>(j,i) > 0) {
				//connu
				//recuperation de la valeur du pixel
				valeurs_connus.push_back(img.at<Vec3b>(j,i));
				//changement de repere du pixel -> coin superieur gauche indexe a (0,0)
				int j_index = j-(y_prior_sg);
				int i_index = i-(x_prior_sg);
				Point p(j_index,i_index);
				indices_connus.push_back(p);
			}
			else {
				//inconnu
				int j_index = j-(y_prior_sg);
				int i_index = i-(x_prior_sg);
				Point p(j_index,i_index);
				indices_inconnus.push_back(p);
			}
		}
	}

	/*--------------------------------------
	Recherche du plus proche voisin
	---------------------------------------*/

	//scan de l'image pour plus proche voisin
	float score_nn = pow(pow(255.0,3.0),2.0);	//score de msd courant du plus proche voisin
	int y_nn = 0;	//indices du
	int x_nn = 0;	//plus proche voisin

	for (int y_patch = 0 ; y_patch < height-t_patch ; y_patch++) {
		for (int x_patch = 0 ; x_patch < width-t_patch ; x_patch++) {

			float msd = 0.0;	//mean square difference au patch courant

			//matching sur les pixels connus
			for (int k = 0 ; k < indices_connus.size() ; k++) {

				//convertir pixel connu vers espace du patch courant
				Point p = indices_connus[k];
				int j_courant = p.y + y_patch;
				int i_courant = p.x + x_patch;

				//square difference
				Vec3f val_courant = normalizeRGB(img.at<Vec3b>(j_courant, i_courant));
				Vec3f val_connu = normalizeRGB(valeurs_connus[k]);
				float sqDiff = squareDiff(val_courant, val_connu);

				//ajout a la somme courante
				msd += sqDiff;
			}

			//fin du calcul de msd
			msd /= (float)indices_connus.size();

			//comparaison au score courant
			if (msd < score_nn)
			{
				score_nn = msd;
				y_nn = y_patch;
				x_nn = x_patch;
			}
		}
	} //fin du scan pour plus proche voisin
 

	/*--------------------------------------
	Application du plus proche voisin a la
	zone prioritaire
	---------------------------------------*/

	//on a y_nn et x_nn, indices du coin superieur gauche du plus proche voisin
	//recuperation des pixels nouveaux
	for (int k = 0 ; k < indices_inconnus.size() ; k++)
	{
		Point courant = indices_inconnus[k];
		//indices
		int y_ref = courant.y + y_nn;
		int x_ref = courant.x + x_nn;

		int y_inc = courant.y + y_prior_sg;
		int x_inc = courant.x + x_prior_sg;

		//remplissage
		img.at<Vec3b>(y_inc,x_inc) = img.at<Vec3b>(y_ref,x_ref);

		//propagation de la priorite
	}



	
	namedWindow("lena_trou", CV_WINDOW_AUTOSIZE);
	imshow("lena_trou", img);
	waitKey(0);
	
	
	return 0;
}