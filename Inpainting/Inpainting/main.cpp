#include <iostream>
#include <string>
#include <vector>
#include <cmath>
#include <opencv2\core\core.hpp>
#include <opencv2\highgui\highgui.hpp>
#include <opencv2\imgproc\imgproc.hpp>


using namespace std;
using namespace cv;

//template latex icip 2015


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

//verifier que le patch ne comporte pas d'inconnu
bool inspectPatch(Mat mask, const int y_sg, const int x_sg, const int t_patch)
{
	for (int j = y_sg ; j < y_sg + t_patch ; j++)
	{
		for (int i = x_sg ; i < x_sg + t_patch ; i++)
		{
			if (mask.at<uchar>(j,i) < (unsigned char)255)
			{
				return false;
			}
		}
	}
	//pas d'inconnu
	return true;
}


//propagation de la priorite
void propagPrior(Mat priorites, Mat mask, const int y_g, const int x_h, const int t_patch, const float prior)
{
	//facteur de propagation
	float cst = 0.5;

	//calculer les abscisses et ordonnees du tour du patch
	int y_d = y_g + t_patch ;
	int x_b = x_h + t_patch ;

	//scan du patch
	for (int j=y_g ; j < y_d+1 ; j++)
	{
		for (int i=x_h ; i < x_b+1 ; i++)
		{
			//action pour tour du patch
			if (j == y_g || j == y_d || i == x_h || i == x_b)
			{
				//chercher si le pixel est sur la ligne de front => priorite non nulle
				//si oui, propager la priorite
				if (priorites.at<float>(j,i) != 0.0)
				{
					//jonction avec la nouvelle ligne de front
					priorites.at<float>(j,i) = prior * cst; //100.0; //;
				}
				//sinon, soit pixel inconnu => propager la priorite
				//		 soit pixel connu => ne rien faire
				else if (mask.at<uchar>(j,i) < (unsigned char) 255)
				{
					priorites.at<float>(j,i) = prior * cst; //100.0;
				}
			}
			//reste du patch, priorite nulle
			else
			{
				priorites.at<float>(j,i) = 0.0;
			}
		}
	}
}


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
	//namedWindow("lena", CV_WINDOW_AUTOSIZE);
	//imshow("lena", img);

	//waitKey(0);

	//cr�er masque binaire
	Mat mask(img.size(), CV_8UC1, 255);

	//d�finir une roi rectangulaire
	Mat roi(mask, Rect(100,100,50,50));
	Mat roiImg(img, Rect(100,100,50,50));

	//appliquer roi � l'image, masque nul
	roi = Scalar(0);
	roiImg = Scalar(0);


	//afficher l'image avec trou
	namedWindow("lenamasque", CV_WINDOW_AUTOSIZE);
	imshow("lenamasque", img);

	waitKey(0);

	//afficher le masque
	//namedWindow("masque", CV_WINDOW_AUTOSIZE);
	//imshow("masque", mask);

	//waitKey(0);

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

	//waitKey(0);




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
	int t_patch = 9;
	int half = t_patch/2;
	float invCard = 1.0/(float)(t_patch*t_patch);


	//retenir la derni�re plus haute priorit�
	int y_prior = 0;
	int x_prior = 0;
	float prior = 0.0;


	for (int y=0 ; y < laplace.size().height ; y++) {
		for (int x=0 ; x < laplace.size().width ; x++) {

			//si ligne de front
			if (laplace.at<float>(y,x) >= 255.0) {
				//compter connus et inconnus, appliquer patch

				int connus = 0;
				for (int i=0 ; i<t_patch ; i++)
				{
					for (int j=0 ; j<t_patch ; j++)
					{

						if (mask.at<uchar>(y-half+i,x-half+j) > 0)
						{
							//connu
							connus++;
						}
					}
				}				
				//fin de compte
				//mettre � jour priorit�
				float loc_prior = ((float)connus) * invCard;
				//float loc_prior = 100.0;
				laplace.at<float>(y,x) = loc_prior;

				//retenir la plus haute priorit�
				//if (loc_prior >= prior) {
				if (loc_prior > prior)
				{
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

	//namedWindow("laplace", CV_WINDOW_AUTOSIZE);
	//imshow("laplace", laplace);

	//waitKey(0);

	/*--------------------------------------
	Iterations du remplissage
	---------------------------------------*/

	cout << "premiere priorite : " << prior << endl;
	cout << "y_prior : "<< y_prior << endl;
	cout << "x_prior : "<< x_prior << endl;

	//remplissage sur la plus haute priorite

	//template matching

	//parametres
	t_patch = 9.0;
	half = t_patch/2;


	//nombre d'iterations
	int it = 0;
	int goal = 2;

	while (it < goal)
	{
		it++;

		//identification des pixels dans le patch autour de la priorit�
		vector<Point> indices_connus, indices_inconnus;
		vector<Vec3b> valeurs_connus;

		//coin superieur gauche du patch
		int y_prior_sg = y_prior-half;
		int x_prior_sg = x_prior-half;

		for (int j=y_prior_sg ; j < y_prior_sg+t_patch ; j++) {
			for (int i = x_prior_sg ; i < x_prior_sg+t_patch ; i++) {

				int j_index = j-(y_prior_sg);
					int i_index = i-(x_prior_sg);
					Point p(j_index,i_index);

				if(mask.at<uchar>(j,i) > (unsigned char)0)
				{
					//connu
					//recuperation de la valeur du pixel
					valeurs_connus.push_back(img.at<Vec3b>(j,i));
					//changement de repere du pixel -> coin superieur gauche indexe a (0,0)
					indices_connus.push_back(p);
				}
				else
				{
					//inconnu
					indices_inconnus.push_back(p);
				}
			}
		}

		for (int i = 0 ; i < indices_inconnus.size() ; i++)
		{
			std::cout << "pixel inconnu : "<< indices_inconnus[i].y << " , " << indices_inconnus[i].x << std::endl;
		}

		/*--------------------------------------
		Recherche du plus proche voisin
		---------------------------------------*/

		//scan de l'image pour plus proche voisin
		float score_nn = pow(pow(255.0,3.0),2.0);	//score de msd courant du plus proche voisin
		int y_nn = 0;	//indices du
		int x_nn = 0;	//plus proche voisin

		for (int y_patch = 0 ; y_patch < height-t_patch ; y_patch++)
		{
			for (int x_patch = 0 ; x_patch < width-t_patch ; x_patch++)
			{

				//inspecter si pas de pixel inconnu
				if (inspectPatch(mask, y_patch, x_patch, t_patch))
				{
					float msd = 0.0;	//mean square difference au patch courant

					//matching sur les pixels connus
					for (unsigned int k = 0 ; k < indices_connus.size() ; k++)
					{
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
			}
		} //fin du scan pour plus proche voisin


		/*--------------------------------------
		Application du plus proche voisin a la
		zone prioritaire
		---------------------------------------*/

		//on a y_nn et x_nn, indices du coin superieur gauche du plus proche voisin
		//recuperation des pixels nouveaux
		cout << "inconnus size : "<< indices_inconnus.size() << endl;
		for (unsigned int k = 0 ; k < indices_inconnus.size() ; k++)
		{
			Point courant = indices_inconnus[k];
			//indices
			int y_ref = courant.y + y_nn;
			int x_ref = courant.x + x_nn;

			int y_inc = courant.y + y_prior_sg;
			int x_inc = courant.x + x_prior_sg;

			//remplissage
			img.at<Vec3b>(y_inc,x_inc) = img.at<Vec3b>(y_ref,x_ref);

			//retrait des inconnus
			mask.at<uchar>(y_inc,x_inc) = (unsigned char)255;

			
		}

		namedWindow("masque", CV_WINDOW_AUTOSIZE);
			imshow("mask", mask);
		//	waitKey(0);

		/*--------------------------------------
		Propagation de la priorite
		---------------------------------------*/
		propagPrior(laplace, mask, y_prior_sg, x_prior_sg, t_patch, prior);

		namedWindow("prio", CV_WINDOW_AUTOSIZE);
	imshow("prio", laplace);
//	waitKey(0);

		/*--------------------------------------
		Recherche du nouveau prioritaire
		---------------------------------------*/
		double minVal;
		double maxVal;
		Point minLoc;
		Point maxLoc;

		minMaxLoc(laplace, &minVal, &maxVal, &minLoc, &maxLoc);

		y_prior = maxLoc.y;
		x_prior = maxLoc.x;
		prior = maxVal;

		cout << maxVal << endl;
		cout << "y_prior : "<< y_prior << endl;
		cout << "x_prior : "<< x_prior << endl;

	}



	namedWindow("lena_trou", CV_WINDOW_AUTOSIZE);
	imshow("lena_trou", img);
	waitKey(0);

	

	std::cout << "fin" << std::endl;

	waitKey(0);

	return 0;
}
