#include <iostream>
#include <string>
#include <vector>
#include <cmath>
#include <algorithm>
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
bool inspectPatch(Mat mask, const int i_sg, const int j_sg, const int t_patch)
{
	for (int i = i_sg ; i < i_sg + t_patch ; i++)
	{
		for (int j = j_sg ; j < j_sg + t_patch ; j++)
		{
			if (mask.at<uchar>(i,j) < (unsigned char)255)
			{
				return false;
			}
		}
	}
	//pas d'inconnu
	return true;
}


//propagation de la priorite
void propagPrior(Mat priorites, Mat mask, const int i_sg, const int j_sg, const int t_patch, const float prior)
{
	//facteur de propagation
	float cst = 0.4;

	//calculer les abscisses et ordonnees du tour du patch
	int i_h = i_sg-1;
	int j_g = j_sg-1;
	int i_b = i_sg + t_patch ;
	int j_d = j_sg + t_patch ;

	//scan du tour et de l'interieur du patch
	for (int i=i_h ; i < i_b+1 ; i++)
	{
		for (int j=j_g ; j < j_d+1 ; j++)
		{
			//action pour tour du patch
			if (i == i_h || i == i_b || j == j_g || j == j_d)
			{
				//chercher si le pixel est sur la ligne de front => priorite non nulle
				//si oui, propager la priorite
				if (priorites.at<float>(i,j) != 0.0)
				{
					//jonction avec la nouvelle ligne de front
					priorites.at<float>(i,j) = prior * cst; //100.0; //;
				}
				//sinon, soit pixel inconnu => propager la priorite
				//		 soit pixel connu => ne rien faire
				else if (mask.at<uchar>(i,j) < (unsigned char) 255)
				{
					priorites.at<float>(i,j) = prior * cst; //100.0;
				}
			}
			//reste du patch, priorite nulle
			else
			{
				priorites.at<float>(i,j) = 0.0;
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
	Mat roi(mask, Rect(200,200,100,50));
	Mat roiImg(img, Rect(200,200,100,50));

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
	for (int i = 148 ; i<153 ; i++) {
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
	for (int i = 148 ; i<153 ; i++) {
		for ( int j=98 ; j < 103 ; j++) {
			cout << "pixel : " << i << " , " << j << " " << laplace.at<float>(i,j) << endl;
		}
	}



	//calcul des priorit�s

	//param�tres du patch
	int t_patch = 5;
	int half = t_patch/2;
	float invCard = 1.0/(float)(t_patch*t_patch);


	//retenir la derni�re plus haute priorit�
	int i_prior = 0;
	int j_prior = 0;
	float prior = 0.0;


	for (int i=0 ; i < laplace.size().height ; i++) {
		for (int j=0 ; j < laplace.size().width ; j++) {

			//si ligne de front
			if (laplace.at<float>(i,j) >= 255.0) {
				//compter connus et inconnus, appliquer patch

				int connus = 0;
				for (int k=0 ; k<t_patch ; k++)
				{
					for (int l=0 ; l<t_patch ; l++)
					{

						if (mask.at<uchar>(i-half+k,j-half+l) > 0)
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
				laplace.at<float>(i,j) = loc_prior;

				//retenir la plus haute priorit�
				//if (loc_prior >= prior) {
				if (loc_prior > prior)
				{
					prior = loc_prior;
					i_prior = i;
					j_prior = j;
				}
			}
		}
	}
	//fin du calcul des priorit�s initiales


	std::cout << "valeur en 150,120 : " << laplace.at<float>(200,120) << std::endl;
	std::cout << "meme valeur par Point : " << laplace.at<float>(Point(200,120)) << std::endl;

	//inspection de qqs pixels
	for (int i = 148 ; i<153 ; i++) {
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
	cout << "i_prior : "<< i_prior << endl;
	cout << "j_prior : "<< j_prior << endl;

	//remplissage sur la plus haute priorite

	//template matching

	//parametres
	t_patch = 9.0;
	half = t_patch/2;


	//nombre d'iterations
	//int it = 0;
	//int goal = 92;

	bool suite = true;

	//while (it < goal)
	while (suite)
	{
		//it++;

		//identification des pixels dans le patch autour de la priorit�
		vector<Point> indices_connus, indices_inconnus;
		vector<Vec3b> valeurs_connus;

		//coin superieur gauche du patch
		int i_prior_sg = i_prior-half;
		int j_prior_sg = j_prior-half;

		for (int i=i_prior_sg ; i < i_prior_sg+t_patch ; i++) {
			for (int j = j_prior_sg ; j < j_prior_sg+t_patch ; j++) {

				int i_index = i-(i_prior_sg);
					int j_index = j-(j_prior_sg);
					Point p(j_index,i_index);	//Point oriente suivant x,y

				if(mask.at<uchar>(i,j) > (unsigned char)0)
				{
					//connu
					//recuperation de la valeur du pixel
					valeurs_connus.push_back(img.at<Vec3b>(i,j));
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

		for (int i = 0 ; i < indices_connus.size() ; i++)
		{
			std::cout << "pixel connu : "<< indices_connus[i].y << " , " << indices_connus[i].x << std::endl;
		}

		/*--------------------------------------
		Recherche du plus proche voisin
		---------------------------------------*/

		//definition d'une fenetre de recherche autour du patch prioritaire
		int t_recherche = 20;
		int i_sg_recherche, j_sg_recherche, i_id_recherche, j_id_recherche;

		i_sg_recherche = std::max(0,i_prior_sg-t_recherche);
		j_sg_recherche = std::max(0,j_prior_sg-t_recherche);
		i_id_recherche = std::min(height-t_patch, i_prior_sg+t_recherche);
		j_id_recherche = std::min(width-t_patch, j_prior_sg+t_recherche);


		//scan de l'image pour plus proche voisin
		float score_nn = pow(pow(255.0,3.0),2.0);	//score de msd courant du plus proche voisin
		int i_nn = 0;	//indices du
		int j_nn = 0;	//plus proche voisin

		//for (int i_patch = 0 ; i_patch < height-t_patch ; i_patch++)
		for (int i_patch = i_sg_recherche ; i_patch < i_id_recherche+1 ; i_patch++)
		{
			//for (int j_patch = 0 ; j_patch < width-t_patch ; j_patch++)
			for (int j_patch = j_sg_recherche ; j_patch < j_id_recherche+1 ; j_patch++)
			{

				//inspecter si pas de pixel inconnu
				if (inspectPatch(mask, i_patch, j_patch, t_patch))
				{
					float msd = 0.0;	//mean square difference au patch courant

					//matching sur les pixels connus
					for (unsigned int k = 0 ; k < indices_connus.size() ; k++)
					{
						//convertir pixel connu vers espace du patch courant
						Point p = indices_connus[k];
						int i_courant = p.y + i_patch;
						int j_courant = p.x + j_patch;

						//square difference
						Vec3f val_courant = normalizeRGB(img.at<Vec3b>(i_courant, j_courant));
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
						i_nn = i_patch;
						j_nn = j_patch;
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
			int i_ref = courant.y + i_nn;
			int j_ref = courant.x + j_nn;

			int i_inc = courant.y + i_prior_sg;
			int j_inc = courant.x + j_prior_sg;

			//remplissage
			img.at<Vec3b>(i_inc,j_inc) = img.at<Vec3b>(i_ref,j_ref);

			//retrait des inconnus
			mask.at<uchar>(i_inc,j_inc) = (unsigned char)255;

			
		}

		namedWindow("masque", CV_WINDOW_AUTOSIZE);
			imshow("mask", mask);
			waitKey(1);

		/*--------------------------------------
		Propagation de la priorite
		---------------------------------------*/
		propagPrior(laplace, mask, i_prior_sg, j_prior_sg, t_patch, prior);

		namedWindow("prio", CV_WINDOW_AUTOSIZE);
	imshow("prio", laplace);
	waitKey(1);

		/*--------------------------------------
		Recherche du nouveau prioritaire
		---------------------------------------*/
		double minVal;
		double maxVal;
		Point minLoc;
		Point maxLoc;

		minMaxLoc(laplace, &minVal, &maxVal, &minLoc, &maxLoc);

		i_prior = maxLoc.y;
		j_prior = maxLoc.x;
		prior = maxVal;

		if (prior == 0.0)
			suite = false;

		cout << maxVal << endl;
		cout << "i_prior : "<< i_prior << endl;
		cout << "j_prior : "<< j_prior << endl;

		namedWindow("lena_constr", CV_WINDOW_AUTOSIZE);
		imshow("lena_constr", img);
		waitKey(1);


	}



	namedWindow("lena_patch", CV_WINDOW_AUTOSIZE);
	imshow("lena_patch", img);
	waitKey(0);

	

	std::cout << "fin" << std::endl;

	waitKey(0);

	return 0;
}
