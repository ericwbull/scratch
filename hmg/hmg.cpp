#include <iostream>
#include <fstream>
#include "opencv2/stitching/stitcher.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/xfeatures2d.hpp"
#include "opencv2/calib3d.hpp"


#include "opencv2/highgui.hpp"
//#include "opencv2/opencv.hpp"
using namespace std;
using namespace cv;
bool try_use_gpu = false;
//Stitcher::Mode mode = Stitcher::PANORAMA;
vector<Mat> imgs;
string result_name = "result.jpg";
void printUsage();
int parseCmdArgs(int argc, char** argv);

void LogPoints(std::vector<KeyPoint>& pts)
{
	ofstream fs("log.txt",ios::app);
	int count = 0;
	fs << "count=" << pts.size() << endl;
	for (auto pt : pts)
	{
		fs << " " << count << " x= " << pt.pt.x << " y=" << pt.pt.y << " angle=" << pt.angle << " octave=" << pt.octave << " response=" << pt.response << " size=" << pt.size << endl;
		count++;
	}
}

void LogMatches(std::vector<DMatch> ms)
{
	ofstream fs("log.txt",ios::app);
	int count = 0;
	fs << "count=" << ms.size() << endl;
	for (auto m : ms)
	{
		fs << " " << count << " distance= " << m.distance << " imgIdx=" << m.imgIdx << " queryIdx=" << m.queryIdx << " trainIdx=" << m.trainIdx << endl;
		count++;
	}
}

void DrawMatch(Point2f qorg, Point2f torg, const Scalar& color, Mat& mat, const DMatch& m, const std::vector<KeyPoint>& qpts, const std::vector<KeyPoint>& tpts)
{
	Point2f pq = qorg + qpts[m.queryIdx].pt;
	Point2f tq = torg + tpts[m.trainIdx].pt;
	line(mat, pq,tq, color);
	
}

void DetectKeypoints(std::string name, Feature2D& f2d, Mat& mat, std::vector<KeyPoint>& pts)
{
f2d.detect( mat, pts );
    Mat keyMat;
    drawKeypoints(mat, pts, keyMat);
    imshow(name,keyMat);
    LogPoints(pts);
}

int main(int argc, char* argv[])
{
    int retval = parseCmdArgs(argc, argv);
    if (retval) return -1;
    Mat pano(700,900, CV_8UC3);
	int offsetx = 100;
	int offsety = 70;
	Mat trans1_mat = (Mat_<double>(2,3) << 1, 0, offsetx, 0, 1, offsety);
	Mat trans2_mat = (Mat_<double>(2,3) << 1, 0, offsetx, 0, 1, offsety+250);
	
//    Stitcher stitcher = Stitcher::createDefault(try_use_gpu);
//    Stitcher::Status status = stitcher.stitch(imgs, pano);
//    if (status != Stitcher::OK)
 //   {
  //      cout << "Can't stitch images, error code = " << int(status) << endl;
  //      return -1;
  //  }

		Mat& im_1 = imgs[0];
	Mat& im_2 = imgs[1];

	
	cv::Ptr<Feature2D> f2d = xfeatures2d::SIFT::create();

	
// Step 1: Detect the keypoints:
std::vector<KeyPoint> keypoints_1, keypoints_2;

	DetectKeypoints("im1", *f2d, im_1, keypoints_1);
	DetectKeypoints("im2", *f2d, im_2, keypoints_2);

// Step 2: Calculate descriptors (feature vectors)
Mat descriptors_1, descriptors_2;
f2d->compute( im_1, keypoints_1, descriptors_1 );

f2d->compute( im_2, keypoints_2, descriptors_2 );

	//imshow("desc1", descriptors_1);
	
// Step 3: Matching descriptor vectors using BFMatcher :
BFMatcher matcher;
std::vector< DMatch > matches;
matcher.match( descriptors_1, descriptors_2, matches );

// Keep 200 best matches only.
// We sort distance between descriptor matches
Mat index;
int nbMatch = int(matches.size());
Mat tab(nbMatch, 1, CV_32F);
for (int i = 0; i < nbMatch; i++)
	tab.at<float>(i, 0) = matches[i].distance;
sortIdx(tab, index, SORT_EVERY_COLUMN + SORT_ASCENDING);
vector<DMatch> bestMatches;

		im_1.copyTo(pano(cv::Rect(100,70,320,240)));
	im_2.copyTo(pano(cv::Rect(100,320,320,240)));

for (int i = 0; i < 200; i++)
{
	bestMatches.push_back(matches[index.at < int > (i, 0)]);
    DrawMatch(Point2f(100,70),Point2f(100,320), Scalar(i), pano,bestMatches.back(), keypoints_1, keypoints_2);
}
	LogMatches(bestMatches);
	//	warpAffine(im_1, pano, trans1_mat, Size(3 * im_1.cols, 3*im_1.rows)); 
	//	warpAffine(im_2, pano, trans2_mat, Size(3 * im_1.cols, 3*im_1.rows));
	
	imshow("pano",pano);
    imwrite(result_name, pano);

	// 1st image is the destination image and the 2nd image is the src image
std::vector<Point2f> dst_pts;                   //1st
std::vector<Point2f> source_pts;                //2nd

for (vector<DMatch>::iterator it = bestMatches.begin(); it != bestMatches.end(); ++it) {
	//-- Get the keypoints from the good matches
	dst_pts.push_back( keypoints_1[ it->queryIdx ].pt );
	source_pts.push_back( keypoints_2[ it->trainIdx ].pt );
}

Mat H = findHomography( source_pts, dst_pts, CV_RANSAC );
cout << H << endl;

	Mat wim_2;
warpPerspective(im_2, wim_2, H, pano.size());
imshow("wim2",wim_2);
	
waitKey();
    return 0;
}
void printUsage()
{
    cout <<
        "Images stitcher.\n\n"
        "stitching img1 img2 [...imgN]\n\n"
        "Flags:\n"
        "  --try_use_gpu (yes|no)\n"
        "      Try to use GPU. The default value is 'no'. All default values\n"
        "      are for CPU mode.\n"
        "  --mode (panorama|scans)\n"
        "      Determines configuration of stitcher. The default is 'panorama',\n"
        "      mode suitable for creating photo panoramas. Option 'scans' is suitable\n"
        "      for stitching materials under affine transformation, such as scans.\n"
        "  --output <result_img>\n"
        "      The default is 'result.jpg'.\n";
}
int parseCmdArgs(int argc, char** argv)
{
    if (argc == 1)
    {
        printUsage();
        return -1;
    }
    for (int i = 1; i < argc; ++i)
    {
        if (string(argv[i]) == "--help" || string(argv[i]) == "/?")
        {
            printUsage();
            return -1;
        }
        else if (string(argv[i]) == "--try_use_gpu")
        {
            if (string(argv[i + 1]) == "no")
                try_use_gpu = false;
            else if (string(argv[i + 1]) == "yes")
                try_use_gpu = true;
            else
            {
                cout << "Bad --try_use_gpu flag value\n";
                return -1;
            }
            i++;
        }
        else if (string(argv[i]) == "--output")
        {
            result_name = argv[i + 1];
            i++;
        }
        else if (string(argv[i]) == "--mode")
        {
            if (string(argv[i + 1]) == "panorama")
			{
            //    mode = Stitcher::PANORAMA;
			}
            else if (string(argv[i + 1]) == "scans")
			{
             //   mode = Stitcher::SCANS;
			}
            else
            {
                cout << "Bad --mode flag value\n";
                return -1;
            }
            i++;
        }
        else
        {
            Mat img = imread(argv[i]);
            if (img.empty())
            {
                cout << "Can't read image '" << argv[i] << "'\n";
                return -1;
            }
            imgs.push_back(img);
        }
    }
    return 0;
}
