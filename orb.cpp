#include <opencv2/opencv.hpp>
#include <opencv2/nonfree/nonfree.hpp>


using namespace cv;
using namespace std;

int main()
{
    Mat img1 = imread("bo.jpg", IMREAD_GRAYSCALE);
    Mat img2 = imread("com.jpg", IMREAD_GRAYSCALE);
   
    if(img1.empty() || img2.empty())
        return -1;
       
       
    vector<KeyPoint> key1, key2;
    Mat des1, des2;
  
    ORB(1000)(img1, noArray(), key1, des1);
    ORB(1000)(img2, noArray(), key2, des2);
  
    cout << "key1.size()=" << key1.size() << endl;
    cout << "key2.size()=" << key2.size() << endl;
   
    vector< vector< DMatch > > matches;
    BFMatcher matcher(NORM_HAMMING);
   
    int k=2;
    matcher.knnMatch(des1, des2, matches, k);
    
    vector< DMatch > good;
    float nndrRatio = 0.6f;
    for(int i=0; i<matches.size(); i++)
    {
        cout << "matches.size() = " << matches.size() << endl;
        if(matches.at(i).size() == 2 && matches.at(i).at(0).distance <= nndrRatio * matches.at(i).at(1).distance)
        {
            good.push_back(matches[i][0]);
        }
       
    }
    cout << "good.size() = " << good.size() << endl;
    
   
    Mat imgMatches;
    drawMatches(img1, key1, img2, key2, good, imgMatches, Scalar::all(-1), Scalar::all(-1), vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
       
    vector<Point2f> obj;
    vector<Point2f> scene;
    for(int i=0; i<good.size(); i++)
    {
        obj.push_back(key1[good[i].queryIdx].pt);
        scene.push_back(key2[good[i].trainIdx].pt);
    }
    Mat H = findHomography(obj, scene, RANSAC);
   
    vector<Point2f> objP(4);
    objP[0] = Point2f(0,0);
    objP[1] = Point2f(img2.cols,0);
    objP[2] = Point2f(img2.cols, img2.rows);
    objP[3] = Point2f(0,img2.rows);
   
    vector<Point2f> sceneP(4);
    perspectiveTransform(objP, sceneP, H);
   
    for(int i=0; i<4; i++)
        sceneP[i] += Point2f(img1.cols, 0);
    for(int i=0; i<4; i++)
        line(imgMatches, sceneP[i], sceneP[(i+1)%4], Scalar(255, 0, 0), 4);
        
    imshow("imgMatches", imgMatches);
   
    waitKey();
    return 0;
}
