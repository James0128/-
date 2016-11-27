
#include "opencv2/opencv.hpp"
#include"highgui.h"

#include "iostream"
using namespace std;
using namespace cv;
void FeatureMatch(Mat FrameROI1,Mat FrameROI2,vector <Point2f> &kp1,vector <Point2f> &kp2,int Flag,float ratio)
{
    Mat des1;
    Mat des2;
    vector <KeyPoint> keypoint1;
    vector <KeyPoint> keypoint2;
    
    
    Ptr <FeatureDetector> dector=ORB::create(500, 1.2f, 8, 31, 0, 2, ORB::FAST_SCORE, 31, 30);
    dector->detect(FrameROI1, keypoint1,Mat());
    dector->detect(FrameROI2, keypoint2,Mat());
    
    
    //Ptr <FeatureDetector> dector2 =ORB::create(50, 1.2f, 1, 31, 0, 2, ORB::FAST_SCORE, 31, 30);
    if (Flag==0)
    {
        Ptr<DescriptorExtractor> pd0=ORB::create(500, 1.2f, 8, 31, 0, 2, ORB::FAST_SCORE, 31, 30);
        pd0->compute(FrameROI1, keypoint1, des1);
        pd0->compute(FrameROI2, keypoint2, des2);
    }
    if (Flag==1)
    {
        Ptr<DescriptorExtractor> pd1=BRISK::create();
        pd1->compute(FrameROI1, keypoint1, des1);
        pd1->compute(FrameROI2, keypoint2, des2);
    }
    
    
    if ((keypoint1.size()>1)&&(keypoint2.size()>1))
    {
        
        vector <vector<DMatch> > Vmatch;
        Ptr <DescriptorMatcher> matcher = DescriptorMatcher::create("BruteForce-Hamming");
        matcher->knnMatch(des1, des2, Vmatch, 2);
        
        vector <DMatch> KNmatch;
        for (int i = 0; i < Vmatch.size(); i++)
        {
            double rat;
            if ( Vmatch.at(i).at(1).distance<0.05) {
                rat=1;
            }
            else {
                rat = Vmatch.at(i).at(0).distance / Vmatch.at(i).at(1).distance;
            }
            
            if (rat < ratio) {
                KNmatch.push_back(Vmatch.at(i).at(0));
            }
            
        }
        
        kp1.clear();
        kp2.clear();
        
        if (KNmatch.size()>0)
        {
            for(int i=0;i<KNmatch.size();i++)
            {
                int quaryx=KNmatch.at(i).queryIdx;
                int trainx=KNmatch.at(i).trainIdx;
                float mx=keypoint1.at(quaryx).pt.x;
                float my=keypoint1.at(quaryx).pt.y;
                float kx=keypoint2.at(trainx).pt.x;
                float ky=keypoint2.at(trainx).pt.y;
                Point2f p1;
                Point2f p2;
                p1.x=mx;
                p1.y=my;
                p2.x=kx;
                p2.y=ky;
                
                if (abs(ky-my)<5)
                {
                    kp1.push_back(p1);
                    kp2.push_back(p2);
                }
                
            }
            
        }
        
    }
    else
    {
        kp1.clear();
        kp2.clear();
    }
    
}
void DrawFeature(Mat &rematch,Mat Frame1,Mat Frame2,vector <Point2f> &kp1,vector <Point2f> &kp2)
{
    rematch=Mat((Frame1.rows>=Frame2.rows)?Frame1.rows:Frame2.rows,Frame1.cols+Frame2.cols,CV_8UC3);
    Rect rectleft(0,0,Frame1.cols,Frame1.rows);
    Rect rectright(Frame1.cols,0,Frame2.cols,Frame2.rows);
    Frame1.copyTo(rematch(rectleft));
    Frame2.copyTo(rematch(rectright));
    
    for (int i=0;i<kp1.size();i++)
    {
        line(rematch,cvPoint(kp1[i].x,kp1[i].y),cvPoint(Frame1.cols+kp2[i].x,kp2[i].y),CV_RGB(255,0,0),2);
    }
    
}
int main()
{
    Mat img=imread("/Users/yifanyang/Documents/opencv3/321.png",1);
    resize(img, img, cvSize(640,480));
    Mat img_gray;
    cvtColor(img, img_gray,CV_RGB2GRAY );
    VideoCapture cap(0);
    while (true) {
        Mat frame;
        cap>>frame;
        Mat frame_gray;
        cvtColor(img, img_gray,CV_RGB2GRAY );
        vector<Point2f> kp1;
        vector<Point2f> kp2;
        FeatureMatch(img_gray, frame_gray, kp1, kp2, 0,0.8);
        Mat rematch;
        DrawFeature(rematch,img, frame,  kp1, kp2);
        imshow("asd", img);
        waitKey(10);
    }
    
    
}
