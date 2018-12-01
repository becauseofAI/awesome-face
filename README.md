# HelloFace
Face Technology Repository(**Updating**)

## Recent Update
###### 2018/12/01
- **AAM-Softmax(CCL)**: Face Recognition via Centralized Coordinate Learning
- **AM-Softmax**: Additive Margin Softmax for Face Verification
- **FeatureIncay**: Feature Incay for Representation Regularization
- **NormFace**: L2 hypersphere embedding for face Verification
- **CocoLoss**: Rethinking Feature Discrimination and Polymerization for Large-scale Recognition
- **L-Softmax**: Large-Margin Softmax Loss for Convolutional Neural Networks
###### 2018/07/21
- **MobileFace**: A face recognition solution on mobile device
- **Trillion Pairs**: Challenge 3: Face Feature Test/Trillion Pairs
- **MobileFaceNets**: Efficient CNNs for Accurate Real-Time Face Verification on Mobile Devices
###### 2018/04/20
- **PyramidBox**: A Context-assisted Single Shot Face Detector
- **PCN**: Real-Time Rotation-Invariant Face Detection with Progressive Calibration Networks
- **S³FD**: Single Shot Scale-invariant Face Detector
- **SSH**: Single Stage Headless Face Detector
- **NPD**: A Fast and Accurate Unconstrained Face Detector
- **PICO**: Object Detection with Pixel Intensity Comparisons Organized in Decision Trees
- **libfacedetection**: A fast binary library for face detection and face landmark detection in images.
- **SeetaFaceEngine**: SeetaFace Detection, SeetaFace Alignment and SeetaFace Identification.
- **FaceID**: An implementation of iPhone X's FaceID using face embeddings and siamese networks on RGBD images.
###### 2018/03/28
- **InsightFace(ArcFace)**: 2D and 3D Face Analysis Project
- **CosFace**: Large Margin Cosine Loss for Deep Face Recognition


## Face Benchmark and Dataset
#### Face Recognition
- **Trillion Pairs**: Challenge 3: Face Feature Test/Trillion Pairs(**MS-Celeb-1M-v1c with 86,876 ids/3,923,399 aligned images  + Asian-Celeb 93,979 ids/2,830,146 aligned images**) [[benckmark]](http://trillionpairs.deepglint.com/overview "DeepGlint") [[dataset]](http://trillionpairs.deepglint.com/data) [[result]](http://trillionpairs.deepglint.com/results)
- **MF2**: Level Playing Field for Million Scale Face Recognition(**672K people in 4.7M images**) [[paper]](https://homes.cs.washington.edu/~kemelmi/ms.pdf "CVPR2017") [[dataset]](http://megaface.cs.washington.edu/dataset/download_training.html) [[result]](http://megaface.cs.washington.edu/results/facescrub_challenge2.html) [[benckmark]](http://megaface.cs.washington.edu/)
- **MegaFace**: The MegaFace Benchmark: 1 Million Faces for Recognition at Scale(**690k people in 1M images**) [[paper]](http://megaface.cs.washington.edu/KemelmacherMegaFaceCVPR16.pdf "CVPR2016") [[dataset]](http://megaface.cs.washington.edu/participate/challenge.html) [[result]](http://megaface.cs.washington.edu/results/facescrub.html) [[benckmark]](http://megaface.cs.washington.edu/)
- **UMDFaces**: An Annotated Face Dataset for Training Deep Networks(**8k people in 367k images with pose, 21 key-points and gender**) [[paper]](https://arxiv.org/pdf/1611.01484.pdf "arXiv2016") [[dataset]](http://www.umdfaces.io/)
- **MS-Celeb-1M**: A Dataset and Benchmark for Large Scale Face Recognition(**100K people in 10M images**) [[paper]](https://arxiv.org/pdf/1607.08221.pdf "ECCV2016") [[dataset]](http://www.msceleb.org/download/sampleset) [[result]](http://www.msceleb.org/leaderboard/iccvworkshop-c1) [[benchmark]](http://www.msceleb.org/) [[project]](https://www.microsoft.com/en-us/research/project/ms-celeb-1m-challenge-recognizing-one-million-celebrities-real-world/)
- **VGGFace2**: A dataset for recognising faces across pose and age(**9k people in 3.3M images**) [[paper]](https://arxiv.org/pdf/1710.08092.pdf "arXiv2017") [[dataset]](http://www.robots.ox.ac.uk/~vgg/data/vgg_face2/)
- **VGGFace**: Deep Face Recognition(**2.6k people in 2.6M images**) [[paper]](http://www.robots.ox.ac.uk/~vgg/publications/2015/Parkhi15/parkhi15.pdf "BMVC2015") [[dataset]](http://www.robots.ox.ac.uk/~vgg/data/vgg_face/)
- **CASIA-WebFace**: Learning Face Representation from Scratch(**10k people in 500k images**) [[paper]](https://arxiv.org/pdf/1411.7923.pdf "arXiv2014") [[dataset]](http://www.cbsr.ia.ac.cn/english/CASIA-WebFace-Database.html)
- **LFW**: Labeled Faces in the Wild: A Database for Studying Face Recognition in Unconstrained Environments(**5.7k people in 13k images**) [[report]](http://vis-www.cs.umass.edu/lfw/lfw.pdf "UMASS2007") [[dataset]](http://vis-www.cs.umass.edu/lfw/#download) [[result]](http://vis-www.cs.umass.edu/lfw/results.html) [[benchmark]](http://vis-www.cs.umass.edu/lfw/)

#### Face Detection
- **WiderFace**: WIDER FACE: A Face Detection Benchmark(**400k people in 32k images with a high degree of variability in scale, pose and occlusion**) [[paper]](https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Yang_WIDER_FACE_A_CVPR_2016_paper.pdf "CVPR2016") [[dataset]](http://mmlab.ie.cuhk.edu.hk/projects/WIDERFace/) [[result]](http://mmlab.ie.cuhk.edu.hk/projects/WIDERFace/WiderFace_Results.html) [[benchmark]](http://mmlab.ie.cuhk.edu.hk/projects/WIDERFace/)
- **FDDB**: A Benchmark for Face Detection in Unconstrained Settings(**5k faces in 2.8k images**) [[report]](https://people.cs.umass.edu/~elm/papers/fddb.pdf "UMASS2010") [[dataset]](http://vis-www.cs.umass.edu/fddb/index.html#download) [[result]](http://vis-www.cs.umass.edu/fddb/results.html) [[benchmark]](http://vis-www.cs.umass.edu/fddb/) 

#### Face Landmark
- **AFLW**: Annotated Facial Landmarks in the Wild: A Large-scale, Real-world Database for Facial Landmark Localization(**25k faces with 21 landmarks**) [[paper]](https://files.icg.tugraz.at/seafhttp/files/460c7623-c919-4d35-b24e-6abaeacb6f31/koestinger_befit_11.pdf "BeFIT2011") [[benchmark]](https://www.tugraz.at/institute/icg/research/team-bischof/lrs/downloads/aflw/)

#### Face Attribute
- **CelebA**: Deep Learning Face Attributes in the Wild(**10k people in 202k images with 5 landmarks and 40 binary attributes per image**) [[paper]](https://www.cv-foundation.org/openaccess/content_iccv_2015/papers/Liu_Deep_Learning_Face_ICCV_2015_paper.pdf "ICCV2015") [[dataset]](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html)


## Face Recognition
- **MobileFace**: A face recognition solution on mobile device [[code]](https://github.com/becauseofAI/MobileFace)
- **MobileFaceNets**: Efficient CNNs for Accurate Real-Time Face Verification on Mobile Devices [[paper]](https://arxiv.org/abs/1804.07573 "arXiv2018") [[code1]](https://github.com/deepinsight/insightface "MXNet") [[code2]](https://github.com/KaleidoZhouYN/mobilefacenet-caffe "Caffe") [[code3]](https://github.com/xsr-ai/MobileFaceNet_TF "TensorFlow") [[code4]](https://github.com/GRAYKEY/mobilefacenet_ncnn "NCNN")
- **FaceID**: An implementation of iPhone X's FaceID using face embeddings and siamese networks on RGBD images. [[code]](https://github.com/normandipalo/faceID_beta "Keras") [[blog]](https://towardsdatascience.com/how-i-implemented-iphone-xs-faceid-using-deep-learning-in-python-d5dbaa128e1d "Medium") 
- **InsightFace(ArcFace)**: 2D and 3D Face Analysis Project [[paper]](https://arxiv.org/abs/1801.07698 "ArcFace: Additive Angular Margin Loss for Deep Face Recognition(arXiv)") [[code1]](https://github.com/deepinsight/insightface "MXNet") [[code2]](https://github.com/auroua/InsightFace_TF "TensorFlow")
- **AAM-Softmax(CCL)**: Face Recognition via Centralized Coordinate Learning [[paper]](https://arxiv.org/abs/1801.05678 "arXiv2018")
- **AM-Softmax**: Additive Margin Softmax for Face Verification [[paper]](https://arxiv.org/abs/1801.05599 "arXiv2018") [[code1]](https://github.com/happynear/AMSoftmax "Caffe") [[code2]](https://github.com/Joker316701882/Additive-Margin-Softmax "TensorFlow")
- **CosFace**: Large Margin Cosine Loss for Deep Face Recognition [[paper]](https://arxiv.org/abs/1801.09414 "CVPR2018") [[code1]](https://github.com/deepinsight/insightface "MXNet") [[code2]](https://github.com/yule-li/CosFace "TensorFlow")
- **FeatureIncay**: Feature Incay for Representation Regularization [[paper]](https://arxiv.org/abs/1705.10284 "ICLR2018")
- **CocoLoss**: Rethinking Feature Discrimination and Polymerization for Large-scale Recognition [[paper]](http://cn.arxiv.org/abs/1710.00870 "NIPS2017") [[code]](https://github.com/sciencefans/coco_loss "Caffe")
- **NormFace**: L2 hypersphere embedding for face Verification [[paper]](http://www.cs.jhu.edu/~alanlab/Pubs17/wang2017normface.pdf "ACM2017 Multimedia Conference") [[code]](https://github.com/happynear/NormFace "Caffe")
- **SphereFace(A-Softmax)**: Deep Hypersphere Embedding for Face Recognition [[paper]](http://openaccess.thecvf.com/content_cvpr_2017/papers/Liu_SphereFace_Deep_Hypersphere_CVPR_2017_paper.pdf "CVPR2017") [[code]](https://github.com/wy1iu/sphereface "Caffe")
- **L-Softmax**: Large-Margin Softmax Loss for Convolutional Neural Networks [[paper]](http://proceedings.mlr.press/v48/liud16.pdf "ICML2016") [[code1]](https://github.com/wy1iu/LargeMargin_Softmax_Loss "Caffe") [[code2]](https://github.com/luoyetx/mx-lsoftmax "MXNet") [[code3]](https://github.com/HiKapok/tf.extra_losses "TensorFlow") [[code4]](https://github.com/auroua/L_Softmax_TensorFlow "TensorFlow") [[code5]](https://github.com/tpys/face-recognition-caffe2 "Caffe2") [[code6]](https://github.com/amirhfarzaneh/lsoftmax-pytorch "PyTorch") [[code7]](https://github.com/jihunchoi/lsoftmax-pytorch "PyTorch")
- **CenterLoss**: A Discriminative Feature Learning Approach for Deep Face Recognition [[paper]](https://ydwen.github.io/papers/WenECCV16.pdf "ECCV2016") [[code1]](https://github.com/ydwen/caffe-face "Caffe") [[code2]](https://github.com/pangyupo/mxnet_center_loss "MXNet") [[code3]](https://github.com/ShownX/mxnet-center-loss "MXNet-Gluon") [[code4]](https://github.com/EncodeTS/TensorFlow_Center_Loss "TensorFlow")
- **OpenFace**: A general-purpose face recognition library with mobile applications [[report]](http://elijah.cs.cmu.edu/DOCS/CMU-CS-16-118.pdf "CMU2016") [[code]](https://github.com/cmusatyalab/openface "Torch") [[project]](http://cmusatyalab.github.io/openface/)
- **FaceNet**: A Unified Embedding for Face Recognition and Clustering [[paper]](https://www.cv-foundation.org/openaccess/content_cvpr_2015/papers/Schroff_FaceNet_A_Unified_2015_CVPR_paper.pdf "CVPR2015") [[code]](https://github.com/davidsandberg/facenet "TensorFlow")
- **DeepID3**: DeepID3: Face Recognition with Very Deep Neural Networks [[paper]](https://arxiv.org/abs/1502.00873 "arXiv2015") 
- **DeepID2+**: Deeply learned face representations are sparse, selective, and robust [[paper]](https://www.cv-foundation.org/openaccess/content_cvpr_2015/papers/Sun_Deeply_Learned_Face_2015_CVPR_paper.pdf "CVPR2015")
- **DeepID2**: Deep Learning Face Representation by Joint Identification-Verification [[paper]](https://papers.nips.cc/paper/5416-deep-learning-face-representation-by-joint-identification-verification.pdf "NIPS2014")
- **DeepID**: Deep Learning Face Representation from Predicting 10,000 Classes [[paper]](https://www.cv-foundation.org/openaccess/content_cvpr_2014/papers/Sun_Deep_Learning_Face_2014_CVPR_paper.pdf "CVPR2014")
- **DeepFace**: Closing the gap to human-level performance in face verification [[paper]](https://www.cv-foundation.org/openaccess/content_cvpr_2014/papers/Taigman_DeepFace_Closing_the_2014_CVPR_paper.pdf "CVPR2014")
- **LBPFace**: Face recognition with local binary patterns [[paper]](https://pdfs.semanticscholar.org/3242/0c65f8ef0c5bd83b14c8ae662cbce73e6781.pdf "ECCV2004") [[code]](https://docs.opencv.org/2.4/modules/contrib/doc/facerec/facerec_tutorial.html "OpenCV")
- **FisherFace(LDA)**: Eigenfaces vs. Fisherfaces: Recognition Using Class Specific Linear Projection [[paper]](https://apps.dtic.mil/dtic/tr/fulltext/u2/1015508.pdf "TPAMI1997") [[code]](https://docs.opencv.org/2.4/modules/contrib/doc/facerec/facerec_tutorial.html "OpenCV")
- **EigenFace(PCA)**: Face recognition using eigenfaces [[paper]](http://www.cs.ucsb.edu/~mturk/Papers/mturk-CVPR91.pdf "CVPR1991") [[code]](https://docs.opencv.org/2.4/modules/contrib/doc/facerec/facerec_tutorial.html "OpenCV")

## Face Detection
- **PyramidBox**: A Context-assisted Single Shot Face Detector [[paper]](https://arxiv.org/pdf/1803.07737.pdf "arXiv2018") [[code]](https://github.com/PaddlePaddle/models/tree/2a6b7dc92f04815f0b298e59030cb779dd0e038c/fluid/face_detction "PaddlePaddle")
- **PCN**: Real-Time Rotation-Invariant Face Detection with Progressive Calibration Networks [[paper]](https://arxiv.org/pdf/1804.06039.pdf "CVPR2018") [[code]](https://github.com/Jack-CV/PCN "C++") 
- **S³FD**: Single Shot Scale-invariant Face Detector [[paper]](https://arxiv.org/pdf/1708.05237.pdf "arXiv2017") [[code]](https://github.com/sfzhang15/SFD "Caffe")
- **SSH**: Single Stage Headless Face Detector [[paper]](http://openaccess.thecvf.com/content_ICCV_2017/papers/Najibi_SSH_Single_Stage_ICCV_2017_paper.pdf "ICCV2017") [[code]](https://github.com/mahyarnajibi/SSH "Caffe")
- **FaceBoxes**: A CPU Real-time Face Detector with High Accuracy [[paper]](https://arxiv.org/pdf/1708.05234.pdf "IJCB2017")[[code1]](https://github.com/zeusees/FaceBoxes "Caffe") [[code2]](https://github.com/lxg2015/faceboxes "PyTorch")
- **TinyFace**: Finding Tiny Faces [[paper]](http://openaccess.thecvf.com/content_cvpr_2017/papers/Hu_Finding_Tiny_Faces_CVPR_2017_paper.pdf "CVPR2017") [[project]](https://www.cs.cmu.edu/~peiyunh/tiny/) [[code1]](https://github.com/peiyunh/tiny "MatConvNet") [[code2]](https://github.com/chinakook/hr101_mxnet "MXNet") [[code3]](https://github.com/cydonia999/Tiny_Faces_in_Tensorflow "TensorFlow")
- **MTCNN**: Joint Face Detection and Alignment using Multi-task Cascaded Convolutional Networks [[paper]](https://kpzhang93.github.io/MTCNN_face_detection_alignment/paper/spl.pdf "SPL2016") [[project]](https://kpzhang93.github.io/MTCNN_face_detection_alignment/) [[code1]](https://github.com/kpzhang93/MTCNN_face_detection_alignment "Caffe") [[code2]](https://github.com/CongWeilin/mtcnn-caffe "Caffe") [[code3]](https://github.com/foreverYoungGitHub/MTCNN "Caffe") [[code4]](https://github.com/Seanlinx/mtcnn "MXNet") [[code5]](https://github.com/pangyupo/mxnet_mtcnn_face_detection "MXNet") [[code6]](https://github.com/TropComplique/mtcnn-pytorch "PyTorch") [[code7]](https://github.com/AITTSMD/MTCNN-Tensorflow "TensorFlow")
- **NPD**: A Fast and Accurate Unconstrained Face Detector [[paper]](http://www.cbsr.ia.ac.cn/users/scliao/papers/Liao-PAMI15-NPD.pdf "TPAMI2015") [[code]](https://github.com/wincle/NPD "C++") [[project]](http://www.cbsr.ia.ac.cn/users/scliao/projects/npdface/index.html)
- **PICO**: Object Detection with Pixel Intensity Comparisons Organized in Decision Trees [[paper]](https://arxiv.org/pdf/1305.4537.pdf "arXiv2014") [[code]](https://github.com/nenadmarkus/pico "C")
- **libfacedetection**: A fast binary library for face detection and face landmark detection in images. [[code]](https://github.com/ShiqiYu/libfacedetection "C++")
- **SeetaFaceEngine**: SeetaFace Detection, SeetaFace Alignment and SeetaFace Identification [[code]](https://github.com/seetaface/SeetaFaceEngine "C++")


## Face Landmark
- **ERT**: One Millisecond Face Alignment with an Ensemble of Regression Trees [[paper]](https://www.cv-foundation.org/openaccess/content_cvpr_2014/papers/Kazemi_One_Millisecond_Face_2014_CVPR_paper.pdf "CVPR2014") [[code]](http://dlib.net/imaging.html "Dlib")

## Face Lib
- **Dlib** [[url]](http://dlib.net/imaging.html "Image Processing") [[github]](https://github.com/davisking/dlib "master")
- **OpenCV** [[docs]](https://docs.opencv.org "All Versions") [[github]](https://github.com/opencv/opencv/ "master")
