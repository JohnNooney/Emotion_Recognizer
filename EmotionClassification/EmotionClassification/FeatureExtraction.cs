using DlibDotNet;
using System;
using System.IO;
using Dlib = DlibDotNet.Dlib;

namespace EmotionClassification
{
    class FeatureExtraction
    {
        // file paths
        public const string _path = "C:/Users/John/source/repos/EmotionClassification/EmotionClassification/";
        private const string datPath = _path + "shape_predictor_68_face_landmarks.dat";
        private const int featureNum = 6;

        public string path { get; private set; }

        public FeatureExtraction()
        {
            path = _path;
        }

        // Create a CSV of feature vectors from the folders of images found within the Images directory
        public void CreateCSV()
        {
            //create CSV file to save feature vectors to
            string header = "label,leftEyebrow,rightEyebrow,leftLip,rightLip,lipWidth,lipHeight\n";
            System.IO.File.WriteAllText(@"feature_vectors.csv", header);

            foreach (var dir in Directory.GetDirectories(_path + "Images"))
            {
                foreach (var dirDepth2 in Directory.GetDirectories(dir))
                {
                    Console.WriteLine("New Image Directory Start.");

                    var emotion = dirDepth2.Split(@"\");
                    Console.WriteLine(emotion[1] + " " + emotion[2] + " Seciton Start");
                    int count = 0;
                    foreach (var file in Directory.GetFiles(dirDepth2))
                    {
                        //Console.WriteLine(file);
                        //get feature vector of all the faces in an image
                        float[,] totalVectors = FeatureMaps(file);

                        if (totalVectors != null)
                        {
                            for (int i = 0; i < totalVectors.GetLength(0); i++)
                            {
                                //write in the data to the file
                                using (System.IO.StreamWriter csv = new System.IO.StreamWriter(@"feature_vectors.csv", true))
                                {
                                    //ex: "anger,5,5,4,4,3,0"
                                    csv.WriteLine(emotion[2] + "," + totalVectors[i,0] + "," + totalVectors[i,1] + "," + totalVectors[i,2] + "," + totalVectors[i,3] + "," + totalVectors[i,4] + "," + totalVectors[i,5]);
                                }
                            }
                        }

                        Console.WriteLine(count + " image done.");
                        count++;
                    }

                    Console.WriteLine(emotion[2] + " section done.");
                }

            }

        }

        //Generates a feature map for the 
        public float[,] FeatureMaps(string imgPath)
        {
            string inputFilePath = imgPath;
            float[,] totalFeatureVectors = null;

            // Set up Dlib Face Detector
            using (var fd = Dlib.GetFrontalFaceDetector())
            // ... and Dlib Shape Detector
            using (var sp = ShapePredictor.Deserialize(datPath))
            {
                // load input image
                var tmp = Dlib.LoadImage<RgbPixel>(inputFilePath);

                //coloured datasets 
                //var bmp = new Bitmap(inputFilePath);
                //var tmp = bmp.To8bppIndexedGrayscale(GrayscalLumaCoefficients.ITU_R_BT_601).ToArray2D<byte>();

                //greyscaled datasets
                //var tmp = Dlib.LoadImage<RgbPixel>(filePaths[i]);

                var img = new Array2D<byte>();
                Dlib.GaussianBlur(tmp, img);

                // find all faces in the image
                var faces = fd.Operator(img);

                //allocate array for number of faces found
                totalFeatureVectors = new float[faces.Length, featureNum];

                // for each face find the feature vector
                for (int i = 0; i < faces.Length; i++)
                {
                    // find the landmark points for this face
                    var shape = sp.Detect(img, faces[i]);

                    //draw points on face
                    //DrawFacePoints(shape, img);

                    //get feature vector of the face
                    double[] features = CalcNormDistances(shape);

                    //fill the 2D array that holds the vectors for each face found
                    for (int j = 0; j < features.Length; j++)
                    {
                        totalFeatureVectors[i,j] = (float)features[j];
                    }
                }

                // export the modified image
                //Dlib.SaveJpeg(img, "output.jpg");
            }

            return totalFeatureVectors;
        }

        //Creates the feature vector of the input image. Gets the distance of 
        //certain points within a region of the face to generate a set of values to associate
        //an emotion to
        private double[] CalcNormDistances(FullObjectDetection shape)
        {
            double[] tempFeats = new double[featureNum];

            //get left eyebrow 
            //calculate distances from eye to eyebrow points
            double d1 = GetDistance(shape.GetPart((uint)39), shape.GetPart((uint)21)); //**actual element is n-1
            double d2 = GetDistance(shape.GetPart((uint)39), shape.GetPart((uint)20));
            double d3 = GetDistance(shape.GetPart((uint)39), shape.GetPart((uint)19));
            double d4 = GetDistance(shape.GetPart((uint)39), shape.GetPart((uint)18));
            //normalize distances
            double normalizer = d1; //normalizes based on size of face
            d1 /= normalizer;
            d2 /= normalizer;
            d3 /= normalizer;
            d4 /= normalizer;
            //sum values
            double leftEyeBrow = d1 + d2 + d3 + d4;
            tempFeats[0] = leftEyeBrow;
            //Console.WriteLine("LE: " + leftEyeBrow);

            //get right eyebrow
            //calculate distances from eye to eyebrow points
            d1 = GetDistance(shape.GetPart((uint)42), shape.GetPart((uint)22));
            d2 = GetDistance(shape.GetPart((uint)42), shape.GetPart((uint)23));
            d3 = GetDistance(shape.GetPart((uint)42), shape.GetPart((uint)24));
            d4 = GetDistance(shape.GetPart((uint)42), shape.GetPart((uint)25));
            //normalize distances
            normalizer = d1; //normalizes based on size of face
            d1 /= normalizer;
            d2 /= normalizer;
            d3 /= normalizer;
            d4 /= normalizer;
            //sum values
            double rightEyeBrow = d1 + d2 + d3 + d4;
            tempFeats[1] = rightEyeBrow;
            //Console.WriteLine("RE: " + rightEyeBrow);

            //get left lip
            d1 = GetDistance(shape.GetPart((uint)33), shape.GetPart((uint)51)); //only used for normalization
            d2 = GetDistance(shape.GetPart((uint)33), shape.GetPart((uint)50));
            d3 = GetDistance(shape.GetPart((uint)33), shape.GetPart((uint)49));
            d4 = GetDistance(shape.GetPart((uint)33), shape.GetPart((uint)48));
            normalizer = d1;
            d2 /= normalizer;
            d3 /= normalizer;
            d4 /= normalizer;
            //sum values
            double leftLip = d2 + d3 + d4;
            tempFeats[2] = leftLip;
            //Console.WriteLine("LL: " + leftLip);

            //get right lip
            d2 = GetDistance(shape.GetPart((uint)33), shape.GetPart((uint)52));
            d3 = GetDistance(shape.GetPart((uint)33), shape.GetPart((uint)53));
            d4 = GetDistance(shape.GetPart((uint)33), shape.GetPart((uint)54));
            //*use same normalizer as from left lip
            d2 /= normalizer;
            d3 /= normalizer;
            d4 /= normalizer;
            double rightLip = d2 + d3 + d4;
            tempFeats[3] = rightLip;
            //Console.WriteLine("RL: " + rightLip);

            //get lip width
            d1 = GetDistance(shape.GetPart((uint)33), shape.GetPart((uint)51));
            d2 = GetDistance(shape.GetPart((uint)48), shape.GetPart((uint)54));
            normalizer = d1;
            d2 /= normalizer;
            tempFeats[4] = d2;
            //Console.WriteLine("LW: " + d2);

            //get lip height
            d2 = GetDistance(shape.GetPart((uint)51), shape.GetPart((uint)57));
            d2 /= normalizer;
            tempFeats[5] = d2;
            //Console.WriteLine("LH: " + d2);

            return tempFeats;
        }

        //Calculate the distance between two given points
        private double GetDistance(DlibDotNet.Point point1, DlibDotNet.Point point2)
        {

            return Math.Sqrt(Math.Pow((point2.X - point1.X), 2) + Math.Pow((point2.Y - point1.Y), 2));
        }

        //For visulaizing how DLib places the certain points on the face of the input image
        private void DrawFacePoints(FullObjectDetection shape, Array2D<RgbPixel> img)
        {
            // draw the landmark points on the image
            for (var i = 0; i < shape.Parts; i++)
            {

                var point = shape.GetPart((uint)i);
                var rect = new DlibDotNet.Rectangle(point);

                switch (i + 1) //compensate for feature map starting at 1
                {
                    case 22:
                        Dlib.DrawRectangle(img, rect, color: new RgbPixel(0, 0, 255), thickness: 4);
                        break;
                    case 21:
                        Dlib.DrawRectangle(img, rect, color: new RgbPixel(255, 0, 0), thickness: 4);
                        break;
                    case 20:
                        Dlib.DrawRectangle(img, rect, color: new RgbPixel(255, 0, 0), thickness: 4);
                        break;
                    case 19:
                        Dlib.DrawRectangle(img, rect, color: new RgbPixel(255, 0, 0), thickness: 4);
                        break;
                    case 40:
                        Dlib.DrawRectangle(img, rect, color: new RgbPixel(0, 0, 255), thickness: 4);
                        break;
                    default:
                        Dlib.DrawRectangle(img, rect, color: new RgbPixel(255, 225, 0), thickness: 4);
                        break;
                }

            }
        }
    }
}
