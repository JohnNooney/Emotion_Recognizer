using System;
using System.IO;

namespace EmotionClassification
{
    class Program
    {
        private static EmotionModel model;
        private static FeatureExtraction fExctractor;

        static void Main(string[] args)
        {
            fExctractor = new FeatureExtraction();
            model = new EmotionModel(fExctractor.path + "feature_vectors.csv");

            Console.WriteLine("Welcome to the Emotion Classifier.");
            while (true)
            {
                Console.WriteLine("Enter <0> to train model, Enter <1> to view model evaluation, <2> to test image, <3> to create csv testing data,, <4> to clear console text, <5> to exit ");
                var input = Console.ReadLine().Trim();

                if (input == "0")
                {
                    Console.WriteLine("");
                    model.TrainModel();
                    model.SaveModel();
                    Console.WriteLine("\n");
                }
                if (input == "1")
                {
                    Console.WriteLine("");
                    model.LoadModel();
                    model.ModelMetrics();
                    Console.WriteLine("\n");
                }
                else if(input == "2")
                {
                    PredictAllImages();
                }
                else if (input == "3")
                {
                    Console.WriteLine("");
                    fExctractor.CreateCSV();
                    Console.WriteLine("\n");

                }
                else if (input == "4")
                {
                    Console.Clear();
                    Console.WriteLine("Welcome to the Emotion Classifier.");
                }
                else if (input == "5") { 
                    Environment.Exit(0);
                }
                else
                {
                    Console.WriteLine("Invalid input. Retry. \n\n");
                }
            }
            
        }

        //method to predict all images within the tesiImages directory
        public static void PredictAllImages()
        {
            model.LoadModel();
            model.InitPredictor();

            //get all image files within the test images directory and predict them
            string testImgPath = fExctractor.path + "testImages";
            foreach (var file in Directory.GetFiles(testImgPath))
            {
                //Console.WriteLine(file);
                //get feature vectors of all the faces from the input image
                float[,] totalVectors = fExctractor.FeatureMaps(file);

                if (totalVectors != null)
                {
                    //cycle through each face present in the image
                    for (int i = 0; i < totalVectors.GetLength(0); i++)
                    {
                        Console.WriteLine("");

                        string[] filename = file.Split(@"\");
                        Console.WriteLine(filename[1] + " done.");
                      
                        model.Predict("", totalVectors[i, 0], totalVectors[i, 1], totalVectors[i, 2], totalVectors[i, 3], totalVectors[i, 4], totalVectors[i, 5]);
                        Console.WriteLine("\n");
                    }
                }
            }
        }
    }
}
