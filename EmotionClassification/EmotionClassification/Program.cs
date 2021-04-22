using System;

namespace EmotionClassification
{
    class Program
    {
        private static EmotionModel model;
        private static FeatureExtraction fExctractor;

        static void Main(string[] args)
        {
            model = new EmotionModel();
            fExctractor = new FeatureExtraction();

            Console.WriteLine("Welcome to the Emotion Classifier.");
            while (true)
            {
                Console.WriteLine("Enter <1> to train model, <2> to test image, <3> to create csv testing data,, <4> to clear console text, <5> to exit ");
                var input = Console.ReadLine().Trim();

                if (input == "1")
                {
                    Console.WriteLine("");
                    model.TrainModel(fExctractor.path + "feature_vectors.csv");
                    model.SaveModel();
                    Console.WriteLine("\n");
                }
                else if(input == "2")
                {
                    model.LoadModel();
                    model.InitPredictor();

                    //get feature vectors of all the faces from the input image
                    float[,] totalVectors = fExctractor.FeatureMaps(fExctractor.path + "test_input.jpg");
                    for (int i = 0; i < totalVectors.GetLength(0); i++)
                    {
                        Console.WriteLine("");
                        model.Predict("", totalVectors[i, 0], totalVectors[i, 1], totalVectors[i, 2], totalVectors[i, 3], totalVectors[i, 4], totalVectors[i, 5]);
                        Console.WriteLine("\n");
                    }

                }
                 else if(input == "3")
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
                else if (input == "5")
                {
                    Environment.Exit(0);
                }
                else
                {
                    Console.WriteLine("Invalid input. Retry. \n\n");
                }
            }
            
        }
    }
}
