using Microsoft.ML;
using System;
using System.IO;
using System.Linq;

namespace EmotionClassification
{
    class EmotionModel
    {
        private MLContext mlContext;
        private PredictionEngine<FaceData, FacePrediction> predictor;
        private DataOperationsCatalog.TrainTestData split;
        private DataViewSchema modelSchema; //Define DataViewSchema for data preparation pipeline and trained model
        private ITransformer model;
        private string csv;

        public EmotionModel(string _csv)
        {
            mlContext = new MLContext();

            csv = _csv;

            //load data into DataView
            IDataView testDataView = mlContext.Data.LoadFromTextFile<FaceData>(csv, hasHeader: true, separatorChar: ',');

            //split the training data by 80/20
            split = mlContext.Data.TrainTestSplit(testDataView, testFraction: 0.2);
        }

        public void TrainModel()
        {
            //create instance to train model
            //var mlContext = new MLContext();


            //define models pipeline
            var featureVectorName = "Features";
                var labelColumnName = "Label";
                var pipeline =
                    mlContext.Transforms.Conversion.MapValueToKey(
                        inputColumnName: "expression", // main label
                        outputColumnName: "Label"
                        )
                        .Append(mlContext.Transforms.Concatenate(featureVectorName,
                        "leftEyebrow",
                        "rightEyebrow",
                        "leftLip",
                        "rightLip",
                        "lipWidth",
                        "lipHeight"
                        ))
                        .AppendCacheCheckpoint(mlContext)
                        .Append(mlContext.MulticlassClassification.Trainers.SdcaMaximumEntropy(labelColumnName, featureVectorName))
                        .Append(mlContext.Transforms.Conversion.MapKeyToValue("PredictedLabel"));

            //fit the model (train)
            model = pipeline.Fit(split.TrainSet);


            //may need to use regularization to reduce overfitting. Works by penalizing the magnitude of model parameters.
            //** SdcaMaxEntropy already implements regularization into the algorithm


            //transform data
            var transformedData = model.Transform(split.TestSet);

            //Evaluate on transformed data
            var testMetrics = mlContext.MulticlassClassification.Evaluate(transformedData);

            Console.WriteLine($"LogLoss is: {testMetrics.LogLoss}");
            
            Console.WriteLine($"PerClassLogLoss is: {String.Join(" , ", testMetrics.PerClassLogLoss.Select(c => c.ToString()))}");
            Console.WriteLine($"MicroAccuracy:    {testMetrics.MicroAccuracy:0.###}");

            //Predict("Iris-setosa", 0f, 0f, 0f, 0f, 0f, 0f);
        }

        public void ModelMetrics()
        {
            //transform data
            var transformedData = model.Transform(split.TestSet);

            //Evaluate on transformed data
            var testMetrics = mlContext.MulticlassClassification.Evaluate(transformedData);

            Console.WriteLine("Following data based on a 20% split from training set.");
            Console.WriteLine(testMetrics.ConfusionMatrix.GetFormattedConfusionTable());
            Console.WriteLine($"Average Log-Loss is: {testMetrics.LogLoss}");
            Console.WriteLine($"Per Class Log-Loss is: {String.Join(" , ", testMetrics.PerClassLogLoss.Select(c => c.ToString()))}");
            Console.WriteLine($"MicroAccuracy:    {testMetrics.MicroAccuracy:0.###}");
            Console.WriteLine($"MacroAccuracy:    {testMetrics.MacroAccuracy:0.###}");

        }

        public void InitPredictor()
        {
            //Init predictor engine
            predictor = mlContext.Model.CreatePredictionEngine<FaceData, FacePrediction>(model);
        }

        public void LoadModel()
        {
            // Load trained model
            if (model == null)
            {
                model = mlContext.Model.Load("model.zip", out modelSchema);
            }
        }

        public void SaveModel()
        {
            //save model
            using (var fileStream = new FileStream("model.zip", FileMode.Create, FileAccess.Write, FileShare.Write))
            { mlContext.Model.Save(model, split.TrainSet.Schema, fileStream); }

        }



        //Make a prediciton based off the feature vector using the trained model model
        public void Predict(string e, float le, float re, float ll, float rl, float lw, float lh)
        {
            //pass data
            var prediction = predictor.Predict(new FaceData()
            {
                expression = e,
                leftEyebrow = le,
                rightEyebrow = re,
                leftLip = ll,
                rightLip = rl,
                lipWidth = lw,
                lipHeight = lh
            });

            Console.WriteLine($"*** Prediciton: {prediction.Expression} ***");
            Console.WriteLine($"*** Scores: {string.Join(" ", prediction.Scores)} ***");
        }
    }
}
