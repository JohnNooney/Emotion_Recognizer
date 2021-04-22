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

        public EmotionModel()
        {
            mlContext = new MLContext();
        }

        public void TrainModel(string csv)
        {
            //create instance to train model
            //var mlContext = new MLContext();

            //load data into DataView
            IDataView testDataView = mlContext.Data.LoadFromTextFile<FaceData>(csv, hasHeader: true, separatorChar: ',');

            //split the training data by 80/20
            split = mlContext.Data.TrainTestSplit(testDataView, testFraction: 0.2);

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

            //cross validation that can be used to tune the hyperparameters. These are responsible for controling the learning process
            var crossValidationResults = mlContext.MulticlassClassification.CrossValidate(split.TrainSet, pipeline, numberOfFolds: 5, labelColumnName: "Label");

            //transform data
            var transformedData = model.Transform(split.TestSet);

            //Evaluate on transformed data
            var testMetrics = mlContext.MulticlassClassification.Evaluate(transformedData);

            Console.WriteLine($"LogLoss is: {testMetrics.LogLoss}");
            Console.WriteLine($"PerClassLogLoss is: {String.Join(" , ", testMetrics.PerClassLogLoss.Select(c => c.ToString()))}");
            Console.WriteLine($"* MicroAccuracy:    {testMetrics.MicroAccuracy:0.###}");

            //Predict("Iris-setosa", 0f, 0f, 0f, 0f, 0f, 0f);
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
