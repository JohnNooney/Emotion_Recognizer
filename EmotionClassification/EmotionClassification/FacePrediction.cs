using Microsoft.ML.Data;

namespace EmotionClassification
{
    class FacePrediction
    {
        [ColumnName("PredictedLabel")]
        public string Expression { get; set; }

        [ColumnName("Score")]
        public float[] Scores { get; set; }
    }
}
