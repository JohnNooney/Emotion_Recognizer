using Microsoft.ML.Data;

namespace EmotionClassification
{
    class FaceData
    {
        [LoadColumn(0)]
        public string expression { get; set; }

        [LoadColumn(1)]
        public float leftEyebrow { get; set; }

        [LoadColumn(2)]
        public float rightEyebrow { get; set; }

        [LoadColumn(3)]
        public float leftLip { get; set; }

        [LoadColumn(4)]
        public float rightLip { get; set; }

        [LoadColumn(5)]
        public float lipWidth { get; set; }

        [LoadColumn(6)]
        public float lipHeight { get; set; }
    }
}
