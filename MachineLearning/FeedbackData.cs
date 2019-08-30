using Microsoft.ML.Data;

namespace MachineLearning
{
    class FeedbackData
    {
        [LoadColumn(0)]
        public string FeedbackText { get; set; }

        [LoadColumn(1)]
        public int Stars { get; set; }
    }

    class FeedbackPrediction
    {
        [ColumnName("PredictedLabel")]
        public int PredictedStars { get; set; }
    }
}
