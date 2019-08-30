using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using Microsoft.ML;
using Microsoft.ML.Data;
using static Microsoft.ML.DataOperationsCatalog;

namespace MachineLearning
{
    class Program
    {
        static readonly string _dataPath = Path.Combine(Environment.CurrentDirectory, "Data", "yelp_labelled_stars_500.txt");
        private static MLContext _mlContext;
        private static PredictionEngine<FeedbackData, FeedbackPrediction> _predEngine;
        private static ITransformer _trainedModel;
        static IDataView _trainingDataView;
        static IDataView _testDataView;

        static void Main(string[] args)
        {
            _mlContext = new MLContext(seed: 0);
            
            LoadData();

            var pipeline = ProcessData();

            BuildAndTrainModel(_trainingDataView, pipeline);

            Evaluate(_trainingDataView.Schema);

            var userInput = "";
            while (!string.Equals(userInput.ToLower(), "exit"))
            {
                Console.WriteLine("Write a review or type 'learn' to update model: ");
                userInput = Console.ReadLine();
                switch(userInput.ToLower())
                {
                    case "learn":
                        BuildAndTrainModel(_trainingDataView, pipeline);
                        Evaluate(_trainingDataView.Schema);
                        break;
                    default:
                        UseModelWithUserInput(userInput);
                        break;
                }
                
            }
        }

        public static void LoadData()
        {
            Console.WriteLine("Loading Data...");
            IDataView dataView = _mlContext.Data.LoadFromTextFile<FeedbackData>(_dataPath, hasHeader: false);
            _trainingDataView =  _mlContext.Data.TrainTestSplit(dataView).TrainSet;
            _testDataView = _mlContext.Data.TrainTestSplit(dataView).TestSet;
        }

        public static IEstimator<ITransformer> ProcessData()
        {
            return _mlContext.Transforms.Text.FeaturizeText(outputColumnName: "Features", inputColumnName: nameof(FeedbackData.FeedbackText))
                .Append(_mlContext.Transforms.Conversion.MapValueToKey(outputColumnName: "Label", inputColumnName: nameof(FeedbackData.Stars)));
        }

        public static void BuildAndTrainModel(IDataView trainingDataView, IEstimator<ITransformer> pipeline)
        {
            var trainingPipeline = pipeline.Append(_mlContext.MulticlassClassification.Trainers.SdcaMaximumEntropy("Label", "Features"))
                .Append(_mlContext.Transforms.Conversion.MapKeyToValue("PredictedLabel"));

            _trainedModel = trainingPipeline.Fit(trainingDataView);
            _predEngine = _mlContext.Model.CreatePredictionEngine<FeedbackData, FeedbackPrediction>(_trainedModel);
            FeedbackData feedback = new FeedbackData()
            {
                FeedbackText = "I love this feedback",
                Stars = 5
            };

            var prediction = _predEngine.Predict(feedback);
            Console.WriteLine($"=============== Single Prediction just-trained-model - Result: {prediction.PredictedStars} ===============");
        }

        public static void Evaluate(DataViewSchema trainingDataViewSchema)
        {
            var testMetrics = _mlContext.MulticlassClassification.Evaluate(_trainedModel.Transform(_testDataView));

            Console.WriteLine($"*************************************************************************************************************");
            Console.WriteLine($"*       Metrics for Multi-class Classification model - Test Data     ");
            Console.WriteLine($"*------------------------------------------------------------------------------------------------------------");
            Console.WriteLine($"*       MicroAccuracy:    {testMetrics.MicroAccuracy:0.###}");
            Console.WriteLine($"*       MacroAccuracy:    {testMetrics.MacroAccuracy:0.###}");
            Console.WriteLine($"*       LogLoss:          {testMetrics.LogLoss:#.###}");
            Console.WriteLine($"*       LogLossReduction: {testMetrics.LogLossReduction:#.###}");
            Console.WriteLine($"*************************************************************************************************************");
        }

        private static void UseModelWithUserInput(string userInput)
        {
            PredictionEngine<FeedbackData, FeedbackPrediction> predictionFunction = _mlContext.Model.CreatePredictionEngine<FeedbackData, FeedbackPrediction>(_trainedModel);
            FeedbackData sampleStatement = new FeedbackData
            {
                FeedbackText = userInput
            };

            var resultPrediction = predictionFunction.Predict(sampleStatement);

            Console.WriteLine();
            Console.WriteLine("=============== Prediction Test of model with a single sample and test dataset ===============");

            Console.WriteLine();
            Console.WriteLine($"Prediction: {resultPrediction.PredictedStars}");

            Console.WriteLine("=============== End of Predictions ===============");
            Console.WriteLine();
            Console.WriteLine("Was this prediction correct? (Y/N): ");
            var correctStars = 0;
            var predictionCorrect = Console.ReadLine();

            if(string.Equals(predictionCorrect.ToUpper(), "N"))
            {
                Console.WriteLine("How many stars should it be? ");
                correctStars = int.Parse(Console.ReadLine());
            }
            else
            {
                correctStars = resultPrediction.PredictedStars;
            }
            File.AppendAllText(_dataPath, userInput + "\t" + correctStars + Environment.NewLine);
        }
    }
}