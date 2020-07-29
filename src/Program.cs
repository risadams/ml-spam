using System;
using System.IO;
using System.Linq;
using Microsoft.ML;
using Microsoft.ML.Data;

namespace spam
{
    internal class Program
    {
        private static void Main(string[] args)
        {
            var trainer = Path.Combine(Environment.CurrentDirectory, "datasets_483_982_spam.tsv");

            var context = new MLContext();

            var data = context.Data.LoadFromTextFile<ClassificationData>(trainer, hasHeader: false, separatorChar: '\t');

            //use 80% training, 20% testing
            var partitions = context.Data.TrainTestSplit(data, 0.2);


            var pipeline = context.Transforms.CustomMapping<FromLabel, ToLabel>((input, output) => { output.Label = input.RawLabel == "spam"; }, "SpamTrainer")
                                  .Append(context.Transforms.Text.FeaturizeText("Features", nameof(ClassificationData.Message)))
                                  .Append(context.BinaryClassification.Trainers.SdcaLogisticRegression())
                ;

            Console.WriteLine("Cross validation...");
            var results = context.BinaryClassification.CrossValidate(partitions.TrainSet, pipeline);
            foreach (var r in results) Console.WriteLine($"\tFold:{r.Fold}, AUC: {r.Metrics.AreaUnderRocCurve}");

            Console.WriteLine($"Average ROC: {results.Average(r => r.Metrics.AreaUnderRocCurve)}");
            Console.WriteLine();

            Console.WriteLine("Training....");
            var model = pipeline.Fit(partitions.TrainSet);

            Console.WriteLine("Evaluating...");
            var predictions = model.Transform(partitions.TestSet);
            var metrics     = context.BinaryClassification.Evaluate(predictions);

            // report the results
            Console.WriteLine($"  Accuracy:          {metrics.Accuracy:P2}");
            Console.WriteLine($"  Auc:               {metrics.AreaUnderRocCurve:P2}");
            Console.WriteLine($"  Auprc:             {metrics.AreaUnderPrecisionRecallCurve:P2}");
            Console.WriteLine($"  F1Score:           {metrics.F1Score:P2}");
            Console.WriteLine($"  LogLoss:           {metrics.LogLoss:0.##}");
            Console.WriteLine($"  LogLossReduction:  {metrics.LogLossReduction:0.##}");
            Console.WriteLine($"  PositivePrecision: {metrics.PositivePrecision:0.##}");
            Console.WriteLine($"  PositiveRecall:    {metrics.PositiveRecall:0.##}");
            Console.WriteLine($"  NegativePrecision: {metrics.NegativePrecision:0.##}");
            Console.WriteLine($"  NegativeRecall:    {metrics.NegativeRecall:0.##}");
            Console.WriteLine();


            // set up a prediction engine
            Console.WriteLine("Predicting spam probabilities for a sample messages...");
            var predictionEngine = context.Model.CreatePredictionEngine<ClassificationData, SpamPrediction>(model);

// create sample messages
            var messages = new[]
            {
                new ClassificationData { Message = "Hi, wanna grab lunch together today?" },
                new ClassificationData { Message = "Win a Nokia, PSP, or €25 every week. Txt YEAHIWANNA now to join" },
                new ClassificationData { Message = "Home in 30 mins. Need anything from store?" },
                new ClassificationData { Message = "CONGRATS U WON LOTERY CLAIM UR 1 MILIONN DOLARS PRIZE" },
            };

            // make the prediction
            var myPredictions = from m in messages
                                select (m.Message, Prediction: predictionEngine.Predict(m));

            // show the results
            foreach (var (message, prediction) in myPredictions)
                Console.WriteLine($"  [{prediction.Probability:P2}] {message}");
        }
    }

    internal class ClassificationData
    {
        [LoadColumn(0)] public string RawLabel { get; set; }
        [LoadColumn(1)] public string Message { get; set; }
    }

    internal class SpamPrediction
    {
        [ColumnName("PredictedLabel")] public bool IsSpam { get; set; }
        public float Score { get; set; }
        public float Probability { get; set; }
    }

    public class FromLabel
    {
        public string RawLabel { get; set; }
    }

    public class ToLabel
    {
        public bool Label { get; set; }
    }
}