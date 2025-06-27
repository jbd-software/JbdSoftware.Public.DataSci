using System;
using System.Collections.Generic;

namespace LinearRegressionFromScratch
{
    class DataPoint
    {
        public double X { get; set; }
        public double Y { get; set; }

        public DataPoint(double x, double y)
        {
            X = x;
            Y = y;
        }
    }

    class LinearRegression
    {
        public double Theta0 { get; private set; } = 0;
        public double Theta1 { get; private set; } = 0;
        private double learningRate;
        private int iterations;

        public LinearRegression(double alpha, int iterations)
        {
            learningRate = alpha;
            this.iterations = iterations;
        }

        private double Hypothesis(double x)
        {
            return Theta0 + Theta1 * x;
        }

        public void Train(List<DataPoint> data)
        {
            int m = data.Count;

            for (int iter = 0; iter < iterations; iter++)
            {
                double grad0 = 0.0;
                double grad1 = 0.0;

                foreach (var point in data)
                {
                    double error = Hypothesis(point.X) - point.Y;
                    grad0 += error;
                    grad1 += error * point.X;
                }

                grad0 /= m;
                grad1 /= m;

                Theta0 -= learningRate * grad0;
                Theta1 -= learningRate * grad1;
            }
        }

        public void PrintModel()
        {
            Console.WriteLine($"Learned Model: h(x) = {Theta0:F4} + {Theta1:F4} * x");
        }

        public void Predict(List<DataPoint> data)
        {
            Console.WriteLine("\nPredictions:");
            foreach (var point in data)
            {
                double prediction = Hypothesis(point.X);
                Console.WriteLine($"Size: {point.X} => Predicted Price: {prediction:F2}, Actual: {point.Y}");
            }
        }
    }

    class Program
    {
        static void Main(string[] args)
        {
            var data = new List<DataPoint>
            {
                new DataPoint(50, 150),
                new DataPoint(60, 160),
                new DataPoint(80, 200),
                new DataPoint(100, 230),
                new DataPoint(120, 275),
            };

            var model = new LinearRegression(alpha: 0.0005, iterations: 10000);
            model.Train(data);
            model.PrintModel();
            model.Predict(data);
        }
    }
}
