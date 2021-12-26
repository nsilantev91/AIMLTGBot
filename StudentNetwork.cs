using System;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Threading;
using System.Collections.Generic;
using System.Text;
using System.Threading.Tasks;

namespace AIMLTGBot
{ 
    public class StudentNetwork : BaseNetwork
    {
        /// <summary>
        /// Массив входных матриц весов
        /// </summary>
        public double[][,] _weights;

        /// <summary>
        /// Входной взвешенный сигнал
        /// </summary>
        public double[][] _charges;

        /// <summary>
        /// Значение ошибки
        /// </summary>
        public double[][] _errors;
        
        /// <summary>
        /// Скорость обучения
        /// </summary>
        public  double Speed = 0.25;
        
       private readonly Stopwatch _stopWatch = new Stopwatch();

        public StudentNetwork() { }
        public StudentNetwork(int[] structure, double lowerBound = -1, double upperBound = 1)
        {
            _charges = new double[structure.Length][];
            _errors = new double[structure.Length][];
            
            for (int i = 0; i < structure.Length; i++)
            {
                _errors[i] = new double[structure[i]];
                _charges[i] = new double[structure[i] + 1];
                _charges[i][structure[i]] = 1;
            }
            
            _weights = new double[structure.Length - 1][,];
            
            for(int n = 0; n < structure.Length - 1; n++)
            {
                var r = new Random();
                var rowsCount = structure[n] + 1;
                var columnsCount = structure[n + 1];
                
                _weights[n] = new double[rowsCount, columnsCount];
                
                for (int i = 0; i < rowsCount; i++)
                    for (int j = 0; j < columnsCount; j++)
                        _weights[n][i, j] = lowerBound + r.NextDouble() * (upperBound - lowerBound);
            }
        }

        /// <summary>
        /// Вычисление квадратичной ошибки по выходу сети, аналог sample.EstimatedError для массива
        /// </summary>
        private double Error(double[] output)
        {
            double result = 0;
            
            for (int i = 0; i < output.Length; i++) 
                result += System.Math.Pow(output[i] - _charges[_charges.Length - 1][i], 2);
            
            result /= output.Length;
            
            return result;
        }

        /// <summary>
        /// Прямой проход сети
        /// </summary>
        private void Run(double[] input)
        {
            for (int j = 0; j < input.Length; j++) 
                _charges[0][j] = input[j];
            
            for (int i = 1; i < _charges.GetLength(0); i++)
                MultiplyAndApplySigmoid(_charges[i - 1], _weights[i - 1], _charges[i]);
        }

        /// <summary>
        /// Функция активации слоя сети
        /// </summary>
        private static void MultiplyAndApplySigmoid(double[] vector, double [,] matrix, double[] result)
        {
            var rowsCount = matrix.GetLength(0);
            var colCount = matrix.GetLength(1);
            
            for (int i = 0; i < colCount ; i++)
            {
                double sum = 0;
                
                for(int j = 0; j < rowsCount; j++) 
                    sum += vector[j] * matrix[j, i];
                
                result[i] = Sigmoid(sum);
            }  
        }

        /// <summary>
        /// Сигмоида, активирующая "нейрон"
        /// </summary>
        private static double Sigmoid(double value) => 1.0 / (System.Math.Exp(-value) + 1);

        /// <summary>
        /// Обратное распространение ошибки
        /// </summary>
        private void BackPropagation(double[] output)
        {
            // Обработка ошибки
            for (var j = 0; j < output.Length; j++)
            {
                var currentCharge = _charges[_errors.Length - 1][j];
                var expectedOutput = output[j];
                _errors[_errors.Length - 1][j] = currentCharge * (1 - currentCharge) * (expectedOutput - currentCharge);
            }
            // Пересчёт ошибки
            for (int i = _errors.Length - 2; i >= 1; i--)
                for (int j = 0; j < _errors[i].Length; j++)
                {
                    var charge = _charges[i][j];
                    charge *= 1 - charge;
                    var sum = 0.0;
                    for (int k = 0; k < _errors[i + 1].Length; k++)
                        sum += _errors[i + 1][k] * _weights[i][j, k];
                    _errors[i][j] = charge * sum;
                }
        }

        /// <summary>
        /// Пересчёт весов
        /// </summary>
        private void ComputeWeights()
        {
           
            for(int n = 0; n < _weights.Length; n++)
            {
                for(int i = 0; i < _weights[n].GetLength(0); i++)
                {
                    for(int j = 0; j < _weights[n].GetLength(1); j++)
                    {
                        var dWeight = Speed * _errors[n + 1][j] * _charges[n][i];
                        _weights[n][i, j] += dWeight;
                    }
                }
            }
        }

        public override int Train(Sample sample, double acceptableError, bool parallel)
        {
            int i = 1;
            while (sample.EstimatedError() > acceptableError)
            {
                Run(sample.input);
                BackPropagation(sample.Output);
                ComputeWeights();
                ++i;
            }

            return i;
        }

        public override double TrainOnDataSet(SamplesSet samplesSet, int epochsCount, double acceptableError, bool parallel)
        {
            // Конструируем массивы входов и выходов
            double[][] inputs = new double[samplesSet.Count][];
            double[][] outputs = new double[samplesSet.Count][];

            // Группируем массивы из samplesSet в inputs и outputs
            for (int i = 0; i < samplesSet.Count; ++i)
            {
                inputs[i] = samplesSet[i].input;
                outputs[i] = samplesSet[i].Output;
            }

            int epochToRun = 0;
            double samplesLooked = 0;
            double samplesCount = inputs.Length * epochsCount;
            double error = double.PositiveInfinity;

#if DEBUG
            StreamWriter errorsFile = File.CreateText("errors.csv");
#endif

           _stopWatch.Restart();

            while (epochToRun++ < epochsCount && error > acceptableError)
            {
                error = 0;
                for (int i = 0; i < inputs.Length; i++)
                {
                    Run(inputs[i]);
                    BackPropagation(outputs[i]);
                    ComputeWeights();
                    error += Error(outputs[i]);
                    ++samplesLooked;
                }
                error /= inputs.Length;
                errorsFile.WriteLine(error);
                OnTrainProgress(samplesLooked / samplesCount, error, _stopWatch.Elapsed);
            }

#if DEBUG
            errorsFile.Close();
#endif
            OnTrainProgress(1, error, _stopWatch.Elapsed);
            _stopWatch.Stop();
            return error;
        }

        protected override double[] Compute(double[] input)
        {
            Run(input);
            return _charges.Last().Take(_charges.Last().Length - 1).ToArray();
        }

  
    }
}