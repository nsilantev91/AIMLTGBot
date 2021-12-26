using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Xml.Serialization;
using System.IO;

namespace AIMLTGBot
{
    [Serializable]
    public class NN     {
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
        public double Speed;

        public NN(double[][,] weights, double[][] charges, double[][] errors, double speed = 0.25)
        {
            _weights = weights;
            _charges = charges;
            _errors = errors;
            Speed = speed;
        }
    }

    static class FileHelper
    {

        public static void WriteToBinaryFile<T>(string filePath, T objectToWrite, bool append = false)
        {
            using (Stream stream = File.Open(filePath, append ? FileMode.Append : FileMode.Create))
            {
                var binaryFormatter = new System.Runtime.Serialization.Formatters.Binary.BinaryFormatter();
                binaryFormatter.Serialize(stream, objectToWrite);
            }
        }
        public static T ReadFromBinaryFile<T>(string filePath)
        {
            using (Stream stream = File.Open(filePath, FileMode.Open))
            {
                var binaryFormatter = new System.Runtime.Serialization.Formatters.Binary.BinaryFormatter();
                return (T)binaryFormatter.Deserialize(stream);
            }
        }

    }
}
