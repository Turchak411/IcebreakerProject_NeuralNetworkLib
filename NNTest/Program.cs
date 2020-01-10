using System;
using NeuralNetworkLib;

namespace NNTest
{
    static class Program
    {
        static void Main(string[] args)
        {
            var serviceNN = new ServiceNN(100000);

            serviceNN.Train();
            var result = serviceNN.Handle(new double[] {85, 25, 15});
            Console.ReadKey();
        }
    }
}
