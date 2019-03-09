
import argparse


import merge

defaultVersion = '0.01'

# Instantiate the parser
parser = argparse.ArgumentParser(description='Optional app description')

parser.add_argument('-e','--epochs', type=int,
                    help='Amount of epochs to run', default=1000)
parser.add_argument('-p','--patience', type=int,
                    help='Amount of patience before ending the model training', default=30)

parser.add_argument('--validation_split', type=float,
                    help='Split the training set for validation', default=0.2)
parser.add_argument('--batch_size', type=int,
                    help='batch_size', default=5000)
parser.add_argument('-v','--verbose', type=int,
                    help='How verbose 0-3', default=0)
parser.add_argument('--version', type=str,
                    help='What version', default=defaultVersion)
parser.add_argument('--test', type=str,
                    help='What are you testing', default="none")
parser.add_argument('--note', type=str,
                    help='any notes', default="none")
parser.add_argument('--FFNN', type=bool,
                    help='any notes', default=False)
parser.add_argument('--LSTM', type=bool,
                    help='any notes', default=False)
parser.add_argument('-m','--MERGE', type=bool,
                    help='any notes', default=False)
parser.add_argument('-T','--run_test', type=bool,
                    help='use the test array in *test.py', default=False)
parser.add_argument('--sequence_size', type=int,
                    help='size of the lstm sequence only works for LSTM and Merge', default=5)

args=parser.parse_args()

if args.FFNN == True:
    if args.run_test == True:
        import FFNNtest
        FFNNtest.run(args)
    else:
        import FFNN
        FFNN.main(args)
elif args.LSTM == True:
    if args.run_test == True:
        import LSTMtest
        LSTMtest.run(args)
    else:
        import LSTM
        LSTM.main(args) 
elif args.MERGE == True:
    if args.run_test == True:
        import mergetest
        mergetest.run(args)
    else:
        import merge
        merge.main(args)
else:
    print("Must select a type")
