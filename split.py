import splitfolders
import random
import sys
import getopt


if __name__ == '__main__':
    opts, args = getopt.getopt(sys.argv[1:],'hi:o:',['train=','test=','validation=','input','output'])
    train_ratio = 0
    test_ratio = 0
    val_ratio = 0
    input = ''
    output = ''
    for opt, arg in opts:
        if opt == '-h':
            print("split.py -i INPUT_PATH -o OUTPUTPATH --train <TRAIN_RATIO> --test <TEST_RATIO> --validation <VALIDATION_RATIO>")
            print('TRAIN_RATIO + TEST_RATIO + VALIDATION_RATIO must be equal with 1.')
            sys.exit()
        elif opt in ('--train'):
            train_ratio = float(arg)
        elif opt in ('--test'):
            test_ratio = float(arg)
        elif opt in ('--validation'):
            val_ratio = float(arg)
        elif opt in ('--input', '-i'):
            input = arg
        elif opt in ('--output', '-i'):
            output = arg
    
    if train_ratio + test_ratio + val_ratio != 1:
        print('TRAIN_RATIO + TEST_RATIO + VALIDATION_RATIO must be equal with 1.')
        sys.exit()
            
    splitfolders.ratio(input=input, output=output, seed=random.randint(0,255), ratio=(train_ratio, val_ratio, test_ratio))
    print('Split complete.')
