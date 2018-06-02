from chainer import Chain, Function, optimizers, iterators
from itertools import product
import chainer.functions as F
import chainer.links as L
import random
import numpy as np
from chainer.dataset import concat_examples
import json
import os

class NetEnsemble:

    def __init__(self, baseDir):
        
        trainPath = os.path.join(os.path.join(baseDir, 'train'), 'data')
        testPath = os.path.join(os.path.join(baseDir, 'train'), 'data')

        train = []
        test = []

        def readSample(path, dest):
            for subdirs, dirs, files in os.walk(path):
                for file in files:
                    if file.endswith('.json'):
                        with open(os.path.join(subdirs, file)) as partOfSample:
                            data = json.load(partOfSample)
                            dest.extend(data)

        readSample(trainPath, train)
        readSample(testPath, test)

        self.phones = {}

        for elem in train:
            fileId, label, phone, mfcc = elem
            
            if phone not in self.phones:
                self.phones[phone] = {}

            if label not in self.phones[phone]:
                self.phones[phone][label] = []

            self.phones[phone][label].append(np.array(mfcc, dtype=np.float32))


        """ Число MFCC коэффициентов """
        self.featuresCount = 23

        """ One-hot вектора с правильными ответами для пары фонема+лэйбл
        dict[phone][label] = (0,0,...,0,1,0,0,0)
        """
        self._rightAnswers = {}

        self.nets = {}
        self._trainData = {}

        for phone in self.phones:
            self._rightAnswers[phone] = {}
            labelsCount = len(self.phones[phone])

            for index, label in enumerate(self.phones[phone]):
                zeros = np.zeros((labelsCount))
                rightAnswer = np.array(zeros, dtype=np.float32)
                rightAnswer[index] = 1.0
                self._rightAnswers[phone][label] = rightAnswer

        self.files = {}

        for elem in test:
            fileId, label, phone, mfcc = elem

            realization = (phone, np.array(mfcc, dtype=np.float32))

            if fileId not in self.files:
                self.files[fileId] = (label, self._rightAnswers[phone][label], [realization])
            else:
                self.files[fileId][2].append(realization)

        for phone in self.phones:
            self.nets[phone] = Phone_Net(self.featuresCount, len(self.phones[phone]))
            self._trainData[phone] = []

            for index, label in enumerate(self.phones[phone]):

                ownPronunciations = [(e, label) for e in self.phones[phone][label]]
                random.shuffle(ownPronunciations)

                foreignLabels = [label2 for label2 in self.phones[phone] if label2 != label]
                foreignPronunciations = []
                for label2 in foreignLabels:
                    foreignPronunciations.extend([(e, label2) for e in self.phones[phone][label2]])
                #[item for sublist in l for item in sublist]
                random.shuffle(foreignPronunciations)


                temp = []
                for e in product(ownPronunciations, foreignPronunciations):
                    ra1 = self._rightAnswers[phone][e[0][1]]
                    ra2 = self._rightAnswers[phone][e[1][1]]
                    temp.append(((e[0][0], e[1][0]), (ra1, ra2)))
                self._trainData[phone].extend(temp)


        return


    def train(self, maxEpoch):
        max_epoch = 10
        batchsize = 128

        for i in range(max_epoch):
            for phone in self.nets:
                self.trainCorrespondingNet(phone, batchsize)

        for file in self.files:
            scores = []
            label, rightAnswer, realisations = self.files[file]
            for realisation in realisations:
                phone, array = realisation
                voice = self.nets[phone](array)
                scores.append(voice)


        return

    def trainCorrespondingNet(self, phone, batchsize):
        model = self.nets[phone]
            
        optimizer = optimizers.MomentumSGD(lr=0.01, momentum=0.9)
        optimizer.setup(model)

        data = self._trainData[phone]
        train_iter = iterators.SerialIterator(data, batchsize)


        train_batch = train_iter.next()
        data, labels = concat_examples(train_batch)

        dataPart_first = data[:,0]
        dataPart_second = data[:,1]

        prediction_first = model(dataPart_first)
        prediction_second = model(dataPart_second)

        rightAnswers_first = labels[:,0]
        rightAnswers_second = labels[:,1]

        loss = self.FRISLoss(prediction_first, prediction_second, rightAnswers_first, rightAnswers_second)

        model.cleargrads()
        loss.backward()

        optimizer.update()

        if train_iter.is_new_epoch:
            print('phone: {0}, epoch:{:02d} train_FRIS_loss:{:.04f} '.format(phone, train_iter.epoch, float(loss.data)), end='')
            return

    def printMeanTestErrors(self, phone, batchsize):
        model = self.nets[phone]

        data = self._testData
        test_iter = iterators.SerialIterator(data, batchsize)

        test_losses = []
        test_accuracies = []
        while True:
            test_batch = test_iter.next()
            image_test, target_test = concat_examples(test_batch)

            # Forward the test data
            prediction_test = model(image_test)

            # Calculate the loss
            loss_test = F.softmax_cross_entropy(prediction_test, target_test)
            test_losses.append(loss_test.data)

            # Calculate the accuracy
            accuracy = F.accuracy(prediction_test, target_test)
            test_accuracies.append(accuracy.data)

            if test_iter.is_new_epoch:
                test_iter.epoch = 0
                test_iter.current_position = 0
                test_iter.is_new_epoch = False
                test_iter._pushed_position = None
                break

        print('mean of loss:{:.04f} mean of accuracy:{:.04f}'.format(
            np.mean(test_losses), np.mean(test_accuracies)))

    def FRISLoss(self, firstOut, secondOut, firstLabels, secondLabels):
        d1 = firstOut - secondOut
        d2 = firstOut - firstLabels
        d3 = secondOut - secondLabels

        d1 = F.sum(d1 * d1)
        d2 = F.sum(d2 * d2)
        d3 = F.sum(d3 * d3)
        
        f1 = (d1 - d2) / (d1 + d2)
        f2 = (d1 - d3) / (d1 + d3)

        similarity = f1 + f2 - 2
        unSimilarity = -similarity

        return -unSimilarity



class Phone_Net(Chain):

    def __init__(self, n_in, n_out):
        super(Phone_Net, self).__init__()
        with self.init_scope():
            self.l1 = L.Linear(None, n_in)
            self.l2 = L.Linear(n_in, n_in)
            self.l3 = L.Linear(n_in, n_out)

    def __call__(self, x):
        h = F.relu(self.l1(x))
        h = F.relu(self.l2(h))
        h = self.l3(h)
        h = F.softmax(h)
        return h


if __name__ == '__main__':

    ens = NetEnsemble(r"/home/kamenev/speechProject/")
    ens.train(10)

