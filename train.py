# import libraries
import numpy as np
import glob, os
import torch
import arff
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import random


def getData(filetype, location):
    os.chdir('/Users/jonahbolotin/Documents/Senior/Spring/Linguistics/emotion-speech-recognizer')
    if filetype == 'csv':
        truth_values, file_names = get_file_information(location)
        matrix = process_data(file_names)
    elif filetype == 'arff':
        class_mapper = {'anger': 0,
                'sadness': 1,
                'neutral': 2,
                'happiness': 3,
                'despair': 4}
        data = arff.load(open(location, 'rb'))
        matrix = np.array(data['data'])
        truth_values = matrix[:,0]
        for i,value in enumerate(truth_values):
            truth_values[i] = class_mapper[value]
        matrix = np.delete(matrix, 0, 1)
    return matrix, truth_values

def numpy_to_torch(array):
    return torch.from_numpy(array)

#sample is either '/train' or '/test'
def get_file_information(sample):
    class_mapper = {'anger': 0,
                    'sadness': 1,
                    'neutral': 2,
                    'happiness': 3,
                    'despair': 4}
    truth_values = []
    file_names = []
    base_directory = os.getcwd() + sample
    subdirectories = ['anger', 'despair', 'happiness', 'neutral', 'sadness']
    for directory in subdirectories:
        os.chdir(base_directory + '/' + directory)
        for file_name in glob.glob("*.csv"):
            truth_values.append(class_mapper[directory])
            file_names.append(os.getcwd() + '/' + file_name)
    return truth_values, file_names

# Load CSV and return np.ndarray
def csv_to_numpy(filename):
    raw_data = open(filename, 'rt')
    data = np.loadtxt(raw_data, delimiter=",",skiprows=1)
    data = np.delete(data, np.s_[:2], 1)
    return data

# Need standard length for file for matrix
# Returns flattened array for big matrix
def format_np_and_flatten(data):
    num_rows = data.shape[0]
    if num_rows > 140: # delete rows
        data = np.delete(data, slice(140,num_rows), axis=0)
    else: # add rows of zeros
        num_rows_to_add = 140 - num_rows
        rows = np.zeros((num_rows_to_add, data.shape[1]))
        data = np.append(data, rows, axis=0)
    return data

def process_data(file_names):
    training_matrix = []
    for filename in file_names:
        data = csv_to_numpy(filename)
        data = format_np_and_flatten(data)
        training_matrix.append(data)
    return training_matrix

class MyDataset(Dataset):
    def __init__(self, x, y, transform=None):
        """
        Args:
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.x = torch.tensor(x, dtype=torch.float)
        self.y = torch.tensor(y)
        self.transform = transform

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        sample = (self.x[idx], self.y[idx])
        return sample

# define model class
class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        # "nn.Linear" performs the last step of prediction, where it maps the hidden layer to # classes
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        # Set initial hidden and cell states
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)

        # Forward propagate LSTM
        out, _ = self.lstm(x, (h0, c0))  # out: tensor of shape (batch_size, seq_length, hidden_size)

        # Decode the hidden state of the last time step
        out = self.fc(out[:, -1, :])
        return out

def main():
    trainX, trainY = getData('csv', '/train')
    testX, testY = getData('csv', '/test')
    train_loader = DataLoader(MyDataset(trainX, trainY), batch_size=32, shuffle=True)
    test_loader = DataLoader(MyDataset(testX, testY), batch_size=32, shuffle=True)
    time_steps = 2
    batch_size = 590 # this is # files you're loading
    input_size = 13 # this is # of features extracted
    nclasses = 5

    # defines loss function
    criterion = nn.CrossEntropyLoss()
    best = 0
    best_lr = -1
    best_hidden = 0

    while True:
        print best, best_lr, best_hidden
        learning_rate = random.uniform(0.18, 0.23)



        hidden_size = random.randint(90, 110)
        # set up model
        model = RNN(input_size, hidden_size, 1, nclasses)
        # defines optimization function (this is using stochastic gradient descent)
        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate) #, momentum=0.9
        top, iteration = train(time_steps, batch_size, input_size, hidden_size, nclasses, model, optimizer, criterion, train_loader, test_loader, trainX, trainY, testX, testY)
        if top > best:
            print 'Found new best!'
            best = top
            best_lr = learning_rate
            best_hidden = hidden_size

def train(time_steps, batch_size, input_size, hidden_size, nclasses, model, optimizer, criterion, train_loader, test_loader, trainX, trainY, testX, testY):
    print("Training")
    best = 0
    iteration = 0
    for epoch in range(40):
        total_loss = 0
        correct = 0
        model.train()
        for batch_idx, (data, target) in enumerate(train_loader):
    #         x = torch.tensor(trainX[i], dtype=torch.float).unsqueeze(0)
    #         y = torch.tensor(trainY[i]).unsqueeze(0)
            # Step 3. Run our forward pass.
            outputs = model(data)
            _, predicted = torch.max(outputs.data, 1)
            correct += (predicted == target).sum().item()

            # Step 4. Compute the loss, gradients, and update the parameters by
            loss = criterion(outputs, target)
            total_loss += loss.item()

            # clear gradient for next step of training
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # if batch_idx % 10 == 0:
            #     print("Step %d / %d, Loss: %f" % (batch_idx, len(train_loader), loss.item()))
        total_loss /= float(len(train_loader))
        correct /= float(len(trainX))
        print("Epoch #%d, Loss: %f, Accuracy: %f" % (epoch, total_loss, correct))

        model.eval()
        # Test model
        with torch.no_grad():
            correct = 0
            total = 0
            for batch_idx, (data, target) in enumerate(test_loader):
        #         x = torch.tensor(testX[i], dtype=torch.float).unsqueeze(0)
        #         y = torch.tensor(testY[i]).unsqueeze(0)

                outputs = model(data)
                _, predicted = torch.max(outputs.data, 1)
                correct += (predicted == target).sum().item()
                total += len(target)

            print('Test Accuracy of the model: {} %'.format(100 * correct / total))
            if 100 * correct / total > best:
                best = 100 * correct / total
                iteration = epoch
    return best, iteration


main()
