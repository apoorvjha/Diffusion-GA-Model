from torch.nn import Module, Conv2d, MaxPool2d, ReLU, Dropout, ConvTranspose2d
from torch import cat as concatenate
from torch import rand
from torch.nn.functional import pad
from torch.cuda import is_available
from torch.optim import Adam
from torch import sum as torch_sum
from time import time

class U_NET(Module):
    '''
    Description : TBU
    '''
    
    def __init__(self, init_channel = 16, p=0.2):
        super(U_NET, self).__init__()
        self.conv1 = Conv2d(in_channels = 1, out_channels = init_channel, kernel_size = (3,3), padding='same')
        self.conv1_1 = Conv2d(in_channels = init_channel, out_channels = init_channel, kernel_size = (3,3), padding='same')
        self.pool = MaxPool2d(kernel_size = (2,2))
        self.relu = ReLU()
        self.dropout = Dropout(p)
        self.conv2 = Conv2d(in_channels = init_channel, out_channels = init_channel * 2, kernel_size = (3,3), padding='same')
        self.conv2_1 = Conv2d(in_channels = init_channel * 2, out_channels = init_channel * 2, kernel_size = (3,3), padding='same')
        self.conv3 = Conv2d(in_channels = init_channel * 2, out_channels = init_channel * 4, kernel_size = (3,3), padding='same')
        self.conv3_1 = Conv2d(in_channels = init_channel * 4, out_channels = init_channel * 4, kernel_size = (3,3), padding='same')
        self.conv4 = Conv2d(in_channels = init_channel * 4, out_channels = init_channel * 8, kernel_size = (3,3),padding='same')
        self.conv4_1 = Conv2d(in_channels = init_channel * 8, out_channels = init_channel * 8, kernel_size = (3,3), padding='same')
        self.conv5 = Conv2d(in_channels = init_channel * 8, out_channels = init_channel * 16, kernel_size = (3,3), padding='same')
        self.conv6 = Conv2d(in_channels = init_channel * 16, out_channels = init_channel * 16, kernel_size = (3,3), padding='same')
        self.deconv4 = ConvTranspose2d(in_channels = init_channel * 16, out_channels = init_channel * 8, kernel_size = (3,3), stride = (2,2))#, padding = (3,3), dilation = (3,3), output_padding = (2,2))
        self.deconv3 = ConvTranspose2d(in_channels = init_channel * 8, out_channels = init_channel * 4, kernel_size = (3,3), stride = (2,2))#, padding = (3,3), dilation = (3,3), output_padding = (2,2))
        self.deconv2 = ConvTranspose2d(in_channels = init_channel * 4, out_channels = init_channel * 2, kernel_size = (3,3), stride = (2,2), padding = (1,1), output_padding = (1,1))
        self.deconv1 = ConvTranspose2d(in_channels = init_channel * 2, out_channels = init_channel, kernel_size = (3,3), stride = (2,2), padding = (1,1), output_padding = (1,1)) 
        self.conv7 = Conv2d(in_channels = init_channel, out_channels = 1, kernel_size = (1,1), padding = 'same')
        self.conv4_2 = Conv2d(in_channels = init_channel * 16, out_channels = init_channel * 8, kernel_size = (3,3), padding = 'same')
        self.conv3_2 = Conv2d(in_channels = init_channel * 8, out_channels = init_channel * 4, kernel_size = (3,3), padding = 'same')
        self.conv2_2 = Conv2d(in_channels = init_channel * 4, out_channels = init_channel * 2, kernel_size = (3,3), padding = 'same')
        self.conv1_2 = Conv2d(in_channels = init_channel * 2, out_channels = init_channel, kernel_size = (3,3), padding = 'same')

    def forward(self, x):
        
        conv1 = self.relu(self.conv1(x))
        conv1 = self.relu(self.conv1_1(conv1))
        pool1 = self.pool(conv1)
        pool1 = self.dropout(pool1)


        conv2 = self.relu(self.conv2(pool1))
        conv2 = self.relu(self.conv2_1(conv2))
        pool2 = self.pool(conv2)
        pool2 = self.dropout(pool2)

 
        conv3 = self.relu(self.conv3(pool2))
        conv3 = self.relu(self.conv3_1(conv3))
        pool3 = self.pool(conv3)
        pool3 = self.dropout(pool3)


        conv4 = self.relu(self.conv4(pool3))
        conv4 = self.relu(self.conv4_1(conv4)) 
        pool4 = self.pool(conv4)
        pool4 = self.dropout(pool4)

        conv5 = self.relu(self.conv5(pool4))
        conv6 = self.relu(self.conv6(conv5))

        deconv4 = self.deconv4(conv6)
        deconv4 = concatenate((deconv4, conv4), dim=1) # CConcatenate in channel dimension.
        deconv4 = self.dropout(deconv4)
        deconv4 = self.relu(self.conv4_2(deconv4))
        deconv4 = self.relu(self.conv4_1(deconv4))

        deconv3 = self.deconv3(deconv4)
        deconv3 = concatenate((deconv3, conv3), dim=1) # CConcatenate in channel dimension.
        deconv3 = self.dropout(deconv3)
        deconv3 = self.relu(self.conv3_2(deconv3))
        deconv3 = self.relu(self.conv3_1(deconv3))
        
        deconv2 = self.deconv2(deconv3)
        deconv2 = concatenate((deconv2, conv2), dim=1) # CConcatenate in channel dimension.
        deconv2 = self.dropout(deconv2)
        deconv2 = self.relu(self.conv2_2(deconv2))
        deconv2 = self.relu(self.conv2_1(deconv2))

        deconv1 = self.deconv1(deconv2)
        deconv1 = concatenate((deconv1, conv1), dim=1) # CConcatenate in channel dimension.
        deconv1 = self.dropout(deconv1)
        deconv1 = self.relu(self.conv1_2(deconv1))
        deconv1 = self.relu(self.conv1_1(deconv1))

        output = self.relu(self.conv7(deconv1))

        return output

class Main:
    def __init__(self, learning_rate = 1e-3):
        if is_available:
            device = 'cuda'
        else:
            device = 'cpu'
        self.model = U_NET()
        self.optimizer = Adam(self.model.parameters(), lr = learning_rate)
    def criterion(self, Y_true, Y_predicted):
        return torch_sum((Y_true - Y_predicted) ** 2)
    def fit(self, X, Y, epochs = 10, batch_size = 10):
        n_batch = X.size(0) // batch_size
        # print(f"Number of batches = {n_batch}")
        for epoch in range(epochs):
            start = time()
            loss_agg = 0
            for batch in range(n_batch - 1):
                self.optimizer.zero_grad()
                predictions = self.model(X[batch * batch_size : (batch + 1) * batch_size])
                loss = self.criterion(Y[batch * batch_size : (batch + 1) * batch_size], predictions)
                loss_agg += loss.item() / batch_size
                loss.backward()
                self.optimizer.step()
            print(f"EPOCH[{epoch+1}] => Loss = {loss_agg / n_batch} ; Took {time() - start} seconds.")
    def predict(X):
        return self.model(X)

if __name__ == "__main__":
    driver = Main()
    X_train = rand(100, 1, 572, 572)
    driver.fit(X_train, X_train)
    X_test = rand(10, 1, 572, 572)
    predictions = driver.predict(X_test)
    print(driver.criterion(X_test, predictions))