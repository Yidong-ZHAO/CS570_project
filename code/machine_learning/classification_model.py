
import numpy as np
import matplotlib.pyplot as plt


# Machine learning model
import torch
import torch.nn as nn
import torch.nn.functional as F



# Load data
sand_data = np.loadtxt('sand_data.csv', delimiter=' ')
# this data saves x, y position

print(sand_data.shape) # (1600, 2)

print('min x: ', np.min(sand_data[:,0])) # 0.16908228397369385
print('max x: ', np.max(sand_data[:,0])) # 1.8317128419876099
print('min y: ', np.min(sand_data[:,1])) # 0.038582753390073776
print('max y: ', np.max(sand_data[:,1])) # 0.3418230712413788





# TODO # MARK (Yidong):
# generate data

x_data = sand_data
y_data = np.ones(len(x_data))

# From numpy to torch tensor
x_data_tensor = torch.FloatTensor(x_data).reshape(-1, 2)
y_data_tensor = torch.FloatTensor(y_data).reshape(-1, 1)







# MLP
class MLPNet (torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(MLPNet, self).__init__()

        self.linear1 = torch.nn.Linear(input_dim, hidden_dim)
        self.sigmoid = torch.nn.Sigmoid()
        self.relu = torch.nn.ReLU()

        self.linear2 = torch.nn.Linear(hidden_dim, output_dim)


    # Forward propagation function
    def forward(self, x):
        # The first layer
        out = self.linear1(x)
        out = self.relu(out)

        # The second layer
        out = self.linear2(out)
        out = self.sigmoid(out)

        return out



# Training function
def train(model, training_x, training_y, optimizer, loss_fn):
	model.train()
	loss = 0

	optimizer.zero_grad()

	out = model(training_x)

	loss = loss_fn(out, training_y)

	loss.backward()
	optimizer.step()

	return loss.item()




# Parameters
args = {
	'input_dim': 2,
	'hidden_dim': 5,
	'output_dim': 1,
	'epochs': 200,
	'lr': 0.01
}
args



model = MLPNet(args['input_dim'],
			   args['hidden_dim'],
			   args['output_dim'])



# MARK (Yidong):
# model parameters initialization




optimizer = torch.optim.Adam(model.parameters(), lr=args['lr'])
loss_fn = F.binary_cross_entropy

for epoch in range(1, 1 + args["epochs"]):
	loss = train(model, x_data_tensor, y_data_tensor, optimizer, loss_fn)
	print(f'Epoch: {epoch:02d}, '
		  f'Loss: {loss:.4f}')


model.eval()
y_predict = model(x_data_tensor)
torch.set_printoptions(profile="full")
print(y_predict) # You will see all outputs are almost 1 (of course because here we only have positive data..)
torch.set_printoptions(profile="default")

