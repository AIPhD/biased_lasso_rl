import torch
from torch import nn
from cartpole_project import config_cartpole as cc
from maze_project import config_maze as cm


class CartpoleValueNetwork(nn.Module):
    '''Fully connected network for CartPole'''

    def __init__(self):

        super(CartpoleValueNetwork, self).__init__()

        self.layer1 = nn.Linear(cc.CART_INPUT, 64)
        self.layer2 = nn.Linear(64, 64)
        self.layern = nn.Linear(64, 1)


    def forward(self, x_input):
        '''Calculates output of the network given data x_input'''
        x_input = nn.functional.relu(self.layer1(x_input))
        x_input = nn.functional.relu(self.layer2(x_input))
        return self.layern(x_input)

    def init_weights_to_zero(self):
        self.layer1.weight.data = torch.zeros(cc.HIDDEN_NODE_COUNT, cc.CART_INPUT).to(cc.DEVICE)
        self.layer1.bias.data = torch.zeros(cc.HIDDEN_NODE_COUNT).to(cc.DEVICE)
        self.layer2.weight.data = torch.zeros(cc.HIDDEN_NODE_COUNT, cc.HIDDEN_NODE_COUNT).to(cc.DEVICE)
        self.layer2.bias.data = torch.zeros(cc.HIDDEN_NODE_COUNT).to(cc.DEVICE)
        self.layern.weight.data = torch.zeros(1, cc.HIDDEN_NODE_COUNT).to(cc.DEVICE)
        self.layern.bias.data = torch.zeros(1).to(cc.DEVICE)

class CartpolePolicyNetwork(nn.Module):
    '''Fully connected network for CartPole Policy'''

    def __init__(self):

        super(CartpoleValueNetwork, self).__init__()

        self.layer1 = nn.Linear(cc.CART_INPUT, 64)
        self.layer2 = nn.Linear(64, 64)
        self.layern = nn.Linear(64, 2)


    def forward(self, x_input):
        '''Calculates output of the network given data x_input'''
        x_input = nn.functional.relu(self.layer1(x_input))
        x_input = nn.functional.relu(self.layer2(x_input))
        return self.layern(x_input)

    def init_weights_to_zero(self):
        self.layer1.weight.data = torch.zeros(cc.HIDDEN_NODE_COUNT, cc.CART_INPUT).to(cc.DEVICE)
        self.layer1.bias.data = torch.zeros(cc.HIDDEN_NODE_COUNT).to(cc.DEVICE)
        self.layer2.weight.data = torch.zeros(cc.HIDDEN_NODE_COUNT, cc.HIDDEN_NODE_COUNT).to(cc.DEVICE)
        self.layer2.bias.data = torch.zeros(cc.HIDDEN_NODE_COUNT).to(cc.DEVICE)
        self.layern.weight.data = torch.zeros(1, cc.HIDDEN_NODE_COUNT).to(cc.DEVICE)
        self.layern.bias.data = torch.zeros(1).to(cc.DEVICE)

class MazeQNetwork(nn.Module):
    '''Fully Connected Value Network for mazes.'''

    def __init__(self):

        super().__init__()

        self.layer1 = nn.Linear(cm.GRID_INPUT, cm.HIDDEN_NODE_COUNT)
        self.layer2 = nn.Linear(cm.HIDDEN_NODE_COUNT, cm.HIDDEN_NODE_COUNT)
        self.layer3 = nn.Linear(cm.HIDDEN_NODE_COUNT, cm.HIDDEN_NODE_COUNT)
        self.layer4 = nn.Linear(cm.HIDDEN_NODE_COUNT, cm.HIDDEN_NODE_COUNT)
        self.layer5 = nn.Linear(cm.HIDDEN_NODE_COUNT, cm.HIDDEN_NODE_COUNT)
        self.layer6 = nn.Linear(cm.HIDDEN_NODE_COUNT, cm.GRID_OUTPUT)

    def forward(self, x_input):
        '''Calculates output of the network given data x_input'''
        x_input = nn.functional.relu(self.layer1(x_input))
        x_input = nn.functional.relu(self.layer2(x_input))
        x_input = nn.functional.relu(self.layer3(x_input))
        x_input = nn.functional.relu(self.layer4(x_input))
        x_input = nn.functional.relu(self.layer5(x_input))
        return self.layer6(x_input)

    def init_weights_to_zero(self):
        self.layer1.weight.data = torch.zeros(cm.HIDDEN_NODE_COUNT, cm.GRID_INPUT).to(cm.DEVICE)
        self.layer1.bias.data = torch.zeros(cm.HIDDEN_NODE_COUNT).to(cm.DEVICE)
        self.layer2.weight.data = torch.zeros(cm.HIDDEN_NODE_COUNT, cm.HIDDEN_NODE_COUNT).to(cm.DEVICE)
        self.layer2.bias.data = torch.zeros(cm.HIDDEN_NODE_COUNT).to(cm.DEVICE)
        self.layer3.weight.data = torch.zeros(cm.HIDDEN_NODE_COUNT, cm.HIDDEN_NODE_COUNT).to(cm.DEVICE)
        self.layer3.bias.data = torch.zeros(cm.HIDDEN_NODE_COUNT).to(cm.DEVICE)
        self.layer4.weight.data = torch.zeros(cm.HIDDEN_NODE_COUNT, cm.HIDDEN_NODE_COUNT).to(cm.DEVICE)
        self.layer4.bias.data = torch.zeros(cm.HIDDEN_NODE_COUNT).to(cm.DEVICE)
        self.layer5.weight.data = torch.zeros(cm.HIDDEN_NODE_COUNT, cm.HIDDEN_NODE_COUNT).to(cm.DEVICE)
        self.layer5.bias.data = torch.zeros(cm.HIDDEN_NODE_COUNT).to(cm.DEVICE)
        self.layer6.weight.data = torch.zeros(cm.GRID_OUTPUT, cm.HIDDEN_NODE_COUNT).to(cm.DEVICE)
        self.layer6.bias.data = torch.zeros(cm.GRID_OUTPUT).to(cm.DEVICE)

class MazePolicyNetwork(nn.Module):
    '''Fully Connected Policy Network with residual connections for mazes.'''

    def __init__(self):

        super().__init__()

        self.firstlayer = nn.Linear(cm.GRID_INPUT, cm.HIDDEN_NODE_COUNT)
        self.secondlayer = nn.Linear(cm.HIDDEN_NODE_COUNT, cm.HIDDEN_NODE_COUNT)
        self.thirdlayer = nn.Linear(cm.HIDDEN_NODE_COUNT, cm.HIDDEN_NODE_COUNT)
        self.fourthlayer = nn.Linear(cm.HIDDEN_NODE_COUNT, cm.HIDDEN_NODE_COUNT)
        self.fifthlayer = nn.Linear(cm.HIDDEN_NODE_COUNT, cm.HIDDEN_NODE_COUNT)
        self.sixthlayer = nn.Linear(cm.HIDDEN_NODE_COUNT, cm.GRID_OUTPUT)
        self.layer_res = nn.Linear(cm.GRID_INPUT, cm.GRID_OUTPUT)
        self.layer_res.bias.data = torch.zeros(cm.GRID_OUTPUT).to(cm.DEVICE)
        self.layer_res.bias.data.requires_grad = False

    def forward(self, x_input):
        '''Calculates output of the network given data x_input'''
        x_input_pass = nn.functional.relu(self.firstlayer(x_input))
        x_input_pass = nn.functional.relu(self.secondlayer(x_input_pass))
        x_input_pass = nn.functional.relu(self.thirdlayer(x_input_pass))
        x_input_pass = nn.functional.relu(self.fourthlayer(x_input_pass))
        x_input_pass = nn.functional.relu(self.fifthlayer(x_input_pass))
        return self.sixthlayer(x_input_pass) + self.layer_res(x_input)

    def init_weights_to_zero(self):
        self.firstlayer.weight.data = torch.zeros(cm.HIDDEN_NODE_COUNT, cm.GRID_INPUT).to(cm.DEVICE)
        self.firstlayer.bias.data = torch.zeros(cm.HIDDEN_NODE_COUNT).to(cm.DEVICE)
        self.secondlayer.weight.data = torch.zeros(cm.HIDDEN_NODE_COUNT, cm.HIDDEN_NODE_COUNT).to(cm.DEVICE)
        self.secondlayer.bias.data = torch.zeros(cm.HIDDEN_NODE_COUNT).to(cm.DEVICE)
        self.thirdlayer.weight.data = torch.zeros(cm.HIDDEN_NODE_COUNT, cm.HIDDEN_NODE_COUNT).to(cm.DEVICE)
        self.thirdlayer.bias.data = torch.zeros(cm.HIDDEN_NODE_COUNT).to(cm.DEVICE)
        self.fourthlayer.weight.data = torch.zeros(cm.HIDDEN_NODE_COUNT, cm.HIDDEN_NODE_COUNT).to(cm.DEVICE)
        self.fourthlayer.bias.data = torch.zeros(cm.HIDDEN_NODE_COUNT).to(cm.DEVICE)
        self.fifthlayer.weight.data = torch.zeros(cm.HIDDEN_NODE_COUNT, cm.HIDDEN_NODE_COUNT).to(cm.DEVICE)
        self.fifthlayer.bias.data = torch.zeros(cm.HIDDEN_NODE_COUNT).to(cm.DEVICE)
        self.sixthlayer.weight.data = torch.zeros(cm.GRID_OUTPUT, cm.HIDDEN_NODE_COUNT).to(cm.DEVICE)
        self.sixthlayer.bias.data = torch.zeros(cm.GRID_OUTPUT).to(cm.DEVICE)
        self.layer_res.weight.data = torch.zeros(cm.GRID_OUTPUT, cm.GRID_INPUT).to(cm.DEVICE)
    
    def init_head(self):
        self.sixthlayer.weight.data = torch.zeros(cm.GRID_OUTPUT, cm.HIDDEN_NODE_COUNT).to(cm.DEVICE)
        self.sixthlayer.bias.data = torch.zeros(cm.GRID_OUTPUT).to(cm.DEVICE)

class MazeValueNetwork(nn.Module):
    '''A2C Value Network for mazes.'''

    def __init__(self):

        super().__init__()

        self.layer1 = nn.Linear(cm.GRID_INPUT, cm.HIDDEN_NODE_COUNT)
        self.layer2 = nn.Linear(cm.HIDDEN_NODE_COUNT, cm.HIDDEN_NODE_COUNT)
        self.layer3 = nn.Linear(cm.HIDDEN_NODE_COUNT, cm.HIDDEN_NODE_COUNT)
        self.layer4 = nn.Linear(cm.HIDDEN_NODE_COUNT, cm.HIDDEN_NODE_COUNT)
        self.layer5 = nn.Linear(cm.HIDDEN_NODE_COUNT, cm.HIDDEN_NODE_COUNT)
        self.layer6 = nn.Linear(cm.HIDDEN_NODE_COUNT, 1)
        self.layer_res = nn.Linear(cm.GRID_INPUT, 1)
        self.layer_res.bias.data = torch.zeros(1).to(cm.DEVICE)
        self.layer_res.bias.data.requires_grad = False

    def forward(self, x_input):
        '''Calculates output of the network given data x_input'''
        x_input_pass = nn.functional.relu(self.layer1(x_input))
        x_input_pass = nn.functional.relu(self.layer2(x_input_pass))
        x_input_pass = nn.functional.relu(self.layer3(x_input_pass))
        x_input_pass = nn.functional.relu(self.layer4(x_input_pass))
        x_input_pass = nn.functional.relu(self.layer5(x_input_pass))
        return self.layer6(x_input_pass) + self.layer_res(x_input)

    def init_weights_to_zero(self):
        self.layer1.weight.data = torch.zeros(cm.HIDDEN_NODE_COUNT, cm.GRID_INPUT).to(cm.DEVICE)
        self.layer1.bias.data = torch.zeros(cm.HIDDEN_NODE_COUNT).to(cm.DEVICE)
        self.layer2.weight.data = torch.zeros(cm.HIDDEN_NODE_COUNT, cm.HIDDEN_NODE_COUNT).to(cm.DEVICE)
        self.layer2.bias.data = torch.zeros(cm.HIDDEN_NODE_COUNT).to(cm.DEVICE)
        self.layer3.weight.data = torch.zeros(cm.HIDDEN_NODE_COUNT, cm.HIDDEN_NODE_COUNT).to(cm.DEVICE)
        self.layer3.bias.data = torch.zeros(cm.HIDDEN_NODE_COUNT).to(cm.DEVICE)
        self.layer4.weight.data = torch.zeros(cm.HIDDEN_NODE_COUNT, cm.HIDDEN_NODE_COUNT).to(cm.DEVICE)
        self.layer4.bias.data = torch.zeros(cm.HIDDEN_NODE_COUNT).to(cm.DEVICE)
        self.layer5.weight.data = torch.zeros(cm.HIDDEN_NODE_COUNT, cm.HIDDEN_NODE_COUNT).to(cm.DEVICE)
        self.layer5.bias.data = torch.zeros(cm.HIDDEN_NODE_COUNT).to(cm.DEVICE)
        self.layer6.weight.data = torch.zeros(1, cm.HIDDEN_NODE_COUNT).to(cm.DEVICE)
        self.layer6.bias.data = torch.zeros(1).to(cm.DEVICE)
        self.layer_res.weight.data = torch.zeros(1, cm.GRID_INPUT).to(cm.DEVICE)

class AtariQNetwork(nn.Module):
    '''Conv-Net for Atari game environments'''

    def __init__(self, num_actions):

        super().__init__()
        self.stacked_layers = nn.Sequential(nn.Conv2d(4, 32, kernel_size=8, stride=4),
                                            nn.ReLU(),
                                            nn.Conv2d(32, 64, kernel_size=4, stride=2),
                                            nn.ReLU(),
                                            nn.Conv2d(64, 64, kernel_size=3, stride=1),
                                            nn.ReLU(),
                                            nn.Flatten(start_dim=1),
                                            nn.Linear(3136, 1600),
                                            nn.ReLU(),
                                            nn.Linear(1600, 256),
                                            nn.ReLU(),
                                            nn.Linear(256, num_actions))

    def forward(self, x_input):
        '''Calculates output of the network given data x_input'''

        y_output = self.stacked_layers(x_input)
        return y_output

class AtariPolicyNetwork(nn.Module):
    '''Policy-Net for Atari game environments using A3C'''

    def __init__(self, num_actions):

        super().__init__()
        self.firstlayer = nn.Conv2d(4, 32, kernel_size=8, stride=4)
        self.secondlayer = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.thirdlayer = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.fourthlayer = nn.Linear(3136, 1600)
        self.fifthlayer = nn.Linear(1600, 256)
        self.sixthlayer = nn.Linear(256, num_actions)
        self.layer_res2 = nn.Linear(cm.GRID_INPUT, cm.GRID_OUTPUT)
        self.num_actions = num_actions
        self.transfer_kernels = ['firstlayer.weight',
                                 'secondlayer.weight',
                                 'thirdlayer.weight']
        self.transfer_bias = ['firstlayer.bias',
                              'secondlayer.weight',
                              'thirdlayer.bias',
                              'fourthlayer.bias']
        self.transfer_linear = ['fourthlayer.weight']

    def forward(self, x_input):
        '''Calculates output of the network given data x_input'''

        x_input = self.firstlayer(x_input)
        x_input = nn.functional.relu(x_input)
        x_input = self.secondlayer(x_input)
        x_input = nn.functional.relu(x_input)
        x_input = self.thirdlayer(x_input)
        x_input = nn.functional.relu(x_input)
        x_input = torch.flatten(x_input, start_dim=1)
        x_input = self.fourthlayer(x_input)
        x_input = nn.functional.relu(x_input)
        x_input = self.fifthlayer(x_input)
        x_input = nn.functional.relu(x_input)
        y_output = self.sixthlayer(x_input)
        return y_output
    
    def init_head(self):
        self.fifthlayer.weight.data = torch.zeros(1600, 256).to(cm.DEVICE)
        self.fifthlayer.bias.data = torch.zeros(256).to(cm.DEVICE)
        self.sixthlayer.weight.data = torch.zeros(256, self.num_actions).to(cm.DEVICE)
        self.sixthlayer.bias.data = torch.zeros(self.num_actions).to(cm.DEVICE)
    
    def copy_layers(self, source_model_dict):
        for key in source_model_dict:
            try:
                self.state_dict()[key] = source_model_dict[key]
            except:
                print('incompatible_layer')

class AtariValueNetwork(nn.Module):
    '''Value-Net for Atari game environments using A3C'''

    def __init__(self):

        super().__init__()
        self.stacked_layers = nn.Sequential(nn.Conv2d(4, 32, kernel_size=8, stride=4),
                                            nn.ReLU(),
                                            nn.Conv2d(32, 64, kernel_size=4, stride=2),
                                            nn.ReLU(),
                                            nn.Conv2d(64, 64, kernel_size=3, stride=1),
                                            nn.ReLU(),
                                            nn.Flatten(start_dim=1),
                                            nn.Linear(3136, 1600),
                                            nn.ReLU(),
                                            nn.Linear(1600, 256),
                                            nn.ReLU(),
                                            nn.Linear(256, 1))

    def forward(self, x_input):
        '''Calculates output of the network given data x_input'''

        y_output = self.stacked_layers(x_input)
        return y_output
