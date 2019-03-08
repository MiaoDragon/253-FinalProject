from cnn import ResNet
class BaselineNet(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(BaselineNet, self).__init__()
        # cnn layer for state extraction
        self.cnn = ResNet(state_dim)
        # three layer MLP
        self.fc1 = nn.Linear(state_dim, 32)
        self.fc2 = nn.Linear(32, 64)
        self.fc3 = nn.Linear(64, action_dim)
        self.opt = optim.Adam(self.parameters(), lr=1e-3)
    def forward(self, s):
        s = F.relu(self.fc1(s))
        s = F.relu(self.fc2(s))
        s = self.fc3(s)  # hamiltonina
        s = F.softmax(s)
        return s # probability distribution
