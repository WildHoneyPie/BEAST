import torch 
from torch import nn 
from Streaming_encoder_layer import ContextualBlockTransformerEncoder


class TransformerModel(nn.Module):
    def __init__(self, ntoken=2, dmodel=256, nhead=2, d_hid=2048, nlayers=9, norm_first=True, dropout=.1, left_size=256, center_size=16, right_size=16):
        super(TransformerModel, self).__init__()
        self.nhead = nhead
        self.nlayers = nlayers
        self.head_dim = dmodel // nhead
        assert self.head_dim * nhead == dmodel, "embed_dim must be divisible by num_heads"

        #self.Er = nn.Parameter(torch.randn(nlayers, nhead, self.head_dim, attn_len))

        # self.pos_enc = StreamPositionalEncoding(dmodel, dropout)

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=(3, 3), stride=1, padding=(2, 0))#126
        #self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=(3, 3), stride=1, padding=(1, 0))#79
        self.maxpool1 = nn.MaxPool2d(kernel_size=(1, 3), stride=(1, 3))#26
        self.dropout1 = nn.Dropout(p=dropout)
        
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(1, 12), stride=1, padding=(0, 0))#31
        #self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(1, 12), stride=1, padding=(0, 0))#15
        self.maxpool2 = nn.MaxPool2d(kernel_size=(1, 3), stride=(1, 3))#5
        self.dropout2 = nn.Dropout(p=dropout)
        
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=dmodel, kernel_size=(2, 6), stride=1, padding=(1, 0))#5
        #self.conv3 = nn.Conv2d(in_channels=64, out_channels=dmodel, kernel_size=(3, 3), stride=1, padding=(1, 0))#3
        self.maxpool3 = nn.MaxPool2d(kernel_size=(1, 3), stride=(1, 3))#1
        self.dropout3 = nn.Dropout(p=dropout)
        
        self.encoder = ContextualBlockTransformerEncoder(input_size= dmodel, 
                                                        output_size = dmodel,
                                                        attention_heads=nhead, 
                                                        linear_units = d_hid,
                                                        num_blocks = nlayers,
                                                        dropout_rate=dropout,
                                                        left_size = left_size,
                                                        center_size = center_size,
                                                        right_size = right_size
                                                        )
        self.out_linear = nn.Linear(dmodel, ntoken)

        self.dropout_t = nn.Dropout(p=.5)
        self.out_linear_t = nn.Linear(dmodel, 300)
        

    def forward(self, x):
        #x: (batch, time, dmodel), FloatTensor
        x = x.unsqueeze(1)  #(batch, channel, time, dmodel)
        x = self.conv1(x)
        x = x[:,:,:-2,:]
        x = self.maxpool1(x)
        x = torch.relu(x)
        x = self.dropout1(x)

        x = self.conv2(x)
        x = self.maxpool2(x)
        x = torch.relu(x)
        x = self.dropout2(x)

        x = self.conv3(x)
        x = x[:,:,:-1,:]
        x = self.maxpool3(x)
        x = torch.relu(x)
        x = self.dropout3(x)    #(batch, channel, time, 1)
        x = x.transpose(1, 3).squeeze(1).contiguous()    #(batch, time, channel=dmodel)

        
        x, _, t = self.encoder(x)

        x = torch.relu(x)
        x = self.out_linear(x)

        t = torch.relu(t)
        t = self.dropout_t(t)
        t = t.mean(dim=1) #(batch, dmodel)
        t = self.out_linear_t(t)
        return x, t

