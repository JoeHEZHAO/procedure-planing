import sys
import os
import torch
from torch import nn
from os import path as osp
from collections import OrderedDict 
sys.path.append(osp.dirname(osp.dirname(osp.abspath(__file__))))    # add parent dir
from models.s3dg import S3D
from paths import S3D_PATH
import torch.nn.functional as F
import torchvision.models as models

device = "cuda" if torch.cuda.is_available() else "cpu"

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        if not os.path.exists(osp.join(S3D_PATH, 's3d_dict.npy')):
            os.system(f"wget wget -P {osp.join(S3D_PATH)} https://www.rocq.inria.fr/cluster-willow/amiech/howto100m/s3d_dict.npy")
        if not os.path.exists(osp.join(S3D_PATH, 's3d_howto100m.pth')):
            os.system(f"wget wget -P {osp.join(S3D_PATH)} https://www.rocq.inria.fr/cluster-willow/amiech/howto100m/s3d_howto100m.pth")

        self.net = S3D(osp.join(S3D_PATH, 's3d_dict.npy'))
        state_dict = torch.load(osp.join(S3D_PATH, 's3d_howto100m.pth'))
        self.net.load_state_dict(state_dict)
        self.net.to(device)
        self.net.eval()
        # self.num_frames = 30 #32
        self.num_frames = 16

    @torch.no_grad()
    def retrieve(self, texts, videos):
        # video frames have to be normalized in [0, 1]
        video_descriptors = torch.cat([self.net(v[None, ...])['video_embedding'] for v in videos], 0)
        text_descriptors = self.net.text_module(texts)['text_embedding']
        scores = text_descriptors @ video_descriptors.t()

        decr_sim_inds = torch.argsort(scores, descending=True, dim=1)
        outs = []
        for i in range(len(texts)):
            sorted_videos = [{'video_ind': j, 'score': scores[i, j]} for j in decr_sim_inds[i]]
            outs.append(sorted_videos)
        return outs

    @torch.no_grad()
    def embed_full_video(self, frames):
        # assuming the video is at 10fps and that we take 32 frames
        # frames is a tensor of size [T, W, H, 3]
        T, W, H, _ = frames.shape
        frames = frames.permute(3, 0, 1, 2)
        N_chunks = T // self.num_frames
        n_last_frames = T % self.num_frames
        if n_last_frames > 0:
            zeros = torch.zeros((3, self.num_frames - n_last_frames, W, H), dtype=torch.float32).to(frames.device)
            frames = torch.cat([frames, zeros], 1)
            N_chunks += 1

        # extract features
        chunk_features = []
        for i in range(0, N_chunks):
            chunk_frames = frames[:, i * self.num_frames : (i + 1) * self.num_frames, ...][None, ...]
            # chunk_feat = self.net(chunk_frames.cuda())['video_embedding']
            chunk_feat = self.net(chunk_frames)['video_embedding']
            chunk_features.append(chunk_feat)

        chunk_features = torch.cat(chunk_features, 0)
        return chunk_features

    @torch.no_grad()
    def embed_full_subs(self, subs):
        clipped_subs = [' '.join(s.split(' ')[:30]) for s in subs]
        sub_features = self.net.text_module(clipped_subs)['text_embedding']
        return sub_features


class NonlinBlock(nn.Module):
    def __init__(self, d_in, d_out, batchnorm):
        super(NonlinBlock, self).__init__()
        self.fc = nn.Linear(d_in, d_out)
        self.relu = nn.ReLU()
        self.do_batchnorm = batchnorm
        if batchnorm:
            self.norm_fn = nn.BatchNorm1d(d_out)
        # self.layer_norm = nn.LayerNorm(d_out)

    def forward(self, x):
        x = self.fc(x)
        if self.do_batchnorm:
            x = self.norm_fn(x)
        x = self.relu(x)
        return x


class NonlinMapping(nn.Module):
    def __init__(self, d, layers=2, normalization_params=None, batchnorm=False):
        super(NonlinMapping, self).__init__()
        self.nonlin_mapping = nn.Sequential(*[NonlinBlock(d, d, batchnorm) for i in range(layers - 1)])
        if layers > 0:
            self.lin_mapping = nn.Linear(d, d)
        else:
            self.lin_mapping = lambda x: torch.zeros_like(x)

        self.register_buffer('norm_mean', torch.zeros(d))
        self.register_buffer('norm_sigma', torch.ones(d))

    def initialize_normalization(self, normalization_params):
        if normalization_params is not None:
            if len(normalization_params) > 0:
                self.norm_mean.data.copy_(normalization_params[0])
            if len(normalization_params) > 1:
                self.norm_sigma.data.copy_(normalization_params[1])

    def forward(self, x):
        x = (x - self.norm_mean) / self.norm_sigma
        res = self.nonlin_mapping(x)
        res = self.lin_mapping(res)
        return x + res 


class EmbeddingsMapping(nn.Module):
    def __init__(self, d, video_layers=2, text_layers=2, normalization_dataset=None, batchnorm=False):
        super(EmbeddingsMapping, self).__init__()
        self.video_mapping = NonlinMapping(d, video_layers, batchnorm=batchnorm)
        self.text_mapping = NonlinMapping(d, text_layers, batchnorm=batchnorm)
        if normalization_dataset is not None:
            norm_params = compute_normalizetion_parameters(normalization_dataset)
            self.video_mapping.initialize_normalization(norm_params[:2])
            self.text_mapping.initialize_normalization(norm_params[2:])

    def map_video(self, x):
        return self.video_mapping(x)

    def map_text(self, z):
        return self.text_mapping(z)


def compute_normalizetion_parameters(dataset):
    mean_x, mean_z = torch.zeros(512), torch.zeros(512)
    mean_x2, mean_z2 = torch.zeros(512), torch.zeros(512)
    x_count, z_count = 0, 0
    for s in dataset:
        mean_x += s['frame_features'].sum(0)
        mean_x2 += (s['frame_features'] ** 2).sum(0)
        x_count += s['frame_features'].shape[0]
                
        mean_z += s['step_features'].sum(0)
        mean_z2 += (s['step_features'] ** 2).sum(0)
        z_count += s['step_features'].shape[0]
    mean_x = mean_x / x_count
    mean_z = mean_z / z_count
    sigma_x = (mean_x2 / x_count - mean_x ** 2).sqrt()
    sigma_z = (mean_z2 / z_count - mean_z ** 2).sqrt()
    return mean_x, sigma_x, mean_z, sigma_z


class ConvEmbedder(nn.Module):
    def __init__(self):
        super(ConvEmbedder, self).__init__()
        self.conv1 = nn.Sequential(OrderedDict([
            ('conv_1', nn.Conv3d(1024, 512, 3, padding=1)),
            ('bn_1', nn.BatchNorm3d(512)),
            ('relu_1', nn.ReLU(inplace=True))]
            ))
        self.conv2 = nn.Sequential(OrderedDict([
            ('conv_2', nn.Conv3d(512, 512, 3, padding=1)),
            ('bn_2', nn.BatchNorm3d(512)),
            ('relu_2', nn.ReLU(inplace=True))
            ]
            ))
        self.fc1 = nn.Sequential(OrderedDict([
            ('drop1', nn.Dropout(p=0.1)),
            ('fc1', nn.Linear(512, 512)),
            ('relu_11', nn.ReLU(inplace=True))
            ]))
        self.fc2 = nn.Sequential(OrderedDict([
            ('drop2', nn.Dropout(p=0.1)),
            ('fc2', nn.Linear(512, 512)),
            ('relu_22', nn.ReLU(inplace=True))
            ]))
        self.embed = nn.Linear(512,128)
    def forward(self, x, num_frames):
        # reshape input to 3d
        batch_size, total_num_steps, c, h, w = x.shape
        num_context = total_num_steps // num_frames
        x = torch.reshape(x, (batch_size * num_frames, c, num_context, h, w))
        # go through conv layers
        x = self.conv1(x)
        x = self.conv2(x)
        # do global pooling
        for _ in range(3):
            x, _ = torch.max(x, -1)
        # go through fully connected layers
        x = self.fc1(x)
        x = self.fc2(x)
        # go through embedding layer
        x = self.embed(x)
        # do L2 norm
        x = F.normalize(x, p=2, dim=1)
        # reshape
        embs = torch.reshape(x, (-1, num_frames, 128))
        return embs
    
class ResNetEmbedder(nn.Module):
    def __init__(self):
        super(ResNetEmbedder, self).__init__()
        # call resnet50
        resnet50 = models.resnet50(pretrained=True, progress=False)
        self.resnet50_layers = torch.nn.Sequential(*list(resnet50.children())[:-3])
        self.conv_embedder = ConvEmbedder()
    def forward(self, x, num_frames):
        # reshape input to use with 2d model
        batch_size, d, c, h, w = x.shape
        x = torch.reshape(x, (batch_size*d, c, h,w))
        # go throgh resnet layers
        x = self.resnet50_layers(x)
        # fix shape back
        _, c, h, w = x.shape
        x = torch.reshape(x, (batch_size, -1, c, h, w))
        # go through embedder layers
        emb = self.conv_embedder(x, num_frames)
        return x, emb
