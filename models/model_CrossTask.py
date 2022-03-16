from TransFormer import *

class ProcedureFormer(nn.Module):
    def __init__(
        self,
        input_dim,
        d_model,
        cls_size,
        f_model=128,
        device="cpu",
        pred_horz=3,
        nhead=2,
        nlayer=2,
        batch=32,
        noise_dim=64,
    ):
        super(ProcedureFormer, self).__init__()
        self.pos_encoder = PositionalAgentEncoding(
            d_model=d_model - noise_dim,
            dropout=0.1, # first dropout that I have in my network.
            concat=True,
            max_t_len=pred_horz + 1,
        )

        decoder_layers = TransFormerDecoderLayer(
            cfg={},
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=f_model,
            dropout=0.3, # 2nd dropout that I have in my network.
        )
        self.tf_decoder = TransFormerDecoder(decoder_layers, num_layers=nlayer)

        self.cls_decoder = MLP(d_model, cls_size, [256])
        self.state_encoder = MLP(input_dim, d_model - noise_dim, [256])
        self.state_decoder = MLP(d_model, d_model - noise_dim, [256])
        self.lang_encoder = MLP(512, d_model - noise_dim, [256])

        """Learnable query for producing the results"""
        self.query_embed = torch.nn.Embedding(pred_horz - 1, d_model)
        self.keyvalue_embed = torch.nn.Embedding(106, d_model)

        self.batch_size = batch
        self.noise_dim = noise_dim

        """Very simple discriminator """
        self.discriminator_pred_cls_enc = torch.nn.Embedding(cls_size, d_model)
        self.discriminator1 = MLP(d_model* 2 - noise_dim, 32, [64, 32])
        self.discriminator2 = MLP(32, 1)

    def discriminator_forward(self, x):
        x = self.discriminator1(x)
        x = x.mean(1)
        return self.discriminator2(x)

    def forward(self, x, pred_horz=3):

        # How to incorporate batch processing?
        x = self.state_encoder(x)

        vis_emb_seq = torch.mean(x, dim=2)
        _, _, _ = vis_emb_seq.shape

        vis_emb_seq = vis_emb_seq.permute(1, 0, 2)
        start_vis = vis_emb_seq[0:1]
        goal_vis = vis_emb_seq[-1:]

        start_pos = self.pos_encoder(start_vis, num_a=1, t_offset=0)
        goal_pos = self.pos_encoder(goal_vis, num_a=1, t_offset=pred_horz)

        placeholder_emb = []
        for i in range(pred_horz - 1):
            placeholder_emb.append(
                self.pos_encoder(
                    # self.query_embed.weight[i : i + 1].unsqueeze(0),
                    torch.zeros_like(
                        start_pos
                    ).cuda(),  # this is non-learnable variables
                    num_a=1,
                    t_offset=i + 1,
                )
            )
        placeholder_emb.insert(0, start_pos)
        placeholder_emb.insert(-1, goal_pos)

        memory = self.keyvalue_embed.weight.unsqueeze(1)
        memory = memory.repeat(1, x.shape[0], 1)

        input_query = torch.cat(placeholder_emb, 0)

        ###########################
        # Initialize random noise #
        ###########################
        "noise_z shape: [pred_horz, batch, dimension]"
        noise_z = torch.randn(
            1, input_query.shape[1], self.noise_dim
        ).cuda()

        "Repeat the same noise for every horizon step:"
        noise_z = noise_z.repeat(input_query.shape[0], 1, 1)

        input_query = torch.cat((input_query, noise_z), -1)

        seq_out, _ = self.tf_decoder(input_query, memory, memory)
        return self.cls_decoder(seq_out[:-1]), self.state_decoder(seq_out[:-1]), noise_z

    def inference(self, x, z, pred_horz=3, NDR=False):
        ##############################
        # Take random noise as input #
        ##############################
        
        x = self.state_encoder(x)

        vis_emb_seq = torch.mean(x, dim=2)
        _, _, _ = vis_emb_seq.shape

        vis_emb_seq = vis_emb_seq.permute(1, 0, 2)
        start_vis = vis_emb_seq[0:1]
        goal_vis = vis_emb_seq[-1:]

        start_pos = self.pos_encoder(start_vis, num_a=1, t_offset=0)
        goal_pos = self.pos_encoder(goal_vis, num_a=1, t_offset=pred_horz)

        placeholder_emb = []
        for i in range(pred_horz - 1):
            placeholder_emb.append(
                self.pos_encoder(
                    torch.zeros_like(
                        start_pos
                    ).cuda(),  # this is non-learnable variables
                    num_a=1,
                    t_offset=i + 1,
                )
            )
        placeholder_emb.insert(0, start_pos)
        placeholder_emb.insert(-1, goal_pos)
        
        # Repeat in BATCH dimension for batch-operation on memories
        memory = self.keyvalue_embed.weight.unsqueeze(1)
        memory = memory.repeat(1, x.shape[0], 1)

        # Initialize random noise
        input_query = torch.cat(placeholder_emb, 0)
        input_query = torch.cat((input_query, z), -1)

        seq_out, _ = self.tf_decoder(input_query, memory, memory)

        # especial output portal for NDR
        if NDR:
            return self.cls_decoder(seq_out[:-1]), seq_out[:-1]
        else:
            return self.cls_decoder(seq_out[:-1]), self.state_decoder(seq_out[:-1])


if __name__ == "__main__":
    model = ProcedureFormer(
        input_dim=512, d_model=128, cls_size=106, noise_dim=32
    ).cuda()
    test_inp = torch.rand(1, 4, 3, 512).cuda()
    out, state, noise_z = model(test_inp)

    print(out.shape)
    print(state.shape)
    print(noise_z.shape)

    test_inp = torch.rand(16, 4, 128 - 32).cuda()
    out = model.discriminator_forward(test_inp)
    print(out.shape)