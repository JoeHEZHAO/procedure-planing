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
    ):
        super(ProcedureFormer, self).__init__()
        self.pos_encoder = PositionalAgentEncoding(
            d_model=d_model,
            dropout=0.1,
            concat=True,
            max_t_len=pred_horz + 1,
        )

        decoder_layers = TransFormerDecoderLayer(
            cfg={}, d_model=d_model, nhead=nhead, dim_feedforward=f_model, dropout=0.1
        )
        self.tf_decoder = TransFormerDecoder(decoder_layers, num_layers=nlayer)

        self.cls_decoder = MLP(d_model, cls_size, [256])
        self.state_encoder = MLP(input_dim, d_model, [256])
        self.lang_encoder = MLP(input_dim, d_model, [256])

        """Learnable query for producing the results"""
        self.query_embed = torch.nn.Embedding(pred_horz - 1, d_model)
        self.keyvalue_embed = torch.nn.Embedding(106, d_model)

        self.batch_size = batch

    def forward(self, x, pred_horz=3):
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
                        self.query_embed.weight[i: i + 1].unsqueeze(0)
                    ).cuda(),  # this is non-learnable variables
                    num_a=1,
                    t_offset=i + 1,
                )
            )
        placeholder_emb.insert(0, start_pos)
        placeholder_emb.insert(-1, goal_pos)

        memory = self.keyvalue_embed.weight.unsqueeze(1)

        seq_out, _ = self.tf_decoder(
            torch.cat(placeholder_emb, 0), memory, memory)
        return self.cls_decoder(seq_out[:-1]), seq_out[:-1]


if __name__ == "__main__":
    model = ProcedureFormer(input_dim=512, d_model=32, cls_size=106).cuda()
    test_inp = torch.rand(1, 4, 3, 512).cuda()
    out = model(test_inp)
    print(out.shape)
