import math
import torch
import torch.nn as nn
from models.encdec import Encoder, Decoder
from models.quantize_cnn import QuantizeEMAReset, Quantizer, QuantizeEMA, QuantizeReset


class VQVAE_251(nn.Module):
    def __init__(self,
                 args,
                 nb_code=1024,
                 code_dim=512,
                 output_emb_width=512,
                 down_t=3,
                 stride_t=2,
                 width=512,
                 depth=3,
                 dilation_growth_rate=3,
                 activation='relu',
                 norm=None,
                 index_groups=None):
        
        super().__init__()
        self.quant = args.quantizer
        input_dim = 263
        if args.dataname.split('_')[0] == "face":
            input_dim = 56
        elif args.dataname.split('_')[0] == "pats":
            input_dim = 129
        elif args.dataname.split('_')[0] == "audio":
            input_dim = 128
        elif args.dataname == "kit":
            input_dim = 251
        if index_groups is None:
            index_groups = [list(range(input_dim))]
        if not isinstance(nb_code, list):
            nb_code = [nb_code for _ in index_groups]
        self.num_code = math.prod(nb_code)
        self.code_dim = code_dim*len(index_groups)
        self.index_groups = index_groups
        encoders = [Encoder(len(group), output_emb_width, down_t, stride_t, width, depth, dilation_growth_rate, activation=activation, norm=norm) for group in index_groups]
        self.encoder = nn.ModuleList(encoders)
        decoders = [Decoder(len(group), output_emb_width, down_t, stride_t, width, depth, dilation_growth_rate, activation=activation, norm=norm) for group in index_groups]
        self.decoder = nn.ModuleList(decoders)
        # self.encoder = Encoder(input_dim, output_emb_width, down_t, stride_t, width, depth, dilation_growth_rate, activation=activation, norm=norm)
        # self.decoder = Decoder(input_dim, output_emb_width, down_t, stride_t, width, depth, dilation_growth_rate, activation=activation, norm=norm)
        if args.quantizer == "ema_reset":
            self.quantizer = nn.ModuleList([QuantizeEMAReset(nb_c, code_dim, args) for nb_c in nb_code])
        elif args.quantizer == "orig":
            self.quantizer = nn.ModuleList([Quantizer(nb_c, code_dim, 1.0) for nb_c in nb_code])
        elif args.quantizer == "ema":
            self.quantizer = nn.ModuleList([QuantizeEMA(nb_c, code_dim, args) for nb_c in nb_code])
        elif args.quantizer == "reset":
            self.quantizer = nn.ModuleList([QuantizeReset(nb_c, code_dim, args) for nb_c in nb_code])


    def preprocess(self, x):
        # (bs, T, Jx3) -> (bs, Jx3, T)
        x = x.permute(0,2,1).float()
        return x


    def postprocess(self, x):
        # (bs, Jx3, T) ->  (bs, T, Jx3)
        x = x.permute(0,2,1)
        return x


    def encode(self, x):
        code_idx_per_group = []
        for group_index, group in enumerate(self.index_groups):
            N, T, _ = x[...,group].shape
            x_in = self.preprocess(x[...,group])
            x_encoder = self.encoder[group_index](x_in)
            x_encoder = self.postprocess(x_encoder)
            x_encoder = x_encoder.contiguous().view(-1, x_encoder.shape[-1])  # (NT, C)
            code_idx = self.quantizer[group_index].quantize(x_encoder)
            # print("code", code_idx.shape, "x", x.shape)
            # assert False
            code_idx = code_idx.view(N, -1)
            code_idx_per_group.append(code_idx)
        if len(code_idx_per_group) == 1:
            return code_idx_per_group[0]
        return tuple(code_idx_per_group)


    def forward(self, x):
        total_loss = 0
        total_perplexity = 0
        output = torch.zeros_like(x).float()
        # print('x', x.shape)
        for group_index, group in enumerate(self.index_groups):
            # print('group', group)
            x_in = self.preprocess(x[...,group])
            # print('x_in', x_in.shape)
            # Encode
            x_encoder = self.encoder[group_index](x_in)
            
            ## quantization
            x_quantized, loss, perplexity  = self.quantizer[group_index](x_encoder)

            total_loss += loss
            total_perplexity += perplexity

            ## decoder
            x_decoder = self.decoder[group_index](x_quantized)
            x_out = self.postprocess(x_decoder)
            output[...,group] = x_out
        return output, total_loss, total_perplexity


    def forward_decoder(self, x):
        output = []
        for group_index, group in enumerate(self.index_groups):
            x_d = self.quantizer[group_index].dequantize(x)
            x_d = x_d.view(1, -1, self.code_dim).permute(0, 2, 1).contiguous()
            
            # decoder
            x_decoder = self.decoder[group_index](x_d)
            x_out = self.postprocess(x_decoder)
            output.append((group, x_out))
        x_out = torch.zeros((*output[0][1].shape[:-1], sum([len(out[0]) for out in output]))).to(x.device).to(output[0][1].dtype)
        for (group, out) in output:
            x_out[...,group] = out
        return x_out



class HumanVQVAE(nn.Module):
    def __init__(self,
                 args,
                 nb_code=512,
                 code_dim=512,
                 output_emb_width=512,
                 down_t=3,
                 stride_t=2,
                 width=512,
                 depth=3,
                 dilation_growth_rate=3,
                 activation='relu',
                 norm=None,
                 aux_labels=0,
                 index_groups=None):
        
        super().__init__()
        
        self.nb_joints = 21 if args.dataname == 'kit' else 22
        self.vqvae = VQVAE_251(args, nb_code, code_dim, output_emb_width, down_t, stride_t, width, depth, dilation_growth_rate, activation=activation, norm=norm, index_groups=index_groups)
        if aux_labels > 0:
            self.aux_classifier = nn.Linear(output_emb_width, aux_labels)

    def encode(self, x):
        b, t, c = x.size()
        quants = self.vqvae.encode(x) # (N, T)
        return quants

    def forward(self, x):

        x_out, loss, perplexity = self.vqvae(x)
        
        return x_out, loss, perplexity

    def forward_decoder(self, x):
        x_out = self.vqvae.forward_decoder(x)
        return x_out
        
    def forward_aux(self, x):
        encoding = self.vqvae.encoder(self.vqvae.preprocess(x))
        return self.aux_classifier(encoding)
