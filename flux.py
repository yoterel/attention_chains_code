import os
import argparse
from pathlib import Path
import numpy as np
import shutil
import torch
from pipeline_flux import (
    FluxPipeline, 
    CustomFluxAttnProcessor2_0
)
from helpers import (
    power_method,
    batch_power_method,
    prep_for_power_method
)
import yaml
import matplotlib.pyplot as plt
from PIL import Image

class Struct:
    def __init__(self, **entries):
        self.__dict__.update(entries)


def main():
    # parse config
    parser = argparse.ArgumentParser()
    parser.add_argument('config')
    pargs = parser.parse_args()
    config = yaml.safe_load(open(pargs.config))
    args = Struct(**config)
    job_index = args.default_job_index
    if "SLURM_ARRAY_TASK_ID" in os.environ:
        print("slurm job index={}".format(os.environ["SLURM_ARRAY_TASK_ID"]))
        job_index = os.environ["SLURM_ARRAY_TASK_ID"]
    job_index = int(job_index)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    config_dst = Path(output_dir, "config.yml")
    shutil.copy(pargs.config, config_dst)
    if args.mixed_precision:
        weight_dtype = torch.float16
    else:
        weight_dtype = torch.float32
    # load pipeline
    pipe = FluxPipeline.from_pretrained(args.pretrained_model_name_or_path,
                                        torch_dtype=weight_dtype,
                                        cache_dir=args.cache_dir)
    pipe.enable_model_cpu_offload()
    # replace with custom attention processor to save attention maps
    if "dual" in args.attention_channel:
        for i, tb in enumerate(pipe.transformer.transformer_blocks):
            if i in args.transformer_blocks_to_save:
                store_processor = CustomFluxAttnProcessor2_0(args)
                tb.attn.processor = store_processor
    if "single" in args.attention_channel:
        for i, stb in enumerate(pipe.transformer.single_transformer_blocks):
            if i in args.single_transformer_blocks_to_save:
                store_processor = CustomFluxAttnProcessor2_0(args)
                stb.attn.processor = store_processor
    torch.manual_seed(args.seed + job_index)
    generator = torch.Generator(device=args.device).manual_seed(args.seed + job_index)
    # generate image per seed & save visualization of attention maps
    for i in range(args.num_seeds):
        if i//args.max_seeds_per_job != job_index:
            print("skipping, not my job")
            continue
        images = pipe(
            prompt=args.prompt,
            guidance_scale=0.,
            height=512,  # 768
            width=512,  # 1360
            num_inference_steps=args.num_inference_steps,
            generator=generator,
            max_sequence_length=args.max_seq_len,
        ).images
        images[0].save(Path(output_dir, "{:05}.png".format(i)))
        tb, tb_raw, stb, stb_raw = get_attentions(pipe, args)  # (blocks, steps, heads, seq, seq)
        for word in args.visualize_words:
            print(word)
            chunks = word.split("_")
            cur_word = chunks[0]
            index = 0
            if len(chunks) == 2:
                index = int(chunks[1])
            token_index = pipe.find_token_indices(args.prompt, cur_word)[index]
            for method in args.methods:
                print(method)
                if "dual" in args.attention_channel:
                    plt_attention(word, token_index, method, tb, tb_raw, None, args, output_dir, "single_stream")
                if "single" in args.attention_channel:
                    plt_attention(word, token_index, method, stb, stb_raw, None, args, output_dir, "dual_stream")


def plt_attention(word, token_index, method, attention, attention_raw, layer_index, args, output_dir, att_type_name):
    """
    visualizes attention matrices
    :param word: the string represented by the token of interest
    :param token_index: the index of the token of interest
    :param method: what type of visualization
        tokenrank_src: visualizes token rank only (outgoing attention)
        row_select: visualizes the row select op (outgoing attention)
        bounce_src: visualizes all bounces, note first bounce is equivalent to row_select (outgoing attention)
        tokenrank_sink: visualizes token rank only (incoming attention)
        tokenrank_sink_pretty: visualizes token rank only, normalizes columns for better visualization (incoming attention)
        column_select: visualizes the column select op (incoming attention)
        bounce_sink: visualizes all bounces, note first bounce is equivalent to column_select (incoming attention)
        bounce_sink_pretty: visualizes all bounces, note first bounce is equivalent to column_select, normalizes columns for better visualization (incoming attention)
    :param attention: (blocks, steps, heads, seq, seq) matrix of softmaxed att scores
    :param attention_raw: (blocks, steps, heads, seq, seq) matrix of raw att scores
    :param layer_index: visualize only for a specific layer
    :param args: command line arguments
    :param output_dir: directory to output visualization
    :param att_type_name: helps distinguish dual or single stream
    Note: for FLUX dev, we average over head dimension prior to this function due to memory constraints.
    """
    attention_raw = attention_raw.detach().clone().to(torch.float32)
    attention_norm = torch.softmax(attention_raw, dim=-1)
    if layer_index is not None:
        attention_norm = attention_norm.mean(dim=(1,2))[layer_index:layer_index+1, ...]  # (1, seq, seq)
    else:
        if not args.pretrained_model_name_or_path == "black-forest-labs/FLUX.1-dev":
            attention_norm = attention_norm.mean(dim=(0,1,2))[None, ...]  # (1, seq, seq)
            attention_norm = attention_norm.to(args.device)
    if method == "tokenrank_sink":
        if args.pretrained_model_name_or_path == "black-forest-labs/FLUX.1-dev":
            attention_norm = attention_norm.mean(dim=0)  # (timesteps, seq, seq)
            attention_norm = attention_norm.to(args.device)
            T1 = prep_for_power_method(attention_norm, args.tokenrank_alpha)
            init_state = torch.ones(T1.shape[-1], dtype=T1.dtype).to(T1.device) / T1.shape[-1]
            stst = batch_power_method(T1, init_state)  # (timesteps, seq)
            signal = stst[:, args.max_seq_len:]
            signal = signal.reshape(10, -1, signal.shape[-1]).mean(dim=1)
        else:
            T1 = prep_for_power_method(attention_norm[0], args.tokenrank_alpha)
            init_state = torch.ones(T1.shape[-1], dtype=T1.dtype).to(T1.device) / T1.shape[-1]
            stst = power_method(T1, init_state)  # , return_intermed=True
            signal = stst[args.max_seq_len:]
    elif method == "tokenrank_sink_pretty":
        my_att = torch.softmax(attention_raw, dim=-2)
        if args.pretrained_model_name_or_path == "black-forest-labs/FLUX.1-dev":
            my_att = my_att.mean(dim=0)  # (timesteps, seq, seq)
            my_att = my_att.to(args.device)
            row_sums = my_att.sum(dim=-1, keepdims=True)
            right_stochastic = my_att / row_sums
            T1 = prep_for_power_method(right_stochastic, args.tokenrank_alpha)
            init_state = torch.ones(T1.shape[-1], dtype=T1.dtype).to(T1.device) / T1.shape[-1]
            stst = batch_power_method(T1, init_state)
            signal = stst[:, args.max_seq_len:]
            signal = signal.reshape(10, -1, signal.shape[-1]).mean(dim=1)
        else:
            my_att = my_att.mean(dim=(0,1,2))[None, ...]  # (1, seq, seq)
            row_sums = my_att.sum(dim=-1, keepdims=True)
            right_stochastic = my_att / row_sums
            T1 = prep_for_power_method(right_stochastic[0], args.tokenrank_alpha)
            init_state = torch.ones(T1.shape[-1], dtype=T1.dtype).to(T1.device) / T1.shape[-1]
            stst = power_method(T1, init_state)  # , return_intermed=True
            signal = stst[args.max_seq_len:]
    elif method == "tokenrank_src":
        T = prep_for_power_method(attention_norm[0], args.tokenrank_alpha)
        col_sums = T.sum(dim=0, keepdims=True)
        left_stochastic = T / col_sums
        init_state = torch.ones(left_stochastic.shape[-1], dtype=left_stochastic.dtype).to(args.device) / left_stochastic.shape[-1]
        stst = power_method(left_stochastic.transpose(1, 0), init_state)  # return_intermed=True
        signal = stst[args.max_seq_len:]
    elif method == "column_select":
        signal = attention_norm[:, args.max_seq_len:, token_index]  # (1, img_seq)
    elif method == "row_select":
        signal = attention_norm[:, token_index, args.max_seq_len:]  # (1, img_seq)
    elif method == "bounce_src":
        T = prep_for_power_method(attention_norm[0])
        col_sums = T.sum(dim=0, keepdims=True)
        left_stochastic = T / col_sums
        image_sequence_size = left_stochastic.shape[-1]
        init_state = torch.zeros(left_stochastic.shape[-1], dtype=left_stochastic.dtype).to(args.device)
        init_state[token_index] = 1.0
        bounces = torch.zeros(args.n_bounces, image_sequence_size, dtype=left_stochastic.dtype).to(args.device)
        bounce = init_state
        for i in range(args.n_bounces):
            bounce = bounce / bounce.sum()
            bounce = left_stochastic @ bounce
            bounces[i] = bounce   
        signal = bounces[:, args.max_seq_len:]
    elif method == "bounce_sink":
        T = prep_for_power_method(attention_norm[0])
        image_sequence_size = T.shape[-1]
        init_state = torch.zeros(T.shape[-1], dtype=T.dtype).to(args.device)
        init_state[token_index] = 1.0
        bounces = torch.zeros(args.n_bounces, image_sequence_size, dtype=T.dtype).to(args.device)
        bounce = init_state
        for i in range(args.n_bounces):
            bounce = bounce / bounce.sum()
            bounce = T.T @ bounce
            bounces[i] = bounce   
        signal = bounces[:, args.max_seq_len:]
    elif method == "bounce_sink_pretty":
        my_att = torch.softmax(attention_raw, dim=-2)
        if args.pretrained_model_name_or_path == "black-forest-labs/FLUX.1-dev":
            my_att = my_att.mean(dim=(0,1))[None, ...]  # (1, seq, seq)
        else:
            my_att = my_att.mean(dim=(0,1,2))[None, ...]  # (1, seq, seq)
        my_att = my_att.to(args.device)
        row_sums = my_att.sum(dim=-1, keepdims=True)
        right_stochastic = my_att / row_sums
        T = prep_for_power_method(right_stochastic[0])
        image_sequence_size = T.shape[-1]
        init_state = torch.zeros(T.shape[-1], dtype=T.dtype).to(args.device)
        init_state[token_index] = 1.0
        bounces = torch.zeros(args.n_bounces, image_sequence_size, dtype=T.dtype).to(args.device)
        bounce = init_state
        for i in range(args.n_bounces):
            bounce = bounce / bounce.sum()
            bounce = T.T @ bounce
            bounces[i] = bounce   
        # signal = bounce[args.max_seq_len:]
        signal = bounces[:, args.max_seq_len:]
    else:
        raise NotImplementedError
    signal = signal.to("cpu")
    signal = signal.reshape(-1, 32, 32)
    signal = torch.nn.functional.interpolate(signal.unsqueeze(0), scale_factor=32, mode="nearest")[0].numpy()
    nom = signal - signal.min(axis=(1,2), keepdims=True)
    denom = signal.max(axis=(1,2), keepdims=True) - signal.min(axis=(1,2), keepdims=True)
    signal = nom / (denom + 1e-6)
    signal = (255*np.clip(signal, 0, 1)).round().astype(np.uint8)
    save_path = Path(output_dir, method, att_type_name)
    save_path.mkdir(exist_ok=True, parents=True)
    for i in range(len(signal)):
        plt.imsave(fname=Path(save_path, "{}_{:02d}.png".format(word, i)), arr=signal[i, :, :], format='png')


def get_attentions(pipe, args):
    att_probs_tb = []
    att_raw_tb = []
    att_probs_stb = []
    att_raw_stb = []
    if "dual" in args.attention_channel:
        for i, tb in enumerate(pipe.transformer.transformer_blocks):
            if i in args.transformer_blocks_to_save:
                att_probs = tb.attn.processor.attention_probs  # list of (1, head, seq, seq)
                att_raw = tb.attn.processor.attention_raw  # list of (1, head, seq, seq)
                att_probs_tb.append(torch.cat([x for x in att_probs]))  # append((timesteps, head, seq, seq)
                att_raw_tb.append(torch.cat([x for x in att_raw]))  # append((timesteps, head, seq, seq)
        if args.pretrained_model_name_or_path == "black-forest-labs/FLUX.1-dev":
            # get rid of head dimension, too heavy otherwise
            att_raw_tb = [x.mean(dim=1) for x in att_raw_tb]
            att_probs_tb = [x.mean(dim=1) for x in att_probs_tb]
        att_probs_tb = torch.cat(att_probs_tb).reshape(len(args.transformer_blocks_to_save), 
                                                        args.num_inference_steps, 
                                                        *att_probs_tb[0].shape[1:])
        att_raw_tb = torch.cat(att_raw_tb).reshape(len(args.transformer_blocks_to_save),
                                                        args.num_inference_steps,
                                                        *att_raw_tb[0].shape[1:])
    if "single" in args.attention_channel:
        for i, stb in enumerate(pipe.transformer.single_transformer_blocks):
            if i in args.single_transformer_blocks_to_save:
                att_probs = stb.attn.processor.attention_probs
                att_raw = stb.attn.processor.attention_raw  # list of (1, head, seq, seq)
                att_probs_stb.append(torch.cat([x for x in att_probs]))
                att_raw_stb.append(torch.cat([x for x in att_raw]))
        att_probs_stb = torch.cat(att_probs_stb).reshape(len(args.single_transformer_blocks_to_save),
                                                        args.num_inference_steps,
                                                        *att_probs_stb[0].shape[1:])
        att_raw_stb = torch.cat(att_raw_stb).reshape(len(args.single_transformer_blocks_to_save),
                                                        args.num_inference_steps,
                                                        *att_raw_stb[0].shape[1:])
    return att_probs_tb, att_raw_tb, att_probs_stb, att_raw_stb

if __name__=="__main__":
    main()