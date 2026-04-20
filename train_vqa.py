import argparse
import datetime
import json
import os
import sys
import time
from pathlib import Path

import ruamel_yaml as yaml
import torch
import torch.distributed as dist

import utils
from dataset import create_dataset, create_loader, create_sampler, vqa_collate_fn
from models.model_vqa import MUMC_VQA, normalize_model_variant
from models.tokenization_bert import BertTokenizer
from models.vision.vit import interpolate_pos_embed
from utils import cosine_lr_schedule
from vqaEvaluate import compute_vqa_acc


def str2bool(value):
    if isinstance(value, bool):
        return value
    value = str(value).lower()
    if value in ['true', '1', 'yes', 'y']:
        return True
    if value in ['false', '0', 'no', 'n']:
        return False
    raise ValueError('Boolean value expected.')


def apply_runtime_overrides(config, args):
    config['model_variant'] = normalize_model_variant(args.model_variant or config.get('model_variant', 'mumc'))
    config['use_diffrep'] = config['model_variant'] in ['mumc_diffrep', 'mumc_diffrepalign']
    config['use_diffalign'] = config['model_variant'] in ['mumc_diffalign', 'mumc_diffrepalign']
    config['train_stage'] = args.train_stage or config.get('train_stage', 'baseline')
    config['use_global_contrast'] = args.use_global_contrast if args.use_global_contrast is not None else config.get(
        'use_global_contrast', config['use_diffrep'] or config['use_diffalign'])
    config['use_answer_contrast'] = args.use_answer_contrast if args.use_answer_contrast is not None else config.get(
        'use_answer_contrast', False)
    config['use_decorrelation'] = args.use_decorrelation if args.use_decorrelation is not None else config.get(
        'use_decorrelation', config['use_diffrep'] or config['use_diffalign'])
    config['use_gate_entropy'] = args.use_gate_entropy if args.use_gate_entropy is not None else config.get(
        'use_gate_entropy', False)

    diffusion_cfg = config.setdefault('diffusion', {})
    diffusion_cfg['enabled'] = config['use_diffrep'] or config['use_diffalign'] or diffusion_cfg.get('enabled', False)
    if args.diffusion_model_id:
        diffusion_cfg['model_id'] = args.diffusion_model_id
    if args.diffusion_local_files_only is not None:
        diffusion_cfg['local_files_only'] = args.diffusion_local_files_only
    if args.diffusion_frozen is not None:
        diffusion_cfg['frozen'] = args.diffusion_frozen

    optimizer_cfg = config.setdefault('optimizer', {})
    if args.new_lr is not None:
        optimizer_cfg['new_lr'] = args.new_lr
    if args.backbone_lr is not None:
        optimizer_cfg['backbone_lr'] = args.backbone_lr
    if args.decoder_lr is not None:
        optimizer_cfg['decoder_lr'] = args.decoder_lr
    if args.diffusion_backbone_lr is not None:
        optimizer_cfg['diffusion_backbone_lr'] = args.diffusion_backbone_lr
    if args.optimizer_weight_decay is not None:
        optimizer_cfg['weight_decay'] = args.optimizer_weight_decay

    loss_cfg = config.setdefault('loss', {})
    if args.lambda_global_contrast is not None:
        loss_cfg['lambda_global_contrast'] = args.lambda_global_contrast
    if args.lambda_answer_contrast is not None:
        loss_cfg['lambda_answer_contrast'] = args.lambda_answer_contrast
    if args.lambda_decorrelation is not None:
        loss_cfg['lambda_decorrelation'] = args.lambda_decorrelation
    if args.lambda_gate_entropy is not None:
        loss_cfg['lambda_gate_entropy'] = args.lambda_gate_entropy
    if args.unfreeze_visual_last_n_blocks is not None:
        config['unfreeze_visual_last_n_blocks'] = args.unfreeze_visual_last_n_blocks
    if args.unfreeze_text_last_n_layers is not None:
        config['unfreeze_text_last_n_layers'] = args.unfreeze_text_last_n_layers

    if config['model_variant'] == 'mumc':
        diffusion_cfg['enabled'] = False
        config['use_global_contrast'] = False if args.use_global_contrast is None else config['use_global_contrast']
        config['use_answer_contrast'] = False if args.use_answer_contrast is None else config['use_answer_contrast']
        config['use_decorrelation'] = False if args.use_decorrelation is None else config['use_decorrelation']
        config['use_gate_entropy'] = False if args.use_gate_entropy is None else config['use_gate_entropy']

    return config


def build_optimizer(model, config):
    parameter_groups = model.get_optimizer_groups(config)
    return torch.optim.AdamW(parameter_groups, lr=config['init_lr'], weight_decay=config['weight_decay'])


def train(model, data_loader, optimizer, epoch, device, config):
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('loss', utils.SmoothedValue(window_size=1, fmt='{value:.4f}'))
    header = 'Train Epoch: [{}]'.format(epoch)
    print_freq = 50
    for i, (image, question, answer) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):

        image = image.to(device, non_blocking=True)

        if epoch > 0 or not config['warm_up']:
            alpha = config['alpha']
        else:
            alpha = config['alpha'] * min(1, i / len(data_loader))

        loss_output = model(image, question, answer, train=True, alpha=alpha)
        loss = loss_output['loss'] if isinstance(loss_output, dict) else loss_output

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if isinstance(loss_output, dict):
            log_dict = {}
            for key, value in loss_output.items():
                if key == 'loss':
                    continue
                log_dict[key] = value.item() if isinstance(value, torch.Tensor) else value
            metric_logger.update(**log_dict)
        metric_logger.update(loss=loss.item())
        metric_logger.update(lr=max(group['lr'] for group in optimizer.param_groups))

    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger.global_avg())
    return {k: "{:.6f}".format(meter.global_avg) for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluation(model, data_loader, device, config):
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Generate VQA test result:'
    print_freq = 50

    result = []

    answer_list = [answer + config['eos'] for answer in data_loader.dataset.answer_list]

    for _, (image, question, question_id) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        image = image.to(device, non_blocking=True)
        topk_ids, topk_probs = model(image, question, answer_list, train=False, k=config['k_test'])

        for ques_id, topk_id, topk_prob in zip(question_id, topk_ids, topk_probs):
            ques_id = int(ques_id.item())
            _, pred = topk_prob.max(dim=0)
            result.append({"qid": ques_id, "answer": data_loader.dataset.answer_list[topk_id[pred]]})
    return result


def main(args, config):
    if args.distributed:
        utils.init_distributed_mode(args)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    utils.set_seed(args.seed + utils.get_rank())

    print('Creating vqa {} datasets'.format(args.dataset_use))
    datasets = create_dataset(args.dataset_use, config)
    print('train dataset size: ', len(datasets[0]))
    print('test dataset size: ', len(datasets[1]))

    if args.distributed:
        num_tasks = utils.get_world_size()
        global_rank = utils.get_rank()
        samplers = create_sampler(datasets, [True, False], num_tasks, global_rank)
    else:
        samplers = [None, None]

    train_loader, test_loader = create_loader(datasets, samplers,
                                              batch_size=[config['batch_size_train'], config['batch_size_test']],
                                              num_workers=[4, 4], is_trains=[True, False],
                                              collate_fns=[vqa_collate_fn, None])

    tokenizer = BertTokenizer.from_pretrained(args.text_encoder)

    print("Creating model")
    model = MUMC_VQA(config=config, text_encoder=args.text_encoder, text_decoder=args.text_decoder, tokenizer=tokenizer)
    model = model.to(device)
    optimizer = build_optimizer(model, config)

    if args.checkpoint:
        checkpoint = torch.load(args.checkpoint, map_location='cpu')
        state_dict = checkpoint['model']

        pos_embed_reshaped = interpolate_pos_embed(state_dict['visual_encoder.pos_embed'], model.visual_encoder)
        state_dict['visual_encoder.pos_embed'] = pos_embed_reshaped

        if not args.evaluate:
            if config['distill']:
                m_pos_embed_reshaped = interpolate_pos_embed(state_dict['visual_encoder_m.pos_embed'],
                                                             model.visual_encoder_m)
                state_dict['visual_encoder_m.pos_embed'] = m_pos_embed_reshaped

            for key in list(state_dict.keys()):
                if 'bert' in key:
                    encoder_key = key.replace('bert.', '')
                    state_dict[encoder_key] = state_dict[key]
                if 'text_encoder' in key:
                    if 'layer' in key:
                        encoder_keys = key.split('.')
                        layer_num = int(encoder_keys[4])
                        if layer_num < 6:
                            del state_dict[key]
                            continue
                        decoder_layer_num = layer_num - 6
                        encoder_keys[4] = str(decoder_layer_num)
                        encoder_key = '.'.join(encoder_keys)
                    else:
                        encoder_key = key
                    decoder_key = encoder_key.replace('text_encoder', 'text_decoder')
                    state_dict[decoder_key] = state_dict[key]

                    del state_dict[key]

        msg = model.load_state_dict(state_dict, strict=False)
        print('load checkpoint from %s' % args.checkpoint)
        print(msg)

    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module

    start_epoch = 0
    print("\nStart training\n")
    start_time = time.time()

    prefix = args.checkpoint.split('/')[-1].split('.')[0] if args.checkpoint else config['model_variant']
    if args.evaluate:
        vqa_result = evaluation(model, test_loader, device, config)
        if utils.is_main_process():
            json.dump(vqa_result, open(os.path.join(args.result_dir, '%s_%s_eval.json' % (prefix, config['model_variant'])), 'w'))
        return

    for epoch in range(start_epoch, config['max_epoch']):
        if not args.evaluate:
            if args.distributed:
                train_loader.sampler.set_epoch(epoch)

            cosine_lr_schedule(optimizer, epoch, config['max_epoch'], config['init_lr'], config['min_lr'])

            train(model, train_loader, optimizer, epoch, device, config)

        if args.evaluate:
            break

        if utils.is_main_process():

            save_obj = {
                'model': model_without_ddp.state_dict(),
            }
            if args.is_save_path and epoch > 20:
                torch.save(save_obj, os.path.join(args.output_dir, '%s_%s_%02d.pth' % (prefix, config['model_variant'], epoch)))
            vqa_result = evaluation(model, test_loader, device, config)
            json.dump(vqa_result, open(os.path.join(args.result_dir, '%s_%s_vqa_result_%s.json' % (prefix, config['model_variant'], epoch)), 'w'))

        if args.distributed:
            dist.barrier()

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))

    res_file_path = '%s/result/%s_%s_vqa_result_<epoch>.json' % (args.output_dir, prefix, config['model_variant'])
    compute_vqa_acc(answer_list_path=config[args.dataset_use]['test_file'][0], epoch=config['max_epoch'], res_file_path=res_file_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='./configs/VQA.yaml')
    parser.add_argument('--dataset_use', default='rad', help='choose medical vqa dataset(rad, pathvqa, slake)')
    parser.add_argument('--is_save_path', default=False)
    parser.add_argument('--checkpoint', default='/mnt/sda/lpf/weights/output/V2/pretrain/std/med_pretrain_29.pth')
    parser.add_argument('--output_suffix', default='', help='output suffix, eg. ../rad_29_1')
    parser.add_argument('--output_dir', default='', help='the final output path')
    parser.add_argument('--evaluate', action='store_true')
    parser.add_argument('--text_encoder', default='bert-base-uncased')
    parser.add_argument('--text_decoder', default='bert-base-uncased')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--world_size', default=1, type=int, help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    parser.add_argument('--distributed', default=False, type=bool)
    parser.add_argument('--model_variant', default='')
    parser.add_argument('--train_stage', default='')
    parser.add_argument('--diffusion_model_id', default='')
    parser.add_argument('--diffusion_local_files_only', type=str2bool, default=None)
    parser.add_argument('--diffusion_frozen', type=str2bool, default=None)
    parser.add_argument('--use_global_contrast', type=str2bool, default=None)
    parser.add_argument('--use_answer_contrast', type=str2bool, default=None)
    parser.add_argument('--use_decorrelation', type=str2bool, default=None)
    parser.add_argument('--use_gate_entropy', type=str2bool, default=None)
    parser.add_argument('--new_lr', type=float, default=None)
    parser.add_argument('--backbone_lr', type=float, default=None)
    parser.add_argument('--decoder_lr', type=float, default=None)
    parser.add_argument('--diffusion_backbone_lr', type=float, default=None)
    parser.add_argument('--optimizer_weight_decay', type=float, default=None)
    parser.add_argument('--lambda_global_contrast', type=float, default=None)
    parser.add_argument('--lambda_answer_contrast', type=float, default=None)
    parser.add_argument('--lambda_decorrelation', type=float, default=None)
    parser.add_argument('--lambda_gate_entropy', type=float, default=None)
    parser.add_argument('--unfreeze_visual_last_n_blocks', type=int, default=None)
    parser.add_argument('--unfreeze_text_last_n_layers', type=int, default=None)

    args = parser.parse_args()

    config = yaml.load(open(args.config, 'r'), Loader=yaml.Loader)
    config = apply_runtime_overrides(config, args)

    if not args.output_dir:
        args.output_dir = '/mnt/sda/lpf/weights/output/V2/vqa/' + args.dataset_use + args.output_suffix

    args.result_dir = os.path.join(args.output_dir, 'result')
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    Path(args.result_dir).mkdir(parents=True, exist_ok=True)

    sys.stdout = utils.Logger(filename=os.path.join(args.output_dir, "log.txt"), stream=sys.stdout)

    print("config: ", config)
    print("args: ", args)
    main(args, config)
