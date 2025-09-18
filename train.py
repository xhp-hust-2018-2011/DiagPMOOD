import os, sys
TMP_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(TMP_DIR)
ROOT_DIR = os.path.dirname(os.path.abspath(TMP_DIR))
sys.path.append(ROOT_DIR)
import datetime
import os
import time
import warnings

# import presets
import torch
import torch.utils.data
import torchvision
import torchvision.transforms
import utils
from sampler import RASampler
from torch import nn
import torch.nn.functional as F
from torch.utils.data.dataloader import default_collate
from torchvision.transforms.functional import InterpolationMode
from xhp_utils import get_fclayer
from xhp_utils import init_weights_kaiming
from torch.hub import load_state_dict_from_url
from torchvision.models import get_model_weights
import presets_v2
import numpy as np
import torchvision.transforms as tvs_trans
from imglist_dataset_b2 import ImglistDataDataset
from transforms_modified0818 import get_mixup_cutmix
from resnet18_32x32 import ResNet18_32x32
from resnet18_224x224 import ResNet18_224x224
from resnet50 import ResNet50
from swin_t import Swin_T
import copy

def compute_entropy(p,if_softmax=False,min_real=1e-10):
    if if_softmax:
        prob = F.softmax(p,dim=1)
    else:
        prob = p
    plog = prob*torch.log(torch.clamp(prob,min=min_real))
    plog = plog.sum(dim=1,keepdim=True)
    return -plog


def train_one_epoch(model, criterion, optimizer,lr_scheduler, data_loader, device, epoch, args, model_ema=None, scaler=None,log_path='train_log.txt',\
    batch_transform=None,batch_seperate=True):
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", utils.SmoothedValue(window_size=1, fmt="{value}"))
    metric_logger.add_meter("img/s", utils.SmoothedValue(window_size=10, fmt="{value}"))

    header = f"Epoch: [{epoch}]"
    cnum = len(data_loader.dataset.classes)
    for i,sample  in enumerate(metric_logger.log_every(data_loader, args.print_freq, header,log_path=log_path)):
        start_time = time.time()
        image, target = sample['data'],sample['label']
        image, target = image.to(device), target.to(device)
        image_test = sample['data_test']
        image_test = image_test.to(device)
        # target should be one-hot
        if len(target.size())==1:
            target = F.one_hot(target,num_classes=cnum).float()

        if (not (batch_transform is None)):
            idx = int( torch.randint(0,len(batch_transform),(1,)).item() )
            image_t,target_t = image_test,target
            tmp_batch_transform = batch_transform[idx]
            image_t,target_t = tmp_batch_transform(image_t,target_t)
            
        with torch.cuda.amp.autocast(enabled=scaler is not None):
            if not (batch_transform is None): 
                idx = int( torch.randint(0,2,(1,)).item() )
                if idx ==0:
                    image = torch.cat((image_test,image_t),dim=0)
                else:
                    image = torch.cat((image,image_t),dim=0)
            else:
                image = image  
            # whether concat the target                 
            B1,B2 = target.size()[0],target_t.size()[0]
            if not (batch_transform is None):          
                target = torch.cat((target,target_t),dim=0)
            output = model(image)
            loss = criterion(output, target)

            if not (batch_transform is None):
                b1_logsumexp = torch.logsumexp(output[0:B1],dim=1,keepdim=True)
                b2_logsumexp = torch.logsumexp(output[B1:B1+B2],dim=1,keepdim=True)
                b12_logsumexp = torch.logsumexp(\
                                            torch.cat([output[0:B1],output[B1:B1+B2]],dim=1),\
                                            dim=1,keepdim=True)
                b1_target = target[0:B1].max(dim=1,keepdim=True)[0]
                b2_target = target[B1:B1+B2].max(dim=1,keepdim=True)[0]
                bsum_target = b1_target+b2_target
                b1_target = b1_target/bsum_target
                b2_target = b2_target/bsum_target

                norm_factor = -(b1_target*torch.log(b1_target)+b2_target*torch.log(b2_target)).mean()
                bb_loss = -(b1_target*b1_logsumexp+b2_target*b2_logsumexp-b12_logsumexp)
                bb_loss = bb_loss.mean()-norm_factor.detach()
                loss = loss+bb_loss

        optimizer.zero_grad()
        if scaler is not None:
            scaler.scale(loss).backward()
            if args.clip_grad_norm is not None:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if args.clip_grad_norm is not None:
                nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad_norm)
            optimizer.step()
        lr_scheduler.step()

        if model_ema and i % args.model_ema_steps == 0:
            model_ema.update_parameters(model)
            if epoch < args.lr_warmup_epochs:
                # Reset ema buffer to keep copying weights during warmup period
                model_ema.n_averaged.fill_(0)

        acc1, acc5 = utils.accuracy(output, target, topk=(1, 5))
        batch_size = image.shape[0]
        metric_logger.update(loss=loss.item(), lr=optimizer.param_groups[0]["lr"])
        metric_logger.meters["acc1"].update(acc1.item(), n=batch_size)
        metric_logger.meters["acc5"].update(acc5.item(), n=batch_size)
        metric_logger.meters["img/s"].update(batch_size / (time.time() - start_time))


def evaluate(model, criterion, data_loader, device, print_freq=100, log_suffix=""):
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = f"Test: {log_suffix}"

    num_processed_samples = 0
    with torch.inference_mode():
        for sample in metric_logger.log_every(data_loader, print_freq, header):
            image, target = sample['data'],sample['label']
            image = image.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            output = model(image)
            loss = criterion(output, target)

            acc1, acc5 = utils.accuracy(output, target, topk=(1, 5))
            batch_size = image.shape[0]
            metric_logger.update(loss=loss.item())
            metric_logger.meters["acc1"].update(acc1.item(), n=batch_size)
            metric_logger.meters["acc5"].update(acc5.item(), n=batch_size)
            num_processed_samples += batch_size

    num_processed_samples = utils.reduce_across_processes(num_processed_samples)
    if (
        hasattr(data_loader.dataset, "__len__")
        and len(data_loader.dataset) != num_processed_samples
        and torch.distributed.get_rank() == 0
    ):
        # See FIXME above
        warnings.warn(
            f"It looks like the dataset has {len(data_loader.dataset)} samples, but {num_processed_samples} "
            "samples were used for the validation, which might bias the results. "
            "Try adjusting the batch size and / or the world size. "
            "Setting the world size to 1 is always a safe bet."
        )

    metric_logger.synchronize_between_processes()

    print(f"{header} Acc@1 {metric_logger.acc1.global_avg:.3f} Acc@5 {metric_logger.acc5.global_avg:.3f}")
    return metric_logger.acc1.global_avg


class Convert:
    def __init__(self, mode='RGB'):
        self.mode = mode

    def __call__(self, image):
        return image.convert(self.mode)

def load_data_v2(args):
    # Data loading code
    print("Loading data")
    interpolation = InterpolationMode(args.interpolation)

    print("Loading training data")
    st = time.time()

    auto_augment_policy = getattr(args, "auto_augment", None)
    random_erase_prob = getattr(args, "random_erase", 0.0)
    ra_magnitude = getattr(args, "ra_magnitude", None)
    augmix_severity = getattr(args, "augmix_severity", None)

    if 'cifar' in args.idname:
        pre_size=32
        image_size=32
        pre_size_test=32
        image_size_test=32
    elif 'imagenet' in args.idname:
        pre_size=256
        image_size=224
        pre_size_test=256
        image_size_test=224

    normalization_dict = {
        'cifar10': [[0.4914, 0.4822, 0.4465], [0.2470, 0.2435, 0.2616]],
        'cifar100': [[0.5071, 0.4867, 0.4408], [0.2675, 0.2565, 0.2761]],
        'imagenet': [[0.485, 0.456, 0.406], [0.229, 0.224, 0.225]],
        'imagenet200': [[0.485, 0.456, 0.406], [0.229, 0.224, 0.225]],
        'covid': [[0.4907, 0.4907, 0.4907], [0.2697, 0.2697, 0.2697]],
        'aircraft': [[0.5, 0.5, 0.5], [0.5, 0.5, 0.5]],
        'cub': [[0.5, 0.5, 0.5], [0.5, 0.5, 0.5]],
        'cars': [[0.5, 0.5, 0.5], [0.5, 0.5, 0.5]],
    }    
    mean,std = normalization_dict[args.idname][0],normalization_dict[args.idname][1]

    
    interpolation = tvs_trans.InterpolationMode.BILINEAR
    transforms = []
    if 'cifar' in args.idname:
        transforms += [Convert('RGB'),
            tvs_trans.Resize(pre_size,
                                interpolation=interpolation),
            tvs_trans.CenterCrop(image_size),
            tvs_trans.RandomHorizontalFlip(),
            tvs_trans.RandomCrop(image_size, padding=4),]
    elif 'imagenet' in args.idname:
        transforms += [Convert('RGB'),
            tvs_trans.RandomResizedCrop(image_size,
                                    interpolation=interpolation),
            tvs_trans.RandomHorizontalFlip(),]
    transforms_train_test = copy.deepcopy(transforms)

    if auto_augment_policy is not None:
        if auto_augment_policy == "ra":
            transforms.append(tvs_trans.RandAugment(interpolation=interpolation, magnitude=ra_magnitude))
        elif auto_augment_policy == "ta_wide":
            transforms.append(tvs_trans.TrivialAugmentWide(interpolation=interpolation))
        elif auto_augment_policy == "augmix":
            transforms.append(tvs_trans.AugMix(interpolation=interpolation, severity=augmix_severity))
        else:
            aa_policy = tvs_trans.AutoAugmentPolicy(auto_augment_policy)
            transforms.append(tvs_trans.AutoAugment(policy=aa_policy, interpolation=interpolation))

    transforms += [        tvs_trans.ToTensor(),
        tvs_trans.Normalize(mean=mean, std=std),]
    transforms_train_test += [   tvs_trans.ToTensor(),
        tvs_trans.Normalize(mean=mean, std=std),]
    if random_erase_prob > 0:
        transforms.append(tvs_trans.RandomErasing(p=random_erase_prob))

    transform = tvs_trans.Compose(transforms)
    transform_train_test= tvs_trans.Compose(transforms_train_test)

    transform_test = tvs_trans.Compose([
        Convert('RGB'),
        tvs_trans.Resize(pre_size_test, interpolation=tvs_trans.InterpolationMode.BILINEAR),
        tvs_trans.CenterCrop(image_size_test),
        tvs_trans.ToTensor(),
        tvs_trans.Normalize(mean=mean, std=std),
    ])


    data_dir = args.data_dir
    imglist_pth_train = os.path.join(os.path.join(args.imglist_dir,args.idname),
                                    f"train_{args.idname}.txt")
    imglist_pth_val = os.path.join(os.path.join(args.imglist_dir,args.idname),
                                    f"val_{args.idname}.txt")
    imglist_pth_test = os.path.join(os.path.join(args.imglist_dir,args.idname),
                                    f"test_{args.idname}.txt")    

    dataset = ImglistDataDataset(args.idname,
                 imglist_pth_train,
                 data_dir,
                 transform,
                 transform_train_test)

    print("Took", time.time() - st)

    print("Loading validation data")
    # actually, this is validation set
    dataset_val = ImglistDataDataset(args.idname,
                 imglist_pth_val,
                 data_dir,
                 transform_test,
                 transform_test)
    dataset_test = ImglistDataDataset(args.idname,
                 imglist_pth_test,
                 data_dir,
                 transform_test,
                 transform_test)

    print("Creating data loaders")

    train_sampler = torch.utils.data.RandomSampler(dataset)
    val_sampler = torch.utils.data.RandomSampler(dataset_val)
    test_sampler = torch.utils.data.SequentialSampler(dataset_test)
    return dataset, dataset_val, dataset_test, train_sampler,val_sampler, test_sampler

def main(args):
    # set the random_seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed) 

    args.output_dir = args.output_dir + f"_s{args.seed}"

    if args.output_dir:
        utils.mkdir(args.output_dir)

    utils.init_distributed_mode(args)
    print(args)

    device = torch.device(args.device)

    if args.use_deterministic_algorithms:
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True)
    else:
        torch.backends.cudnn.benchmark = True
    
    if 'imagenet' in args.idname:
        if args.idname == 'imagenet':
            cnum = 1000
        elif args.idname == 'imagenet200':
            cnum = 200
        args.cnum = cnum
    elif 'cifar' in args.idname:
        if args.idname == 'cifar10':
            cnum = 10
        elif args.idname == 'cifar100':
            cnum = 100
        args.cnum = cnum
    else:
        print('No such dataset')       
    dataset, dataset_val,dataset_test, train_sampler,val_sampler, test_sampler = load_data_v2(args)

    num_classes = cnum
    dataset.classes = list(range(cnum)) 
    dataset.cnum = cnum
    batch_transform = []
    if args.mixup_alpha>0 or args.cutmix_alpha>0:
        mixup_cutmix = get_mixup_cutmix(
            mixup_alpha=args.mixup_alpha, cutmix_alpha=args.cutmix_alpha, num_classes=num_classes, use_v2=args.use_v2
        )
        batch_transform.append(mixup_cutmix)

    if len(batch_transform)==0:
        batch_transform = None


    collate_fn = default_collate

    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        sampler=train_sampler,
        num_workers=args.workers,
        pin_memory=True,
        collate_fn=collate_fn,
    )
    data_loader_val = torch.utils.data.DataLoader(
        dataset_val, batch_size=args.batch_size, sampler=val_sampler, num_workers=args.workers, pin_memory=True
    )    
    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=args.batch_size, sampler=test_sampler, num_workers=args.workers, pin_memory=True
    )

    print("Creating model")
    if args.model in ["resnet18_32x32",
                      "resnet18_224x224",
                      "resnet50",
                      'swin_t']:
        if args.model == "resnet18_32x32":
            model = ResNet18_32x32(num_classes=len(dataset.classes))
        elif args.model == "resnet18_224x224":
            model = ResNet18_224x224(num_classes=len(dataset.classes))
        elif args.model == "resnet50":
            model = ResNet50(num_classes=len(dataset.classes))
        elif args.model == 'swin_t':
            model = Swin_T(num_classes=len(dataset.classes))
        else:
            print('No such models!')
    else:
        model = torchvision.models.get_model(args.model, weights=None, num_classes=num_classes)
    
    # here load the model weights
    if args.tvs_pretrained:
        weights = get_model_weights(args.model)[f'IMAGENET1K_V{args.tvs_version}']
        weights = load_state_dict_from_url(weights.url)
        model.load_state_dict(weights)
    else:
        if not (args.ckpt_path is None):
            ckpt = torch.load(args.ckpt_path, map_location='cpu')
            try:
                ckpt = ckpt['model']
            except:
                ckpt = ckpt        
            model.load_state_dict(ckpt)   
    if args.train_params == 'fc':
        try:
            fc_layer = get_fclayer(model)
        except: 
            fc_layer = model.get_fc_layer()
        fc_layer.apply(init_weights_kaiming)

    # =======================================================================
    # here if finetune only fc, should specify the fc layer name
    # =======================================================================
    if args.train_params == 'all':
        for param in model.parameters():
            param.requires_grad = True
    elif args.train_params == 'fc':
        for param in model.parameters():
            param.requires_grad = False
        fc_layer = get_fclayer(model)
        for param in fc_layer.parameters():
            param.requires_grad = True
        for module in model.modules():
            if isinstance(module, nn.Dropout):
                module.eval()
            if isinstance(module, nn.BatchNorm2d):
                module.eval()
    else:
        print("No such train params setting.")
        
    model.to(device)

    if args.distributed and args.sync_bn:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)

    custom_keys_weight_decay = []
    if args.bias_weight_decay is not None:
        custom_keys_weight_decay.append(("bias", args.bias_weight_decay))
    if args.transformer_embedding_decay is not None:
        for key in ["class_token", "position_embedding", "relative_position_bias_table"]:
            custom_keys_weight_decay.append((key, args.transformer_embedding_decay))
    

    # this will automatically filter out p.requires_grad=False
    parameters = utils.set_weight_decay(
        model,
        args.weight_decay,
        norm_weight_decay=args.norm_weight_decay,
        custom_keys_weight_decay=custom_keys_weight_decay if len(custom_keys_weight_decay) > 0 else None,
    )

    opt_name = args.opt.lower()
    if opt_name.startswith("sgd"):
        optimizer = torch.optim.SGD(
            parameters,
            lr=args.lr,
            momentum=args.momentum,
            weight_decay=args.weight_decay,
            nesterov="nesterov" in opt_name,
        )
    elif opt_name == "rmsprop":
        optimizer = torch.optim.RMSprop(
            parameters, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay, eps=0.0316, alpha=0.9
        )
    elif opt_name == "adamw":
        optimizer = torch.optim.AdamW(parameters, lr=args.lr, weight_decay=args.weight_decay)
    else:
        raise RuntimeError(f"Invalid optimizer {args.opt}. Only SGD, RMSprop and AdamW are supported.")

    scaler = torch.cuda.amp.GradScaler() if args.amp else None

    def cosine_annealing(step, total_steps, lr_max, lr_min):
        return lr_min + (lr_max - lr_min) * 0.5 * \
                (1 + np.cos(step / total_steps * np.pi))
    main_lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda step: cosine_annealing(
            step,
            args.epochs*len(data_loader),
            1,
            1e-6 / args.lr,
        ),
    )

    if args.lr_warmup_epochs > 0:
        if args.lr_warmup_method == "linear":
            warmup_lr_scheduler = torch.optim.lr_scheduler.LinearLR(
                optimizer, start_factor=args.lr_warmup_decay, total_iters=args.lr_warmup_epochs
            )
        elif args.lr_warmup_method == "constant":
            warmup_lr_scheduler = torch.optim.lr_scheduler.ConstantLR(
                optimizer, factor=args.lr_warmup_decay, total_iters=args.lr_warmup_epochs
            )
        else:
            raise RuntimeError(
                f"Invalid warmup lr method '{args.lr_warmup_method}'. Only linear and constant are supported."
            )
        lr_scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer, schedulers=[warmup_lr_scheduler, main_lr_scheduler], milestones=[args.lr_warmup_epochs]
        )
    else:
        lr_scheduler = main_lr_scheduler

    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module

    model_ema = None
    if args.model_ema:
        adjust = args.world_size * args.batch_size * args.model_ema_steps / args.epochs
        alpha = 1.0 - args.model_ema_decay
        alpha = min(1.0, alpha * adjust)
        model_ema = utils.ExponentialMovingAverage(model_without_ddp, device=device, decay=1.0 - alpha)

    if args.resume:
        checkpoint = torch.load(args.resume, map_location="cpu", weights_only=True)
        model_without_ddp.load_state_dict(checkpoint["model"])
        if not args.test_only:
            optimizer.load_state_dict(checkpoint["optimizer"])
            lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
        args.start_epoch = checkpoint["epoch"] + 1
        if model_ema:
            model_ema.load_state_dict(checkpoint["model_ema"])
        if scaler:
            scaler.load_state_dict(checkpoint["scaler"])

    if args.test_only:
        if model_ema:
            evaluate(model_ema, criterion, data_loader_test, device=device, log_suffix="EMA")
        else:
            evaluate(model, criterion, data_loader_test, device=device)
        return

    best_acc1 = 0.0
    best_epoch = -1
    print("Start training")
    start_time = time.time()
    log_path = os.path.join(args.output_dir,'train_log.txt')
    for epoch in range(args.start_epoch, args.epochs):
        train_one_epoch(model, criterion, optimizer,lr_scheduler, data_loader, device, 
                        epoch, args, model_ema, scaler,log_path=log_path,
                        batch_transform=batch_transform,batch_seperate=args.batch_seperate)
        tmp_acc1 = evaluate(model, criterion, data_loader_val, device=device)
        if model_ema:
            evaluate(model_ema, criterion, data_loader_val, device=device, log_suffix="EMA")
        if args.output_dir:
            checkpoint = {
                "model": model_without_ddp.state_dict(),
                "optimizer": optimizer.state_dict(),
                "lr_scheduler": lr_scheduler.state_dict(),
                "epoch": epoch,
                "args": args,
            }
            if model_ema:
                checkpoint["model_ema"] = model_ema.state_dict()
            if scaler:
                checkpoint["scaler"] = scaler.state_dict()
            utils.save_on_master(checkpoint, os.path.join(args.output_dir, "checkpoint.pth"))
            if best_acc1<=tmp_acc1:
                utils.save_on_master(checkpoint, os.path.join(args.output_dir, "model_best.pth"))
                best_acc1 = tmp_acc1
                best_epoch = epoch
        tmp_print_str= f'Best acc1:{best_acc1:.2f}, epoch_{best_epoch}' +'\n'
        print(tmp_print_str)
        if not os.path.exists(log_path):
            with open(log_path, "w") as file:
                file.write(tmp_print_str)
        else:
            with open(log_path, "a") as file:
                file.write(tmp_print_str)

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    tmp_print_str= f'Best acc1:{best_acc1:.2f}' +'\n'
    tmp_print_str+= f"Training time {total_time_str}" +'\n'
    print(tmp_print_str)
    if not os.path.exists(log_path):
        with open(log_path, "w") as file:
            file.write(tmp_print_str)
    else:
        with open(log_path, "a") as file:
            file.write(tmp_print_str)

    # load the best model and test on the test set
    checkpoint = torch.load(os.path.join(args.output_dir, "model_best.pth"), map_location="cpu")
    model_without_ddp.load_state_dict(checkpoint["model"])    
    if model_ema:
        tmp_acc1 = evaluate(model_ema, criterion, data_loader_test, device=device, log_suffix="EMA")
    else:
        tmp_acc1 = evaluate(model, criterion, data_loader_test, device=device)
    tmp_print_str= f'Test Best acc1:{tmp_acc1:.2f}' +'\n'
    print(tmp_print_str)
    if not os.path.exists(log_path):
        with open(log_path, "w") as file:
            file.write(tmp_print_str)
    else:
        with open(log_path, "a") as file:
            file.write(tmp_print_str)

def get_args_parser(add_help=True):
    import argparse

    parser = argparse.ArgumentParser(description="PyTorch Classification Training", add_help=add_help)
    parser.add_argument("--model", default="resnet18", type=str, help="model name")
    parser.add_argument("--device", default="cuda", type=str, help="device (Use cuda or cpu Default: cuda)")
    parser.add_argument(
        "-b", "--batch-size", default=32, type=int, help="images per gpu, the total batch size is $NGPU x batch_size"
    )
    parser.add_argument("--epochs", default=100, type=int, metavar="N", help="number of total epochs to run")
    parser.add_argument(
        "-j", "--workers", default=4, type=int, metavar="N", help="number of data loading workers (default: 16)"
    )
    parser.add_argument("--opt", default="sgd-nesterov", type=str, help="optimizer")
    parser.add_argument("--lr", default=0.1, type=float, help="initial learning rate")
    parser.add_argument("--momentum", default=0.9, type=float, metavar="M", help="momentum")
    parser.add_argument(
        "--wd",
        "--weight-decay",
        default=5e-4,
        type=float,
        metavar="W",
        help="weight decay (default: 1e-4)",
        dest="weight_decay",
    )
    parser.add_argument(
        "--norm-weight-decay",
        default=None,
        type=float,
        help="weight decay for Normalization layers (default: None, same value as --wd)",
    )
    parser.add_argument(
        "--bias-weight-decay",
        default=None,
        type=float,
        help="weight decay for bias parameters of all layers (default: None, same value as --wd)",
    )
    parser.add_argument(
        "--transformer-embedding-decay",
        default=None,
        type=float,
        help="weight decay for embedding parameters for vision transformer models (default: None, same value as --wd)",
    )
    parser.add_argument(
        "--label-smoothing", default=0.0, type=float, help="label smoothing (default: 0.0)", dest="label_smoothing"
    )
    parser.add_argument("--mixup-alpha", default=0.0, type=float, help="mixup alpha (default: 0.0)")
    parser.add_argument("--cutmix-alpha", default=0.0, type=float, help="cutmix alpha (default: 0.0)")
    parser.add_argument("--lr-scheduler", default="steplr", type=str, help="the lr scheduler (default: steplr)")
    parser.add_argument("--lr-warmup-epochs", default=0, type=int, help="the number of epochs to warmup (default: 0)")
    parser.add_argument(
        "--lr-warmup-method", default="constant", type=str, help="the warmup method (default: constant)"
    )
    parser.add_argument("--lr-warmup-decay", default=0.01, type=float, help="the decay for lr")
    parser.add_argument("--lr-step-size", default=30, type=int, help="decrease lr every step-size epochs")
    parser.add_argument("--lr-gamma", default=0.1, type=float, help="decrease lr by a factor of lr-gamma")
    parser.add_argument("--lr-min", default=0.0, type=float, help="minimum lr of lr schedule (default: 0.0)")
    parser.add_argument("--print-freq", default=100, type=int, help="print frequency")
    parser.add_argument("--output-dir", default="./debug/", type=str, help="path to save outputs")
    parser.add_argument("--resume", default="", type=str, help="path of checkpoint")
    parser.add_argument("--start-epoch", default=0, type=int, metavar="N", help="start epoch")
    parser.add_argument(
        "--cache-dataset",
        dest="cache_dataset",
        help="Cache the datasets for quicker initialization. It also serializes the transforms",
        action="store_true",
    )
    parser.add_argument(
        "--sync-bn",
        dest="sync_bn",
        help="Use sync batch norm",
        action="store_true",
    )
    parser.add_argument(
        "--test-only",
        dest="test_only",
        help="Only test the model",
        action="store_true",
    )
    parser.add_argument("--auto-augment", default=None, type=str, help="auto augment policy (default: None)")
    parser.add_argument("--ra-magnitude", default=9, type=int, help="magnitude of auto augment policy")
    parser.add_argument("--augmix-severity", default=3, type=int, help="severity of augmix policy")
    parser.add_argument("--random-erase", default=0.0, type=float, help="random erasing probability (default: 0.0)")

    # Mixed precision training parameters
    parser.add_argument("--amp", action="store_true", help="Use torch.cuda.amp for mixed precision training")

    # distributed training parameters
    parser.add_argument("--world-size", default=1, type=int, help="number of distributed processes")
    parser.add_argument("--dist-url", default="env://", type=str, help="url used to set up distributed training")
    parser.add_argument(
        "--model-ema", action="store_true", help="enable tracking Exponential Moving Average of model parameters"
    )
    parser.add_argument(
        "--model-ema-steps",
        type=int,
        default=32,
        help="the number of iterations that controls how often to update the EMA model (default: 32)",
    )
    parser.add_argument(
        "--model-ema-decay",
        type=float,
        default=0.99998,
        help="decay factor for Exponential Moving Average of model parameters (default: 0.99998)",
    )
    parser.add_argument(
        "--use-deterministic-algorithms", action="store_true", help="Forces the use of deterministic algorithms only."
    )
    parser.add_argument(
        "--interpolation", default="bilinear", type=str, help="the interpolation method (default: bilinear)"
    )
    parser.add_argument(
        "--val-resize-size", default=256, type=int, help="the resize size used for validation (default: 256)"
    )
    parser.add_argument(
        "--val-crop-size", default=224, type=int, help="the central crop size used for validation (default: 224)"
    )
    parser.add_argument(
        "--train-crop-size", default=224, type=int, help="the random crop size used for training (default: 224)"
    )
    parser.add_argument("--clip-grad-norm", default=None, type=float, help="the maximum gradient norm (default None)")
    parser.add_argument("--ra-sampler", action="store_true", help="whether to use Repeated Augmentation in training")
    parser.add_argument(
        "--ra-reps", default=3, type=int, help="number of repetitions for Repeated Augmentation (default: 3)"
    )
    parser.add_argument("--weights", default=None, type=str, help="the weights enum name to load")
    parser.add_argument("--backend", default="PIL", type=str.lower, help="PIL or tensor - case insensitive")
    parser.add_argument("--use-v2", action="store_true", help="Use V2 transforms")
    
    # 2407
    parser.add_argument("--idname", default="cifar100", type=str,
                        choices=['cifar10','cifar100','imagenet200','imagenet'],
                         help="the dataset, this decide the data preprocess")    
    parser.add_argument("--data_dir", default=r"/mnt/d/BaiduYunDownload/Openoodv1.5/data/images_classic", type=str, help="dataset path") 
    parser.add_argument("--imglist_dir", default=r"/mnt/d/BaiduYunDownload/Openoodv1.5/data/benchmark_imglist", type=str, help="dataset path")    
    parser.add_argument("--train_params", default="all", type=str,
                        choices=['fc','all'],
                         help="train all the parameter or just the fc layer")
    parser.add_argument('--tvs_version', default=1,type=str,
                        help='this is only useful for imagenet')#, choices=[1, 2]
    parser.add_argument('--tvs_pretrained', action='store_true',default=False)
    parser.add_argument('--ckpt_path',
                        default=None, 
                        help='this is loading the checkpoints if not using tvs model')
    parser.add_argument('--seed', type=int, default=99999,
                    help='the seed for training')
    parser.add_argument('--batch_seperate', action='store_true',default=False)
    return parser


if __name__ == "__main__":
    args = get_args_parser().parse_args()
    main(args)
