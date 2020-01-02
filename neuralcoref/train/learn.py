"""Conll training algorithm"""

import os
import time
import argparse
import socket
from datetime import datetime
import numpy as np

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.optim import RMSprop
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter

from neuralcoref.train.model import Model
from neuralcoref.train.dataset import (
    NCDataset,
    NCBatchSampler,
    load_embeddings_from_file,
    padder_collate,
    SIZE_PAIR_IN,
    SIZE_SINGLE_IN,
)
from neuralcoref.train.utils import SIZE_EMBEDDING
from neuralcoref.train.evaluator import ConllEvaluator

PACKAGE_DIRECTORY = os.path.dirname(os.path.abspath(__file__))
STAGES = ["allpairs", "toppairs", "ranking"]


def clipped_sigmoid(inputs):
    epsilon = 1.0e-7
    return torch.sigmoid(inputs).clamp(epsilon, 1.0 - epsilon)


def get_all_pairs_loss(n):
    def all_pair_loss(scores, targets):
        """ All pairs and single mentions probabilistic loss
        """
        labels = targets[0]
        weights = targets[4].data if len(targets) == 5 else None
        loss_op = nn.BCEWithLogitsLoss(weight=weights, reduction="sum")
        loss = loss_op(scores, labels)
        return loss / n

    return all_pair_loss


def get_top_pair_loss(n):
    def top_pair_loss(scores, targets, debug=False):
        """ Top pairs (best true and best mistaken) and single mention probabilistic loss
        """
        true_ants = targets[2]
        false_ants = targets[3] if len(targets) == 5 else None
        s_scores = clipped_sigmoid(scores)
        true_pairs = torch.gather(s_scores, 1, true_ants)
        top_true, top_true_arg = torch.log(true_pairs).max(
            dim=1
        )  # max(log(p)), p=sigmoid(s)
        if debug:
            print("true_pairs", true_pairs.data)
            print("top_true", top_true.data)
            print("top_true_arg", top_true_arg.data)
        out_score = torch.sum(top_true).neg()
        if (
            false_ants is not None
        ):  # We have no false antecedents when there are no pairs
            false_pairs = torch.gather(s_scores, 1, false_ants)
            top_false, _ = torch.log(1 - false_pairs).min(
                dim=1
            )  # min(log(1-p)), p=sigmoid(s)
            out_score = out_score + torch.sum(top_false).neg()
        return out_score / n

    return top_pair_loss


def get_ranking_loss(n):
    def ranking_loss(scores, targets):
        """ Slack-rescaled max margin loss
        """
        costs = targets[1]
        true_ants = targets[2]
        weights = targets[4] if len(targets) == 5 else None
        true_ant_score = torch.gather(scores, 1, true_ants)
        top_true, _ = true_ant_score.max(dim=1)
        tmp_loss = scores.add(1).add(
            top_true.unsqueeze(1).neg()
        )  # 1 + scores - top_true
        if weights is not None:
            tmp_loss = tmp_loss.mul(weights)
        tmp_loss = tmp_loss.mul(costs)
        loss, _ = tmp_loss.max(dim=1)
        out_score = torch.sum(loss)
        return out_score / n

    return ranking_loss


def decrease_lr(optim_func, factor=0.1, min_lrs=0, eps=0, verbose=True):
    for i, param_group in enumerate(optim_func.param_groups):
        old_lr = float(param_group["lr"])
        new_lr = max(old_lr * factor, min_lrs)
        if old_lr - new_lr > eps:
            param_group["lr"] = new_lr
            if verbose:
                print(f"Reducing learning rate" " of group {i} to {new_lr:.4e}.")
    return new_lr


def load_model(model, path):
    print("⛄️ Reloading model from", path)
    model.load_state_dict(
        torch.load(path)
        if args.cuda
        else torch.load(path, map_location=lambda storage, loc: storage)
    )


def run_model(args):
    print(
        "Training for",
        args.all_pairs_epoch,
        args.top_pairs_epoch,
        args.ranking_epoch,
        "epochs",
    )
    # Tensorboard server
    writer = SummaryWriter()

    # Load datasets and embeddings
    embed_path = args.weights if args.weights is not None else args.train
    tensor_embeddings, voc = load_embeddings_from_file(embed_path + "tuned_word")
    dataset = NCDataset(args.train, args)
    eval_dataset = NCDataset(args.eval, args)
    print("Vocabulary:", len(voc))

    # Construct model
    print("🏝 Build model")
    model = Model(
        len(voc),
        SIZE_EMBEDDING,
        args.h1,
        args.h2,
        args.h3,
        SIZE_PAIR_IN,
        SIZE_SINGLE_IN,
    )
    model.load_embeddings(tensor_embeddings)
    if args.cuda:
        model.cuda()
    if args.weights is not None:
        print("🏝 Loading pre-trained weights")
        model.load_weights(args.weights)
    if args.checkpoint_file is not None:
        print("⛄️ Loading model from", args.checkpoint_file)
        model.load_state_dict(
            torch.load(args.checkpoint_file)
            if args.cuda
            else torch.load(
                args.checkpoint_file, map_location=lambda storage, loc: storage
            )
        )

    print("🏝 Loading conll evaluator")
    eval_evaluator = ConllEvaluator(
        model, eval_dataset, args.eval, args.evalkey, embed_path, args
    )
    train_evaluator = ConllEvaluator(
        model, dataset, args.train, args.trainkey, embed_path, args
    )
    print("🏝 Testing evaluator and getting first eval score")
    eval_evaluator.test_model()
    start_time = time.time()
    eval_evaluator.build_test_file()
    score, f1_conll, ident = eval_evaluator.get_score()
    elapsed = time.time() - start_time
    print(f"|| s/evaluation {elapsed:5.2f}")
    writer.add_scalar("eval/" + "F1_conll", f1_conll, 0)

    # Preparing dataloader
    print("🏝 Preparing dataloader")
    print(
        "Dataloader parameters: batchsize",
        args.batchsize,
        "numworkers",
        args.numworkers,
    )
    batch_sampler = NCBatchSampler(
        dataset.mentions_pair_length, shuffle=True, batchsize=args.batchsize
    )
    dataloader = DataLoader(
        dataset,
        collate_fn=padder_collate,
        batch_sampler=batch_sampler,
        num_workers=args.numworkers,
        pin_memory=args.cuda,
    )
    mentions_idx, n_pairs = batch_sampler.get_batch_info()

    print("🏝 Start training")
    g_step = 0
    start_from = (
        args.startstep
        if args.startstep is not None and args.startstage is not None
        else 0
    )

    def run_epochs(
        start_epoch, end_epoch, loss_func, optim_func, save_name, lr, g_step, debug=None
    ):
        best_model_path = args.save_path + "best_model" + save_name
        start_time_all = time.time()
        best_f1_conll = 0
        lower_eval = 0
        for epoch in range(start_epoch, end_epoch):
            """ Run an epoch """
            print(f"🚘 {save_name} Epoch {epoch:d}")
            model.train()
            start_time_log = time.time()
            start_time_epoch = time.time()
            epoch_loss = 0
            for batch_i, (m_idx, n_pairs_l, batch) in enumerate(
                zip(mentions_idx, n_pairs, dataloader)
            ):
                if debug is not None and (debug == -1 or debug in m_idx):
                    l = list(dataset.flat_m_loc[m][2:] for m in m_idx)
                    print(
                        "🏔 Batch",
                        batch_i,
                        "m_idx:",
                        "|".join(str(i) for i in m_idx),
                        "mentions:",
                        "|".join(dataset.docs[d]["mentions"][i] for u, i, d in l),
                    )
                    print("Batch n_pairs:", "|".join(str(p) for p in n_pairs_l))
                inputs, targets = batch
                inputs = tuple(Variable(inp, requires_grad=False) for inp in inputs)
                targets = tuple(Variable(tar, requires_grad=False) for tar in targets)
                if args.cuda:
                    inputs = tuple(i.cuda() for i in inputs)
                    targets = tuple(t.cuda() for t in targets)
                scores = model(inputs)
                if debug is not None and (debug == -1 or debug in m_idx):
                    print(
                        "Scores:\n"
                        + "\n".join(
                            "|".join(str(s) for s in s_l)
                            for s_l in scores.data.cpu().numpy()
                        )
                    )
                    print(
                        "Labels:\n"
                        + "\n".join(
                            "|".join(str(s) for s in s_l)
                            for s_l in targets[0].data.cpu().numpy()
                        )
                    )
                loss = loss_func(scores, targets)
                if debug is not None and (debug == -1 or debug in m_idx):
                    print("Loss", loss.item())
                # Zero gradients, perform a backward pass, and update the weights.
                optim_func.zero_grad()
                loss.backward()
                epoch_loss += loss.item()
                optim_func.step()
                writer.add_scalar("train/" + save_name + "_loss", loss.item(), g_step)
                writer.add_scalar("meta/" + "lr", lr, g_step)
                writer.add_scalar("meta/" + "stage", STAGES.index(save_name), g_step)
                g_step += 1
                if batch_i % args.log_interval == 0 and batch_i > 0:
                    elapsed = time.time() - start_time_log
                    lr = optim_func.param_groups[0]["lr"]
                    ea = elapsed * 1000 / args.log_interval
                    li = loss.item()
                    print(
                        f"| epoch {epoch:3d} | {batch_i:5d}/{len(dataloader):5d} batches | "
                        f"lr {lr:.2e} | ms/batch {ea:5.2f} | "
                        f"loss {li:.2e}"
                    )
                    start_time_log = time.time()
            elapsed_all = time.time() - start_time_all
            elapsed_epoch = time.time() - start_time_epoch
            ep = elapsed_epoch / 60
            ea = (
                elapsed_all
                / 3600
                * float(end_epoch - epoch)
                / float(epoch - start_epoch + 1)
            )
            print(
                f"|| min/epoch {ep:5.2f} | est. remaining time (h) {ea:5.2f} | loss {epoch_loss:.2e}"
            )
            writer.add_scalar("epoch/" + "loss", epoch_loss, g_step)
            if epoch % args.conll_train_interval == 0:
                start_time = time.time()
                train_evaluator.build_test_file()
                score, f1_conll, ident = train_evaluator.get_score()
                elapsed = time.time() - start_time
                ep = elapsed_epoch / 60
                print(f"|| min/train evaluation {ep:5.2f} | F1_conll {f1_conll:5.2f}")
                writer.add_scalar("epoch/" + "F1_conll", f1_conll, g_step)
            if epoch % args.conll_eval_interval == 0:
                start_time = time.time()
                eval_evaluator.build_test_file()
                score, f1_conll, ident = eval_evaluator.get_score()
                elapsed = time.time() - start_time
                ep = elapsed_epoch / 60
                print(f"|| min/evaluation {ep:5.2f}")
                writer.add_scalar("eval/" + "F1_conll", f1_conll, g_step)
                g_step += 1
                save_path = args.save_path + save_name + "_" + str(epoch)
                torch.save(model.state_dict(), save_path)
                if f1_conll > best_f1_conll:
                    best_f1_conll = f1_conll
                    torch.save(model.state_dict(), best_model_path)
                    lower_eval = 0
                elif args.on_eval_decrease != "nothing":
                    print("Evaluation metric decreases")
                    lower_eval += 1
                    if lower_eval >= args.patience:
                        if (
                            args.on_eval_decrease == "divide_lr"
                            or args.on_eval_decrease == "divide_then_next"
                        ):
                            print("reload best model and decrease lr")
                            load_model(model, best_model_path)
                            lr = decrease_lr(optim_func)
                        if args.on_eval_decrease == "next_stage" or lr <= args.min_lr:
                            print("Switch to next stage")
                            break
        # Save last step
        start_time = time.time()
        eval_evaluator.build_test_file()
        score, f1_conll, ident = eval_evaluator.get_score()
        elapsed = time.time() - start_time
        ep = elapsed / 60
        print(f"|| min/evaluation {ep:5.2f}")
        writer.add_scalar("eval/" + "F1_conll", f1_conll, g_step)
        g_step += 1
        save_path = args.save_path + save_name + "_" + str(epoch)
        torch.save(model.state_dict(), save_path)
        load_model(model, best_model_path)
        return g_step

    if args.startstage is None or args.startstage == "allpairs":
        optimizer = RMSprop(
            model.parameters(), lr=args.all_pairs_lr, weight_decay=args.all_pairs_l2
        )
        loss_func = get_all_pairs_loss(batch_sampler.pairs_per_batch)
        g_step = run_epochs(
            start_from,
            args.all_pairs_epoch,
            loss_func,
            optimizer,
            "allpairs",
            args.all_pairs_lr,
            g_step,
        )
        start_from = 0

    if args.startstage is None or args.startstage in ["allpairs", "toppairs"]:
        optimizer = RMSprop(
            model.parameters(), lr=args.top_pairs_lr, weight_decay=args.top_pairs_l2
        )
        loss_func = get_top_pair_loss(10 * batch_sampler.mentions_per_batch)
        g_step = run_epochs(
            start_from,
            args.top_pairs_epoch,
            loss_func,
            optimizer,
            "toppairs",
            args.top_pairs_lr,
            g_step,
        )
        start_from = 0

    if args.startstage is None or args.startstage in [
        "ranking",
        "allpairs",
        "toppairs",
    ]:
        optimizer = RMSprop(
            model.parameters(), lr=args.ranking_lr, weight_decay=args.ranking_l2
        )
        loss_func = get_ranking_loss(batch_sampler.mentions_per_batch)
        g_step = run_epochs(
            start_from,
            args.ranking_epoch,
            loss_func,
            optimizer,
            "ranking",
            args.ranking_lr,
            g_step,
        )


if __name__ == "__main__":
    DIR_PATH = os.path.dirname(os.path.realpath(__file__))
    parser = argparse.ArgumentParser(
        description="Training the neural coreference model"
    )
    parser.add_argument(
        "--train",
        type=str,
        default=DIR_PATH + "/data/",
        help="Path to the train dataset",
    )
    parser.add_argument(
        "--eval", type=str, default=DIR_PATH + "/data/", help="Path to the eval dataset"
    )
    parser.add_argument(
        "--evalkey", type=str, help="Path to an optional key file for scoring"
    )
    parser.add_argument(
        "--weights",
        type=str,
        help="Path to pre-trained weights (if you only want to test the scoring for e.g.)",
    )
    parser.add_argument(
        "--batchsize",
        type=int,
        default=20000,
        help="Size of a batch in total number of pairs",
    )
    parser.add_argument(
        "--numworkers",
        type=int,
        default=8,
        help="Number of workers for loading batches",
    )
    parser.add_argument(
        "--startstage",
        type=str,
        help='Start from a specific stage ("allpairs", "toppairs", "ranking")',
    )
    parser.add_argument("--startstep", type=int, help="Start from a specific step")
    parser.add_argument(
        "--checkpoint_file",
        type=str,
        help="Start from a previously saved checkpoint file",
    )
    parser.add_argument(
        "--log_interval", type=int, default=10, help="test every X mini-batches"
    )
    parser.add_argument(
        "--conll_eval_interval",
        type=int,
        default=10,
        help="evaluate eval F1 conll every X epochs",
    )
    parser.add_argument(
        "--conll_train_interval",
        type=int,
        default=20,
        help="evaluate train F1 conll every X epochs",
    )
    parser.add_argument("--seed", type=int, default=1111, help="random seed")
    parser.add_argument("--costfn", type=float, default=0.8, help="cost of false new")
    parser.add_argument("--costfl", type=float, default=0.4, help="cost of false link")
    parser.add_argument("--costwl", type=float, default=1.0, help="cost of wrong link")
    parser.add_argument(
        "--h1", type=int, default=1000, help="number of hidden unit on layer 1"
    )
    parser.add_argument(
        "--h2", type=int, default=500, help="number of hidden unit on layer 2"
    )
    parser.add_argument(
        "--h3", type=int, default=500, help="number of hidden unit on layer 3"
    )
    parser.add_argument(
        "--all_pairs_epoch",
        type=int,
        default=200,
        help="number of epochs for all-pairs pre-training",
    )
    parser.add_argument(
        "--top_pairs_epoch",
        type=int,
        default=200,
        help="number of epochs for top-pairs pre-training",
    )
    parser.add_argument(
        "--ranking_epoch",
        type=int,
        default=200,
        help="number of epochs for ranking training",
    )
    parser.add_argument(
        "--all_pairs_lr",
        type=float,
        default=2e-4,
        help="all pairs pre-training learning rate",
    )
    parser.add_argument(
        "--top_pairs_lr",
        type=float,
        default=2e-4,
        help="top pairs pre-training learning rate",
    )
    parser.add_argument(
        "--ranking_lr", type=float, default=2e-6, help="ranking training learning rate"
    )
    parser.add_argument(
        "--all_pairs_l2",
        type=float,
        default=1e-6,
        help="all pairs pre-training l2 regularization",
    )
    parser.add_argument(
        "--top_pairs_l2",
        type=float,
        default=1e-5,
        help="top pairs pre-training l2 regularization",
    )
    parser.add_argument(
        "--ranking_l2",
        type=float,
        default=1e-5,
        help="ranking training l2 regularization",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=3,
        help="patience (epochs) before considering evaluation has decreased",
    )
    parser.add_argument("--min_lr", type=float, default=2e-8, help="min learning rate")
    parser.add_argument(
        "--on_eval_decrease",
        type=str,
        default="nothing",
        help='What to do when evaluation decreases ("nothing", "divide_lr", "next_stage", "divide_then_next")',
    )
    parser.add_argument(
        "--lazy",
        type=int,
        default=1,
        choices=(0, 1),
        help="Use lazy loading (1, default) or not (0) while loading the npy files",
    )
    args = parser.parse_args()
    args.costs = {"FN": args.costfn, "FL": args.costfl, "WL": args.costwl}
    args.lazy = bool(args.lazy)
    current_time = datetime.now().strftime("%b%d_%H-%M-%S")
    args.save_path = os.path.join(
        PACKAGE_DIRECTORY,
        "checkpoints",
        current_time + "_" + socket.gethostname() + "_",
    )

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    args.cuda = torch.cuda.is_available()
    if args.cuda:
        torch.cuda.manual_seed(args.seed)

    args.evalkey = args.evalkey if args.evalkey is not None else args.eval + "/key.txt"
    args.trainkey = args.train + "/key.txt"
    args.train = args.train + "/numpy/"
    args.eval = args.eval + "/numpy/"
    print(args)
    run_model(args)
