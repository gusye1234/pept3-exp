"""
enable training for semi-supervised learning
"""
import os
import argparse
from sklearn.utils import shuffle
import torch
import json
import sys
from tqdm import tqdm
import numpy as np
from . import helper
from .dataset import FragDataset, IrtDataset, SemiDataset, SemiDataset_twofold, SemiDataset_nfold, pDeep_nfold
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from copy import deepcopy


def semisupervised_finetune(model, input_table, batch_size=2048, gpu_index=0, max_epochs=10,
                            update_interval=1, q_threshold=0.1, validate_q_threshold=0.01, pearson=False,
                            enable_test=False):
    helper.set_seed(2022)
    if torch.cuda.is_available():
        device = torch.device(f"cuda:{gpu_index}")
    else:
        device = torch.device("cpu")
    print("Run on", device)

    model = deepcopy(model)
    model = model.train()
    train_data = SemiDataset(input_table)
    infer_loader = DataLoader(
        train_data.train_all_data(), batch_size=batch_size, shuffle=False)
    if enable_test:
        test_infer_loader = DataLoader(
            train_data.test_all_data(), batch_size=batch_size, shuffle=False)

    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, eps=1e-8)
    loss_fn = helper.FinetuneSALoss(pearson=pearson)
    model = model.to(device)

    print("Baseline Andromeda: ", end='')
    q_values = train_data.Q_values()
    print(np.sum(q_values < 0.001), np.sum(
        q_values < 0.01), np.sum(q_values < 0.1))
    if enable_test:
        q_values = train_data.Q_values_test()
        print(np.sum(q_values < 0.001), np.sum(
            q_values < 0.01), np.sum(q_values < 0.1))
    print("---------------")
    best_model = None
    best_q_value_num = 0
    for epoch in range(max_epochs):
        # print(f"Iteration [{epoch:2d}/{max_epochs:2d}]")
        loss = 0
        loss_l1 = 0.
        loss_sa = 0.
        if (epoch % update_interval) == 0:
            with torch.no_grad():
                scores = []
                for i, data in enumerate(infer_loader):
                    data = {k: v.to(device) for k, v in data.items()}
                    data["peptide_mask"] = helper.create_mask(
                        data['sequence_integer'])
                    pred = model(data)
                    if not pearson:
                        sas = helper.spectral_angle(
                            data['intensities_raw'], pred)
                    else:
                        sas = helper.pearson_coff(
                            data['intensities_raw'], pred)
                    scores.append(sas.detach().cpu().numpy())
                scores = np.concatenate(scores, axis=0)
                train_data.assign_train_score(scores)
                q_values = train_data.Q_values()
                q_values_num = np.sum(q_values < validate_q_threshold)
                if q_values_num > best_q_value_num:
                    best_model = deepcopy(model)
                    best_q_value_num = q_values_num
                    print("Achieve best: ", end='')
                print("Train:", np.sum(q_values < 0.001), np.sum(
                    q_values < 0.01), np.sum(q_values < 0.1))
                train_loader = DataLoader(train_data.semisupervised_sa_finetune(
                    threshold=q_threshold), batch_size=batch_size, shuffle=True)

                if enable_test:
                    with torch.no_grad():
                        scores = []
                        for i, data in enumerate(test_infer_loader):
                            data = {k: v.to(device) for k, v in data.items()}
                            data["peptide_mask"] = helper.create_mask(
                                data['sequence_integer'])
                            pred = model(data)
                            if not pearson:
                                sas = helper.spectral_angle(
                                    data['intensities_raw'], pred)
                            else:
                                sas = helper.pearson_coff(
                                    data['intensities_raw'], pred)
                            scores.append(sas.detach().cpu().numpy())
                        scores = np.concatenate(scores, axis=0)
                        train_data.assign_test_score(scores)
                        test_q_values = train_data.Q_values_test()
                        print(" Test:", np.sum(test_q_values < 0.001),
                              np.sum(test_q_values < 0.01), np.sum(test_q_values < 0.1))
            # train_loader = DataLoader(train_data.semisupervised_sa_finetune_noneg(
            # ), batch_size=batch_size, shuffle=True)
        for i, data in enumerate(train_loader):
            train_count = i + 1
            data = {k: v.to(device) for k, v in data.items()}
            data["peptide_mask"] = helper.create_mask(
                data['sequence_integer'])
            pred = model(data)
            loss_b, fine_loss, l1_loss = loss_fn(
                data['intensities_raw'], pred, data['label'])
            optimizer.zero_grad()
            loss_b.backward()
            optimizer.step()
            sys.stdout.flush()
            # print(
            #     f"\r    -Train Loss {loss/train_count:.3f}, {loss_l1/train_count:.3f}, {loss_sa/train_count:.3f}", end="")
            loss += loss_b.item()
            loss_l1 += l1_loss
            loss_sa += fine_loss
        # print()
    return best_model, train_data.id2remove()


def semisupervised_finetune_random_match(model, input_table, batch_size=2048, gpu_index=0, max_epochs=10,
                                         update_interval=1, q_threshold=0.1, validate_q_threshold=0.01, pearson=False,
                                         enable_test=False):
    helper.set_seed(2022)
    if torch.cuda.is_available():
        device = torch.device(f"cuda:{gpu_index}")
    else:
        device = torch.device("cpu")
    print("Run on", device)

    model = deepcopy(model)
    model = model.train()
    train_data = SemiDataset(input_table)
    train_data.shuffle()
    infer_loader = DataLoader(
        train_data.train_all_data(), batch_size=batch_size, shuffle=False)
    if enable_test:
        test_infer_loader = DataLoader(
            train_data.test_all_data(), batch_size=batch_size, shuffle=False)

    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, eps=1e-8)
    loss_fn = helper.FinetuneSALoss(pearson=pearson)
    model = model.to(device)

    print("Baseline Andromeda: ", end='')
    q_values = train_data.Q_values()
    print(np.sum(q_values < 0.001), np.sum(
        q_values < 0.01), np.sum(q_values < 0.1))
    if enable_test:
        q_values = train_data.Q_values_test()
        print(np.sum(q_values < 0.001), np.sum(
            q_values < 0.01), np.sum(q_values < 0.1))
    print("---------------")
    best_model = None
    best_q_value_num = 0
    for epoch in range(max_epochs):
        # print(f"Iteration [{epoch:2d}/{max_epochs:2d}]")
        loss = 0
        loss_l1 = 0.
        loss_sa = 0.
        if (epoch % update_interval) == 0:
            with torch.no_grad():
                scores = []
                for i, data in enumerate(infer_loader):
                    data = {k: v.to(device) for k, v in data.items()}
                    data["peptide_mask"] = helper.create_mask(
                        data['sequence_integer'])
                    pred = model(data)
                    if not pearson:
                        sas = helper.spectral_angle(
                            data['intensities_raw'], pred)
                    else:
                        sas = helper.pearson_coff(
                            data['intensities_raw'], pred)
                    scores.append(sas.detach().cpu().numpy())
                scores = np.concatenate(scores, axis=0)
                train_data.assign_train_score(scores)
                q_values = train_data.Q_values()
                q_values_num = np.sum(q_values < validate_q_threshold)
                if q_values_num > best_q_value_num:
                    best_model = deepcopy(model)
                    best_q_value_num = q_values_num
                    print("Achieve best: ", end='')
                print("Train:", np.sum(q_values < 0.001), np.sum(
                    q_values < 0.01), np.sum(q_values < 0.1))
                train_loader = DataLoader(train_data.semisupervised_sa_finetune(
                    threshold=q_threshold), batch_size=batch_size, shuffle=True)

                if enable_test:
                    with torch.no_grad():
                        scores = []
                        for i, data in enumerate(test_infer_loader):
                            data = {k: v.to(device) for k, v in data.items()}
                            data["peptide_mask"] = helper.create_mask(
                                data['sequence_integer'])
                            pred = model(data)
                            if not pearson:
                                sas = helper.spectral_angle(
                                    data['intensities_raw'], pred)
                            else:
                                sas = helper.pearson_coff(
                                    data['intensities_raw'], pred)
                            scores.append(sas.detach().cpu().numpy())
                        scores = np.concatenate(scores, axis=0)
                        train_data.assign_test_score(scores)
                        test_q_values = train_data.Q_values_test()
                        print(" Test:", np.sum(test_q_values < 0.001),
                              np.sum(test_q_values < 0.01), np.sum(test_q_values < 0.1))
            # train_loader = DataLoader(train_data.semisupervised_sa_finetune_noneg(
            # ), batch_size=batch_size, shuffle=True)
        for i, data in enumerate(train_loader):
            train_count = i + 1
            data = {k: v.to(device) for k, v in data.items()}
            data["peptide_mask"] = helper.create_mask(
                data['sequence_integer'])
            pred = model(data)
            loss_b, fine_loss, l1_loss = loss_fn(
                data['intensities_raw'], pred, data['label'])
            optimizer.zero_grad()
            loss_b.backward()
            optimizer.step()
            sys.stdout.flush()
            # print(
            #     f"\r    -Train Loss {loss/train_count:.3f}, {loss_l1/train_count:.3f}, {loss_sa/train_count:.3f}", end="")
            loss += loss_b.item()
            loss_l1 += l1_loss
            loss_sa += fine_loss
        # print()
    return best_model, train_data.id2remove()


def semisupervised_finetune_twofold(ori_model, input_table, batch_size=1024, gpu_index=0, max_epochs=10,
                                    update_interval=1, q_threshold=0.1, validate_q_threshold=0.01, pearson=False,
                                    enable_test=False, only_id2remove=False, onlypos=False):
    helper.set_seed(2022)
    if torch.cuda.is_available():
        device = torch.device(f"cuda:{gpu_index}")
    else:
        device = torch.device("cpu")
    print(
        f"Run on {device}, with training-q {q_threshold}, valid-q {validate_q_threshold}, epoch {max_epochs}")

    def finetune(dataset):
        model = deepcopy(ori_model)
        model = model.train()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, eps=1e-8)
        if not onlypos:
            loss_fn = helper.FinetuneSALoss(pearson=pearson)
        else:
            loss_fn = helper.FinetuneSALossNoneg(pearson=pearson)
        model = model.to(device)
        data_loader = DataLoader(
            dataset.train_all_data(), batch_size=1024, shuffle=False)
        ori_q_values = dataset.Q_values()
        print(f"Baseline Andromeda({len(ori_q_values)}): ", end='')
        print(np.sum(ori_q_values < 0.001), np.sum(
            ori_q_values < 0.01), np.sum(ori_q_values < 0.1))
        print("---------------")
        best_model = None
        best_q_value_num = 0
        for epoch in range(max_epochs * update_interval):
            loss = 0
            loss_l1 = 0.
            loss_sa = 0.
            if (epoch % update_interval) == 0:
                with torch.no_grad():
                    scores = []
                    for i, data in enumerate(data_loader):
                        data = {k: v.to(device) for k, v in data.items()}
                        data["peptide_mask"] = helper.create_mask(
                            data['sequence_integer'])
                        pred = model(data)
                        if not pearson:
                            sas = helper.spectral_angle(
                                data['intensities_raw'], pred)
                        else:
                            sas = helper.pearson_coff(
                                data['intensities_raw'], pred)
                        scores.append(sas.detach().cpu().numpy())
                    scores = np.concatenate(scores, axis=0)
                    dataset.assign_train_score(scores)
                    q_values = dataset.Q_values()
                    q_values_num = np.sum(q_values < validate_q_threshold)
                    if q_values_num > best_q_value_num:
                        del best_model
                        best_model = deepcopy(model)
                        best_q_value_num = q_values_num
                        print(
                            f"({epoch}){(np.sum(q_values < 0.001), np.sum(q_values < 0.01))}*", end=' ')
                    else:
                        print(f"({epoch})", (np.sum(q_values < 0.001),
                              np.sum(q_values < 0.01)), end=' ')
                    train_loader = DataLoader(dataset.semisupervised_sa_finetune(
                        threshold=q_threshold), batch_size=batch_size, shuffle=True)
                # train_loader = DataLoader(dataset.semisupervised_sa_finetune_noneg(
                # ), batch_size=batch_size, shuffle=True)
            for i, data in enumerate(train_loader):
                train_count = i + 1
                data = {k: v.to(device) for k, v in data.items()}
                data["peptide_mask"] = helper.create_mask(
                    data['sequence_integer'])
                pred = model(data)
                loss_b, fine_loss, l1_loss = loss_fn(
                    data['intensities_raw'], pred, data['label'])
                optimizer.zero_grad()
                loss_b.backward()
                optimizer.step()
                sys.stdout.flush()
                # print(
                #     f"\r    -Train Loss {loss/train_count:.3f}, {loss_l1/train_count:.3f}, {loss_sa/train_count:.3f}", end="")
                loss += loss_b.item()
                loss_l1 += l1_loss
                loss_sa += fine_loss
        if np.sum(ori_q_values < validate_q_threshold) > best_q_value_num:
            del best_model
            best_model = deepcopy(ori_model)
            print("Bad fine-tuning results, roll back to the original")
        return best_model

    dataset_manager = SemiDataset(input_table)
    id2remove = dataset_manager.id2remove()  # default first part
    if only_id2remove:
        return ori_model, ori_model, id2remove
    model1 = finetune(dataset_manager)

    dataset_manager = dataset_manager.reverse()
    model2 = finetune(dataset_manager)
    return model1, model2, id2remove


def semisupervised_finetune_twofold_test(ori_model, input_table, batch_size=1024, gpu_index=0, max_epochs=10,
                                         update_interval=1, q_threshold=0.1, validate_q_threshold=0.01, pearson=False,
                                         enable_test=False, only_id2remove=False, onlypos=False):
    helper.set_seed(2022)
    if torch.cuda.is_available():
        device = torch.device(f"cuda:{gpu_index}")
    else:
        device = torch.device("cpu")
    print(
        f"Run on {device}, with training-q {q_threshold}, valid-q {validate_q_threshold}, epoch {max_epochs}")

    def finetune(dataset: SemiDataset_twofold):
        model = deepcopy(ori_model)
        model = model.train()
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, eps=1e-8)
        if not onlypos:
            loss_fn = helper.FinetuneSALoss(pearson=pearson)
        else:
            loss_fn = helper.FinetuneSALossNoneg(pearson=pearson)
        model = model.to(device)
        data_loader = DataLoader(
            dataset.train_all_data(), batch_size=1024, shuffle=False)
        infer_loader = DataLoader(
            dataset.test_all_data(), batch_size=1024, shuffle=False)
        ori_q_values = dataset.Q_values()
        print(f"Baseline Andromeda({len(ori_q_values)}): ", end='')
        print(np.sum(ori_q_values < 0.001), np.sum(
            ori_q_values < 0.01), np.sum(ori_q_values < 0.1))
        print("---------------")
        best_model = None
        best_q_value_num = 0
        for epoch in range(max_epochs * update_interval):
            loss = 0
            loss_l1 = 0.
            loss_sa = 0.
            if (epoch % update_interval) == 0:
                with torch.no_grad():
                    scores = []
                    for i, data in enumerate(data_loader):
                        data = {k: v.to(device) for k, v in data.items()}
                        data["peptide_mask"] = helper.create_mask(
                            data['sequence_integer'])
                        pred = model(data)
                        if not pearson:
                            sas = helper.spectral_angle(
                                data['intensities_raw'], pred)
                        else:
                            sas = helper.pearson_coff(
                                data['intensities_raw'], pred)
                        scores.append(sas.detach().cpu().numpy())
                    scores = np.concatenate(scores, axis=0)
                    dataset.assign_train_score(scores)
                    q_values = dataset.Q_values()
                    q_values_num = np.sum(q_values < validate_q_threshold)
                    if q_values_num > best_q_value_num:
                        del best_model
                        best_model = deepcopy(model)
                        best_q_value_num = q_values_num
                        print(
                            f"({epoch}){(np.sum(q_values < 0.001), np.sum(q_values < 0.01))}*", end=' ')
                    else:
                        print(f"({epoch})", (np.sum(q_values < 0.001),
                              np.sum(q_values < 0.01)), end=' ')
                    train_loader = DataLoader(dataset.semisupervised_sa_finetune(
                        threshold=q_threshold), batch_size=batch_size, shuffle=True)
                    if enable_test:
                        scores = []
                        for i, data in enumerate(infer_loader):
                            data = {k: v.to(device) for k, v in data.items()}
                            data["peptide_mask"] = helper.create_mask(
                                data['sequence_integer'])
                            pred = model(data)
                            if not pearson:
                                sas = helper.spectral_angle(
                                    data['intensities_raw'], pred)
                            else:
                                sas = helper.pearson_coff(
                                    data['intensities_raw'], pred)
                            scores.append(sas.detach().cpu().numpy())
                        scores = np.concatenate(scores, axis=0)
                        dataset.assign_test_score(scores)
                        test_q_v = dataset.Q_values_test()
                        print(
                            f"test: ({epoch}){(np.sum(test_q_v < 0.001), np.sum(test_q_v < 0.01))}")

                # train_loader = DataLoader(dataset.semisupervised_sa_finetune_noneg(
                # ), batch_size=batch_size, shuffle=True)
            for i, data in enumerate(train_loader):
                train_count = i + 1
                data = {k: v.to(device) for k, v in data.items()}
                data["peptide_mask"] = helper.create_mask(
                    data['sequence_integer'])
                pred = model(data)
                loss_b, fine_loss, l1_loss = loss_fn(
                    data['intensities_raw'], pred, data['label'])
                optimizer.zero_grad()
                loss_b.backward()
                optimizer.step()
                sys.stdout.flush()
                
                loss += loss_b.item()
                loss_l1 += l1_loss
                loss_sa += fine_loss
        if np.sum(ori_q_values < validate_q_threshold) > best_q_value_num:
            del best_model
            best_model = deepcopy(ori_model)
            print("Bad fine-tuning results, roll back to the original")
        return best_model

    dataset_manager = SemiDataset_twofold(input_table)
    # dataset_manager = SemiDataset_nfold(input_table, nfold=2)
    # dataset_manager.set_index(1)
    id2remove = dataset_manager.id2remove()  # default first part
    if only_id2remove:
        return ori_model, ori_model, id2remove
    model1 = finetune(dataset_manager)

    dataset_manager = dataset_manager.reverse()
    model2 = finetune(dataset_manager)
    return model1, model2, id2remove


def semisupervised_finetune_nfold(ori_model, input_table, batch_size=1024, gpu_index=0, max_epochs=10, nfold=3,
                                  update_interval=1, q_threshold=0.1, validate_q_threshold=0.01, pearson=False,
                                  enable_test=False, only_id2select=False, onlypos=False):
    helper.set_seed(2022)
    if torch.cuda.is_available():
        device = torch.device(f"cuda:{gpu_index}")
    else:
        device = torch.device("cpu")
    print(
        f"Run on {device}, nfold = {nfold},with training-q {q_threshold}, valid-q {validate_q_threshold}, epoch {max_epochs}"
    )

    def ce_caliration(dataset):
        pass

    def finetune(dataset: SemiDataset_twofold):
        model = deepcopy(ori_model)
        model = model.train()
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, eps=1e-8)
        if not onlypos:
            loss_fn = helper.FinetuneSALoss(pearson=pearson)
        else:
            loss_fn = helper.FinetuneSALossNoneg(pearson=pearson)
        model = model.to(device)
        data_loader = DataLoader(
            dataset.train_all_data(), batch_size=1024, shuffle=False)
        infer_loader = DataLoader(
            dataset.test_all_data(), batch_size=1024, shuffle=False)
        ori_q_values = dataset.Q_values()
        print(f">>Baseline Andromeda({len(ori_q_values)}): ", end='')
        print(np.sum(ori_q_values < 0.001), np.sum(
            ori_q_values < 0.01), np.sum(ori_q_values < 0.1))
        best_model = None
        best_q_value_num = 0
        for epoch in range(max_epochs * update_interval):
            loss = 0
            loss_l1 = 0.
            loss_sa = 0.
            if (epoch % update_interval) == 0:
                with torch.no_grad():
                    scores = []
                    for i, data in enumerate(data_loader):
                        data = {k: v.to(device) for k, v in data.items()}
                        data["peptide_mask"] = helper.create_mask(
                            data['sequence_integer'])
                        pred = model(data)
                        if not pearson:
                            sas = helper.spectral_angle(
                                data['intensities_raw'], pred)
                        else:
                            sas = helper.pearson_coff(
                                data['intensities_raw'], pred)
                        scores.append(sas.detach().cpu().numpy())
                    scores = np.concatenate(scores, axis=0)
                    dataset.assign_train_score(scores)
                    q_values = dataset.Q_values()
                    q_values_num = np.sum(q_values < validate_q_threshold)
                    if q_values_num > best_q_value_num:
                        del best_model
                        best_model = deepcopy(model)
                        best_q_value_num = q_values_num
                        print(
                            f"({epoch}){(np.sum(q_values < 0.001), np.sum(q_values < 0.01))}*", end=' ')
                    else:
                        print(f"({epoch})", (np.sum(q_values < 0.001),
                              np.sum(q_values < 0.01)), end=' ')
                    train_loader = DataLoader(dataset.semisupervised_sa_finetune(
                        threshold=q_threshold), batch_size=batch_size, shuffle=True)
                    if enable_test:
                        scores = []
                        for i, data in enumerate(infer_loader):
                            data = {k: v.to(device) for k, v in data.items()}
                            data["peptide_mask"] = helper.create_mask(
                                data['sequence_integer'])
                            pred = model(data)
                            if not pearson:
                                sas = helper.spectral_angle(
                                    data['intensities_raw'], pred)
                            else:
                                sas = helper.pearson_coff(
                                    data['intensities_raw'], pred)
                            scores.append(sas.detach().cpu().numpy())
                        scores = np.concatenate(scores, axis=0)
                        dataset.assign_test_score(scores)
                        test_q_v = dataset.Q_values_test()
                        print(
                            f"test: ({epoch}){(np.sum(test_q_v < 0.001), np.sum(test_q_v < 0.01))}")

                # train_loader = DataLoader(dataset.semisupervised_sa_finetune_noneg(
                # ), batch_size=batch_size, shuffle=True)
            for i, data in enumerate(train_loader):
                train_count = i + 1
                data = {k: v.to(device) for k, v in data.items()}
                data["peptide_mask"] = helper.create_mask(
                    data['sequence_integer'])
                pred = model(data)
                loss_b, fine_loss, l1_loss = loss_fn(
                    data['intensities_raw'], pred, data['label'])
                optimizer.zero_grad()
                loss_b.backward()
                optimizer.step()
                sys.stdout.flush()
                
                loss += loss_b.item()
                loss_l1 += l1_loss
                loss_sa += fine_loss
        if np.sum(ori_q_values < validate_q_threshold) > best_q_value_num:
            del best_model
            best_model = deepcopy(ori_model)
            print("Bad fine-tuning results, roll back to the original")
        return best_model

    dataset_manager = SemiDataset_nfold(input_table, nfold=nfold)
    
    id2select = dataset_manager.id2predict()
    if only_id2select:
        return [ori_model for _ in range(nfold)], id2select
    
    models = []
    for i in range(nfold):
        dataset_manager.set_index(i)
        print(f">>Running fold-{i}, train set {len(dataset_manager._d)}, test set {len(dataset_manager._test_d)}")
        model = finetune(dataset_manager)
        models.append(model)
    return models, id2select


# adapted from https://github.com/pFindStudio/pDeep3
def pdeep3_few_shot_score(ori_model, input_table, batch_size=1024, gpu_index=0, max_epochs=100, nfold=3,
                          update_interval=1, q_threshold=0.1, validate_q_threshold=0.01, pearson=False,
                          enable_test=False, only_id2select=False, onlypos=False):
    helper.set_seed(2022)
    if torch.cuda.is_available():
        device = torch.device(f"cuda:{gpu_index}")
    else:
        device = torch.device("cpu")
    print(
        f"Run on {device}, nfold = {nfold}, max_sample=100"
    )

    def ce_caliration(dataset):
        pass

    def finetune(dataset: pDeep_nfold):
        model = deepcopy(ori_model)
        model = model.train()
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, eps=1e-8)
        if not onlypos:
            loss_fn = helper.FinetuneSALoss(pearson=pearson)
        else:
            loss_fn = helper.FinetuneSALossNoneg(pearson=pearson)
        model = model.to(device)
        
        ori_q_values = dataset.Q_values()
        print(f">>Baseline Andromeda({len(ori_q_values)}): ", end='')
        print(np.sum(ori_q_values < 0.001), np.sum(
            ori_q_values < 0.01), np.sum(ori_q_values < 0.1))
        best_model = None
        best_q_value_num = 0
        

        scores = dataset.pdeep_train_score()
        dataset.assign_train_score(scores)
        # q_values_num = np.sum(q_values < validate_q_threshold)
        train_loader = DataLoader(dataset.pdeep3_finetune(max_sample=100), 
                                  batch_size=batch_size, 
                                  shuffle=True)
        for epoch in range(max_epochs):
            loss = 0
            loss_l1 = 0.
            loss_sa = 0.
                # train_loader = DataLoader(dataset.semisupervised_sa_finetune_noneg(
                # ), batch_size=batch_size, shuffle=True)
            for i, data in enumerate(train_loader):
                train_count = i + 1
                data = {k: v.to(device) for k, v in data.items()}
                data["peptide_mask"] = helper.create_mask(
                    data['sequence_integer'])
                pred = model(data)
                loss_b, fine_loss, l1_loss = loss_fn(
                    data['intensities_raw'], pred, data['label'])
                optimizer.zero_grad()
                loss_b.backward()
                optimizer.step()
                sys.stdout.flush()
                
                loss += loss_b.item()
                loss_l1 += l1_loss
                loss_sa += fine_loss
                # print(f"{loss_b.item():.6f}\b", end=' ')
        # if np.sum(ori_q_values < validate_q_threshold) > best_q_value_num:
        #     del best_model
        #     best_model = deepcopy(ori_model)
        #     print("Bad fine-tuning results, roll back to the original")
        return model

    dataset_manager = pDeep_nfold(input_table, nfold=nfold)
    
    id2select = dataset_manager.id2predict()
    if only_id2select:
        return [ori_model for _ in range(nfold)], id2select
    
    models = []
    for i in range(nfold):
        dataset_manager.set_index(i)
        print(f">>Running fold-{i}, train set {len(dataset_manager._d)}, test set {len(dataset_manager._test_d)}")
        model = finetune(dataset_manager)
        models.append(model)
    return models, id2select

if __name__ == "__main__":
    from . import model
    which = 'lysc'
    data_file = f"/data/prosit/figs/fig235/{which}/percolator_up/try/prosit_l1/fixed_features.tab"

    # run_model = model.PrositFrag()
    # run_model.load_state_dict(torch.load("/home/gus/Desktop/ms_pred/checkpoints/frag/best_frag_l1_PrositFrag-1024.pth", map_location="cpu"))
    # semisupervised_pair_finetune(run_model, data_file)

    # run_model = model.PrositIRT()
    # run_model.load_state_dict(torch.load(
    #     f"./checkpoints/irt/best_valid_irt_{run_model.comment()}-1024.pth", map_location="cpu"))
    # semisupervised_rt_finetune(run_model, data_file)

    run_spect_model = model.PrositFrag()
    run_spect_model.load_state_dict(torch.load(
        "/home/gus/Desktop/ms_pred/checkpoints/frag/best_frag_l1_PrositFrag-1024.pth", map_location="cpu"))
    run_rt_model = model.PrositIRT()
    run_rt_model.load_state_dict(torch.load(
        f"./checkpoints/irt/best_valid_irt_{run_rt_model.comment()}-1024.pth", map_location="cpu"))
    run_model = model.Compose_single(run_spect_model, run_rt_model)
    # semisupervised_single_finetune(run_model, data_file)


