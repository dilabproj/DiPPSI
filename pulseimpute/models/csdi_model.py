import os
import multiprocessing
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from torch.optim import Adam
import numpy as np
from datetime import datetime
from tqdm import tqdm
import csv
from ast import literal_eval
import pickle


class csdi():
    def __init__(self, config, modelname,
                 data_name="",
                 train_data=None, val_data=None,
                 # annotations are used for creating folders to save models in
                 annotate="", annotate_test="",
                 bs=2, gpus=[0, 1],
                 # this is passed during testing, to serve as ground truth
                 imputation_dict=None,
                 # this is passed during training, for configuring how missingness is simulated while training the imputation model
                 missingness_config=None,
                 # when reloading the model, which iteration to reload
                 reload_iter="latest",
                 # how many iterations until running validation and saving model
                 iter_save=1000
                 ):
        '''
        Constructs necessary attributes for tutorial class

                Parameters:
                        data_name (str): A decimal integer
                        train_data (int): Another decimal integer

                Returns:
                        binary_sum (str): Binary string of the sum of a and b
        '''


        outpath = "out/"
        # data loader setupi
        self.bs = bs
        self.gpu_list = gpus
        self.iter_save = iter_save

        self.reload_iter = reload_iter

        self.data_loader_setup(train_data, val_data, imputation_dict=imputation_dict, missingness_config=missingness_config)

        if len(self.gpu_list) == 1:
            torch.cuda.set_device(self.gpu_list[0])

        # import model class
        model_module = __import__(f'models.csdi.{modelname}', fromlist=[""])
        model_module_class = getattr(model_module, "CSDI")
        self.model = nn.DataParallel(model_module_class(config=config, device=torch.device(f"cuda:{self.gpu_list[0]}")), device_ids=self.gpu_list)
        self.model.to(torch.device(f"cuda:{self.gpu_list[0]}"))
        
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=.001)

        self.ckpt_path = os.path.join(outpath, data_name+annotate_test, modelname+annotate)
        self.reload_ckpt_path = os.path.join(outpath, data_name, modelname+annotate)
        self.reload_model()

    def data_loader_setup(self, train_data, val_data=None, imputation_dict=None, missingness_config=None):
        num_threads_used = multiprocessing.cpu_count()

        if val_data is not None:
            train_dataset = dataset(waveforms=train_data, missingness_config=missingness_config, type="train")
            self.train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=self.bs, shuffle=True, num_workers=num_threads_used)

            val_dataset = dataset(waveforms=val_data, missingness_config=missingness_config, type="val")
            self.val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=self.bs, shuffle=False, num_workers=num_threads_used)
            temp = next(iter(self.val_loader))
        else:
            test_dataset = dataset(waveforms=train_data, imputation_dict=imputation_dict, missingness_config=missingness_config, type="test")
            self.test_loader = torch.utils.data.DataLoader( test_dataset, batch_size=self.bs, shuffle=False, num_workers=num_threads_used)
            temp = next(iter(self.test_loader))

        self.total_channels = temp[0].shape[2]

    def reload_model(self):
        self.iter_list = [-1]
        os.makedirs(self.ckpt_path, exist_ok=True)
        os.makedirs(os.path.join(self.reload_ckpt_path, "iter_latest"), exist_ok=True)
        os.makedirs(os.path.join(self.reload_ckpt_path, "iter_best"), exist_ok=True)
        self.best_val_loss = 9999999
        if os.path.isfile(os.path.join(self.reload_ckpt_path, "iter_best", "iter_best.pkl")):
            state = torch.load(os.path.join(self.reload_ckpt_path, "iter_best", "iter_best.pkl"), map_location=f"cuda:{self.gpu_list[0]}")
            best_iter = state["iter"]
            printlog(f"Identified best iter: {best_iter}")
            self.best_val_loss = state["l2valloss"].cpu()

        if os.path.isfile(os.path.join(self.reload_ckpt_path, f"iter_{self.reload_iter}", f"iter_{self.reload_iter}.pkl")):
            state = torch.load(os.path.join(self.reload_ckpt_path, f"iter_{self.reload_iter}", f"iter_{self.reload_iter}.pkl"), map_location=f"cuda:{self.gpu_list[0]}")
            self.iter_list.append(state["iter"])

            printlog(f"Reloading given iter: {np.max(self.iter_list)}", self.reload_ckpt_path)
            print(self.model.load_state_dict(state['state_dict'], strict=True))
            print(self.optimizer.load_state_dict(state['optimizer']))
        else:
            printlog(f"cannot reload iter {self.reload_iter}",
                      self.reload_ckpt_path)
            #printlog(f"Reloading ./out/model.pth")
            #print(self.model.load_state_dict(torch.load("./out/" + "/model.pth"), strict=True))

    def calc_denominator(target, eval_points):
        return torch.sum(torch.abs(target * eval_points))

    def calc_quantile_CRPS(self, target, forecast, eval_points, mean_scaler, scaler):

        target = target * scaler + mean_scaler
        forecast = forecast * scaler + mean_scaler

        quantiles = np.arange(0.05, 1.0, 0.05)
        denom = calc_denominator(target, eval_points)
        CRPS = 0
        for i in range(len(quantiles)):
            q_pred = []
            for j in range(len(forecast)):
                q_pred.append(torch.quantile(forecast[j : j + 1], quantiles[i], dim=1))
            q_pred = torch.cat(q_pred, 0)
            q_loss = quantile_loss(target, q_pred, quantiles[i], eval_points)
            CRPS += q_loss / denom
        return CRPS.item() / len(quantiles)

    def calc_quantile_CRPS_sum(self, target, forecast, eval_points, mean_scaler, scaler):

        eval_points = eval_points.mean(-1)
        target = target * scaler + mean_scaler
        target = target.sum(-1)
        forecast = forecast * scaler + mean_scaler

        quantiles = np.arange(0.05, 1.0, 0.05)
        denom = calc_denominator(target, eval_points)
        CRPS = 0
        for i in range(len(quantiles)):
            q_pred = torch.quantile(forecast.sum(-1),quantiles[i],dim=1)
            q_loss = quantile_loss(target, q_pred, quantiles[i], eval_points)
            CRPS += q_loss / denom
        return CRPS.item() / len(quantiles)

    def testimp(self, nsample=100, scaler=1, mean_scaler=0, foldername="out"):
        """
        Function to compute and save the imputation error on the test dataset 
        """

        print(f'{datetime.now().strftime("%d/%m/%Y %H:%M")} | Start')

        with torch.no_grad():
            self.model.eval()
            mse_total = 0
            mae_total = 0
            evalpoints_total = 0

            all_target = []
            all_observed_point = []
            all_observed_time = []
            all_evalpoint = []
            all_generated_samples = []
            all_generated_samples_median = []
            with tqdm(self.test_loader, mininterval=5.0, maxinterval=50.0) as it:
                for batch_no, test_batch in enumerate(it, start=1):
                    output = self.model.module.to(torch.device(f"cuda:{self.gpu_list[0]}")).evaluate(test_batch, nsample)

                    samples, c_target, eval_points, observed_points, observed_time = output
                    samples = samples.permute(0, 1, 3, 2)  # (B,nsample,L,K)
                    c_target = c_target.permute(0, 2, 1)  # (B,L,K)
                    eval_points = eval_points.permute(0, 2, 1)
                    observed_points = observed_points.permute(0, 2, 1)

                    samples_median = samples.median(dim=1)
                    all_target.append(c_target)
                    all_evalpoint.append(eval_points)
                    all_observed_point.append(observed_points)
                    all_observed_time.append(observed_time)
                    all_generated_samples.append(samples)
                    all_generated_samples_median.append(samples_median)

                    mse_current = (
                        ((samples_median.values - c_target) * eval_points) ** 2
                    ) * (scaler ** 2)
                    mae_current = (
                        torch.abs((samples_median.values - c_target) * eval_points) 
                    ) * scaler

                    mse_total += mse_current.sum().item()
                    mae_total += mae_current.sum().item()
                    evalpoints_total += eval_points.sum().item()

                    it.set_postfix(
                        ordered_dict={
                            "rmse_total": np.sqrt(mse_total / evalpoints_total),
                            "mae_total": mae_total / evalpoints_total,
                            "batch_no": batch_no,
                        },
                        refresh=True,
                    )

                    break

                with open(
                    foldername + "/generated_outputs_nsample" + str(nsample) + ".pk", "wb"
                ) as f:
                    all_target = torch.cat(all_target, dim=0)
                    all_evalpoint = torch.cat(all_evalpoint, dim=0)
                    all_observed_point = torch.cat(all_observed_point, dim=0)
                    all_observed_time = torch.cat(all_observed_time, dim=0)
                    all_generated_samples = torch.cat(all_generated_samples, dim=0)
                    # 提取每个median对象的values属性，并收集到一个新列表中
                    tensor_list = [item.values for item in all_generated_samples_median]
                    all_generated_samples_median = torch.cat(tensor_list, dim=0)
                    print("all_target.shape: ", all_target.shape)
                    print("all_evalpoint.shape: ", all_evalpoint.shape)
                    print("all_observed_point.shape: ", all_observed_point.shape)
                    print("all_observed_time.shape: ", all_observed_time.shape)
                    print("all_generated_samples.shape: ", all_generated_samples.shape)
                    print("all_generated_samples_median.shape: ", all_generated_samples_median.shape)

                    pickle.dump(
                        [
                            all_generated_samples_median,
                            all_generated_samples,
                            all_target,
                            all_evalpoint,
                            all_observed_point,
                            all_observed_time,
                            scaler,
                            mean_scaler,
                        ],
                        f,
                    )

                CRPS = self.calc_quantile_CRPS(
                    all_target, all_generated_samples, all_evalpoint, mean_scaler, scaler
                )
                CRPS_sum = self.calc_quantile_CRPS_sum(
                    all_target, all_generated_samples, all_evalpoint, mean_scaler, scaler
                )

                with open(
                    foldername + "/result_nsample" + str(nsample) + ".pk", "wb"
                ) as f:
                    pickle.dump(
                        [
                            np.sqrt(mse_total / evalpoints_total),
                            mae_total / evalpoints_total,
                            CRPS,
                        ],
                        f,
                    )
                    print("RMSE:", np.sqrt(mse_total / evalpoints_total))
                    print("MAE:", mae_total / evalpoints_total)
                    print("CRPS:", CRPS)
                    print("CRPS_sum:", CRPS_sum)

        # printlog(
        #     f'{datetime.now().strftime("%d/%m/%Y %H:%M")} | MSE:{total_test_l2_loss:.10f} \n',
        #     iter_check_path)

        # imputation_all = np.concatenate(imputation_list, axis=0)
        # np.save(os.path.join(iter_check_path, "imputation.npy"), imputation_all)

    def train(
        self,
        config,
        train_loader=None,
        valid_loader=None,
        valid_epoch_interval=20,
        foldername="out",
    ):
        print(torch.cuda.memory_summary())
        optimizer = Adam(self.model.parameters(), lr=config["lr"], weight_decay=1e-6)
        if foldername != "":
            output_path = foldername + "/model.pth"

        p1 = int(0.75 * config["epochs"])
        p2 = int(0.9 * config["epochs"])
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=[p1, p2], gamma=0.1
        )

        best_valid_loss = 1e10
        for epoch_no in range(config["epochs"]):
            avg_loss = 0
            self.model.train()
            with tqdm(self.train_loader, mininterval=5.0, maxinterval=50.0) as it:
                for batch_no, train_batch in enumerate(it, start=1):
                    optimizer.zero_grad()

                    loss = self.model(train_batch)
                    loss.backward()
                    avg_loss += loss.item()
                    optimizer.step()
                    it.set_postfix(
                        ordered_dict={
                            "avg_epoch_loss": avg_loss / batch_no,
                            "epoch": epoch_no,
                        },
                        refresh=False,
                    )
                    if batch_no >= config["itr_per_epoch"]:
                        break

                lr_scheduler.step()
            if (epoch_no + 1) % valid_epoch_interval == 0:
                self.model.eval()
                avg_loss_valid = 0
                with torch.no_grad():
                    with tqdm(self.val_loader, mininterval=5.0, maxinterval=50.0) as it:
                        for batch_no, valid_batch in enumerate(it, start=1):
                            loss = self.model(valid_batch, is_train=0)
                            avg_loss_valid += loss.item()
                            it.set_postfix(
                                ordered_dict={
                                    "valid_avg_epoch_loss": avg_loss_valid / batch_no,
                                    "epoch": epoch_no,
                                },
                                refresh=False,
                            )
                print(torch.cuda.memory_summary())
                if best_valid_loss > avg_loss_valid:
                    best_valid_loss = avg_loss_valid
                    print(
                        "\n best loss is updated to ",
                        avg_loss_valid / batch_no,
                        "at",
                        epoch_no,
                    )

        if foldername != "":
            torch.save(self.model.state_dict(), output_path)


class dataset(torch.utils.data.Dataset):
    def __init__(self, waveforms, imputation_dict=None,
                 missingness_config=None,
                 type=None):

        self.waveforms = waveforms
        self.imputation_dict = imputation_dict
        self.missingness_config = missingness_config

        if "real" in missingness_config.miss_type:
            if missingness_config.miss_type == "miss_realppg":
                #tuples_path = os.path.join("data", "missingness_patterns", f"missing_ppg_{type}.csv")
                tuples_path = os.path.join("/mnt/1stHDD/nfs/roeywu/PPG_data", "missingness_patterns/mHealth_missing_ppg", f"missing_ppg_{type}.csv")
            elif missingness_config.miss_type == "miss_realecg":
                #tuples_path = os.path.join("data", "missingness_patterns", f"missing_ecg_{type}.csv")
                tuples_path = os.path.join("/mnt/1stHDD/nfs/roeywu/PPG_data", "missingness_patterns/mHealth_missing_ecg", f"missing_ecg_{type}.csv")

            with open(tuples_path, 'r') as read_obj:
                csv_reader = csv.reader(read_obj)
                self.list_of_miss = list(csv_reader)

    def __len__(self):
        return len(self.waveforms)

    def __getitem__(self, idx):
        X = torch.clone(self.waveforms[idx, :, :])
        X_original = torch.clone(X)
        y = np.empty(X.shape, dtype=np.float32)
        y[:] = np.nan
        y = torch.from_numpy(y)

        # using real mHealth missingness patterns
        if "real" in self.missingness_config.miss_type:
            miss_idx = np.random.randint(len(self.list_of_miss))
            miss_vector = miss_tuple_to_vector(self.list_of_miss[miss_idx])
            y[np.where(miss_vector == 0)] = X[np.where(miss_vector == 0)]
            X[np.where(miss_vector == 0)] = 0
        # using simulated extended missingness
        elif self.missingness_config.miss_type == "miss_extended":
            amt = self.missingness_config.miss
            start_impute = np.random.randint(
                0, X.shape[0]-amt)
            y[start_impute:start_impute+amt,
                :] = X[start_impute:start_impute+amt, :]
            X[start_impute:start_impute+amt, :] = 0
        # using simulated transient missingness
        elif self.missingness_config.miss_type == "miss_transient":
            window = self.missingness_config.miss["wind"]
            probability = self.missingness_config.miss["prob"]
            for j in range(0, X.shape[0], window):
                rand = np.random.random_sample()
                if rand <= probability:
                    if X.shape[0]-j < window:
                        incr = X.shape[0]-j
                    else:
                        incr = window
                    y[j:j+incr, :] = X[j:j+incr, :]
                    X[j:j+incr, :] = 0

        y_dict = {"target_seq": y,
                  "original": X_original,
                  "name": idx}

        if self.imputation_dict:
            y_dict["target_seq"] = self.imputation_dict["target_seq"][idx]

        return X, y_dict


def miss_tuple_to_vector(listoftuples):
    def onesorzeros_vector(miss_tuple):
        miss_tuple = literal_eval(miss_tuple)
        if miss_tuple[0] == 0:
            return np.zeros(miss_tuple[1])
        elif miss_tuple[0] == 1:
            return np.ones(miss_tuple[1])

    miss_vector = onesorzeros_vector(listoftuples[0])
    for i in range(1, len(listoftuples)):
        miss_vector = np.concatenate(
            (miss_vector, onesorzeros_vector(listoftuples[i])))
    miss_vector = np.expand_dims(miss_vector, 1)
    return miss_vector


def l2_loss(logits, target):
    logits_temp = torch.clone(logits)
    target_temp = torch.clone(target)

    print("logits_temp.shape=",logits_temp.shape)
    print("target.shape=",target.shape)
    logits_temp[torch.isnan(target)] = 0
    target_temp[torch.isnan(target)] = 0
    difference = torch.square(logits_temp - target_temp)

    l2_loss = torch.sum(difference)
    missing_total = torch.sum(~torch.isnan(target))

    return l2_loss, missing_total


def printlog(line, path="", type="a"):
    print(line)
    with open(os.path.join(path, 'log.txt'), type) as file:
        file.write(line+'\n')
