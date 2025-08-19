class missingness_config():
    def __init__(self,
                 miss_transient=None,
                 miss_extended=None,
                 miss_realppg=None,
                 miss_realecg=None):

        assert bool(miss_transient) ^ bool(miss_extended) ^ bool(
            miss_realppg) ^ bool(miss_realecg)

        if miss_transient:
            self.miss_type = "miss_transient"
            self.miss = miss_transient
        elif miss_extended:
            self.miss_type = "miss_extended"
            self.miss = miss_extended
        elif miss_realppg:
            self.miss_type = "miss_realppg"
        elif miss_realecg:
            self.miss_type = "miss_realecg"

csdi_mimic_ppg = {"modelname": "csdiv1",
                            "modeltype": 'csdi',
                            "annotate": "_mimic_ppg",
                            "data_name": "mimic_ppg",
                            "data_load": {"mean" : True, "bounds": 1},
                            "modelparams": {},
                            "epochs": 200,
                            "lr": 1.0e-3,
                            "itr_per_epoch": 1.0e+8,
                            "train": {"bs": 2, "gpus": [2,3], "missingness_config": missingness_config(miss_realppg="True")},
                            "diffusion": {
                                "layers": 1,
                                "channels": 8,
                                "nheads": 1,
                                "diffusion_embedding_dim": 16,
                                "beta_start": 0.0001,
                                "beta_end": 0.5,
                                "num_steps": 10,
                                "schedule": "quad",
                                "is_linear": False,
                            },
                            "model": {
                                "is_unconditional": 0,
                                "timeemb": 16,
                                "featureemb": 16,
                                "target_strategy": "random"
                            }
                           }

csdi_mimic_ecg = {"modelname": "csdiv1",
                            "modeltype": 'csdi',
                            "annotate": "_mimic_ecg",
                            "data_name": "mimic_ecg",
                            "data_load": {},
                            "modelparams": {},
                            "epochs": 200,
                            "lr": 1.0e-3,
                            "itr_per_epoch": 1.0e+8,
                            "train": {"bs": 2, "gpus": [3], "missingness_config": missingness_config(miss_realecg="True")},
                            "diffusion": {
                                "layers": 4, 
                                "channels": 64,
                                "nheads": 1, #8
                                "diffusion_embedding_dim": 128,
                                "beta_start": 0.0001,
                                "beta_end": 0.5,
                                "num_steps": 50,
                                "schedule": "quad",
                                "is_linear": False,
                            },
                            "model": {
                                "is_unconditional": 0,
                                "timeemb": 128,
                                "featureemb": 16,
                                "target_strategy": "random"
                            }
                           }

csdi_extended_ptbxl = {"modelname": "csdiv1",
                            "modeltype": 'csdi',
                            "annotate": "_extended_ptbxl",
                            "data_name": "ptbxl",
                            "data_load": {"mode": True, "bounds": 1, "channels": [0]},
                            "modelparams": {},
                            "epochs": 200,
                            "lr": 1.0e-3,
                            "itr_per_epoch": 1.0e+8,
                            "train": {"bs": 16, "gpus": [0],
                                     "missingness_config": missingness_config(miss_extended=300)
                                     },
                            "diffusion": {
                                "layers": 4,
                                "channels": 64,
                                "nheads": 8,
                                "diffusion_embedding_dim": 128,
                                "beta_start": 0.0001,
                                "beta_end": 0.5,
                                "num_steps": 50,
                                "schedule": "quad",
                                "is_linear": False,
                            },
                            "model": {
                                "is_unconditional": 0,
                                "timeemb": 128,
                                "featureemb": 16,
                                "target_strategy": "random"
                            }
                           }

csdi_transient_ptbxl = {"modelname": "csdiv1",
                            "modeltype": 'csdi',
                            "annotate": "_transient_ptbxl",
                            "data_name": "ptbxl",
                            "data_load": {"mode": True, "bounds": 1, "channels": [0]},
                            "modelparams": {},
                            "epochs": 200,
                            "lr": 1.0e-3,
                            "itr_per_epoch": 1.0e+8,
                            "train": {"bs": 16, "gpus": [0],
                                      "missingness_config": missingness_config(miss_transient={"wind": 5, "prob": .3}),
                                      },
                            "diffusion": {
                                "layers": 4,
                                "channels": 64,
                                "nheads": 8,
                                "diffusion_embedding_dim": 128,
                                "beta_start": 0.0001,
                                "beta_end": 0.5,
                                "num_steps": 50,
                                "schedule": "quad",
                                "is_linear": False,
                            },
                            "model": {
                                "is_unconditional": 0,
                                "timeemb": 128,
                                "featureemb": 16,
                                "target_strategy": "random"
                            }
                        }

csdi_scientisst_ecg = {"modelname": "csdiv1",
                            "modeltype": 'csdi',
                            "annotate": "_scientisst_ecg",
                            "data_name": "scientisst_ecg",
                            "data_load": {"mode": True, "bounds": 1, "channels": ['ecg:dry']},
                            "modelparams": {},
                            "epochs": 200,
                            "lr": 1.0e-3,
                            "itr_per_epoch": 1.0e+8,
                            "train": {"bs": 4, "gpus": [3],
                                      "missingness_config": missingness_config(miss_transient={"wind": 50, "prob": .3}),
                                      },
                            "diffusion": {
                                "layers": 4,
                                "channels": 64,
                                "nheads": 8,
                                "diffusion_embedding_dim": 128,
                                "beta_start": 0.0001,
                                "beta_end": 0.5,
                                "num_steps": 50,
                                "schedule": "quad",
                                "is_linear": False,
                            },
                            "model": {
                                "is_unconditional": 0,
                                "timeemb": 128,
                                "featureemb": 16,
                                "target_strategy": "random"
                            }
                        }
