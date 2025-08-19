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

csdi_mimic_ppg_test = {'modelname':'csdiv1', "annotate":"_mimic_ppg", 'modeltype':'csdi',
                            "annotate_test":"_test",
                            "data_name":"mimic_ppg","data_load": {"mean":True, "bounds":1, "train":False, "val":False, "test":True, "addmissing":True},
                            "modelparams": {},
                            "epochs": 1,
                            "lr": 1.0e-3,
                            "itr_per_epoch": 1.0e+8,
                            "train": {"bs": 2, "gpus": [0], "missingness_config": missingness_config(miss_realppg="False")},
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

csdi_mimic_ecg_test = {'modelname':'csdiv1', "annotate":"_mimic_ecg", 'modeltype':'csdi',
                            "annotate_test":"_test",
                            "data_name":"mimic_ecg","data_load": {"mean":True, "bounds":1, "train":False, "val":False, "test":True, "addmissing":True},
                            "modelparams": {},
                            "epochs": 1,
                            "lr": 1.0e-3,
                            "itr_per_epoch": 1.0e+8,
                            "train": {"bs": 2, "gpus": [0],},
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

csdi_scientisst_ecg_testtransient_30percent = {"modelname": "csdiv1",
                            "modeltype": 'csdi',
                            "annotate": "_scientisst_ecg",
                            "annotate_test": "_test",
                            "data_name": "scientisst_ecg",
                            "data_load": {"mode": True, "bounds": 1, "channels": ['ecg:dry'], "impute_transient":{"window":50, "prob":.30}},
                            "modelparams": {},
                            "epochs": 1,
                            "lr": 1.0e-3,
                            "itr_per_epoch": 1.0e+8,
                            "train": {"bs": 8, "gpus": [0],
                                      "missingness_config": missingness_config(miss_extended=300),
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

csdi_extended_ptbxl_testextended_10percent = {"modelname": "csdiv1",
                            "modeltype": 'csdi',
                            "annotate": "_extended_ptbxl",
                            "annotate_test": "_testextended_10percent",
                            "data_name": "ptbxl",
                            "data_load": {"mode": True, "bounds": 1, "channels": [0]},
                            "modelparams": {},
                            "epochs": 1,
                            "lr": 1.0e-3,
                            "itr_per_epoch": 1.0e+8,
                            "train": {"bs": 8, "gpus": [0],
                                      "missingness_config": missingness_config(miss_extended=300),
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

csdi_extended_ptbxl_testextended_20percent = {"modelname": "csdiv1",
                            "modeltype": 'csdi',
                            "annotate": "_extended_ptbxl",
                            "annotate_test": "_testextended_20percent",
                            "data_name": "ptbxl",
                            "data_load": {"mode": True, "bounds": 1, "channels": [0]},
                            "modelparams": {},
                            "epochs": 1,
                            "lr": 1.0e-3,
                            "itr_per_epoch": 1.0e+8,
                            "train": {"bs": 8, "gpus": [0],
                                      "missingness_config": missingness_config(miss_extended=300),
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

csdi_extended_ptbxl_testextended_30percent = {"modelname": "csdiv1",
                            "modeltype": 'csdi',
                            "annotate": "_extended_ptbxl",
                            "annotate_test": "_testextended_30percent",
                            "data_name": "ptbxl",
                            "data_load": {"mode": True, "bounds": 1, "channels": [0]},
                            "modelparams": {},
                            "epochs": 1,
                            "lr": 1.0e-3,
                            "itr_per_epoch": 1.0e+8,
                            "train": {"bs": 8, "gpus": [0],
                                      "missingness_config": missingness_config(miss_extended=300),
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

csdi_extended_ptbxl_testextended_40percent = {"modelname": "csdiv1",
                            "modeltype": 'csdi',
                            "annotate": "_extended_ptbxl",
                            "annotate_test": "_testextended_40percent",
                            "data_name": "ptbxl",
                            "data_load": {"mode": True, "bounds": 1, "channels": [0]},
                            "modelparams": {},
                            "epochs": 1,
                            "lr": 1.0e-3,
                            "itr_per_epoch": 1.0e+8,
                            "train": {"bs": 8, "gpus": [0],
                                      "missingness_config": missingness_config(miss_extended=300),
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

csdi_extended_ptbxl_testextended_50percent = {"modelname": "csdiv1",
                            "modeltype": 'csdi',
                            "annotate": "_extended_ptbxl",
                            "annotate_test": "_testextended_50percent",
                            "data_name": "ptbxl",
                            "data_load": {"mode": True, "bounds": 1, "channels": [0]},
                            "modelparams": {},
                            "epochs": 1,
                            "lr": 1.0e-3,
                            "itr_per_epoch": 1.0e+8,
                            "train": {"bs": 8, "gpus": [0],
                                      "missingness_config": missingness_config(miss_extended=300),
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

csdi_transient_ptbxl_testtransient_10percent = {"modelname": "csdiv1",
                            "modeltype": 'csdi',
                            "annotate": "_transient_ptbxl",
                            "annotate_test": "_testtransient_10percent",
                            "data_name": "ptbxl",
                            "data_load": {"mode": True, "bounds": 1, "channels": [0]},
                            "modelparams": {},
                            "epochs": 1,
                            "lr": 1.0e-3,
                            "itr_per_epoch": 1.0e+8,
                            "train": {"bs": 8, "gpus": [0],
                                      "missingness_config": missingness_config(miss_transient={"wind": 5, "prob": .1}),
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

csdi_transient_ptbxl_testtransient_20percent = {"modelname": "csdiv1",
                            "modeltype": 'csdi',
                            "annotate": "_transient_ptbxl",
                            "annotate_test": "_testtransient_20percent",
                            "data_name": "ptbxl",
                            "data_load": {"mode": True, "bounds": 1, "channels": [0]},
                            "modelparams": {},
                            "epochs": 1,
                            "lr": 1.0e-3,
                            "itr_per_epoch": 1.0e+8,
                            "train": {"bs": 8, "gpus": [0],
                                      "missingness_config": missingness_config(miss_transient={"wind": 5, "prob": .2}),
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

csdi_transient_ptbxl_testtransient_30percent = {"modelname": "csdiv1",
                            "modeltype": 'csdi',
                            "annotate": "_transient_ptbxl",
                            "annotate_test": "_testtransient_30percent",
                            "data_name": "ptbxl",
                            "data_load": {"mode": True, "bounds": 1, "channels": [0]},
                            "modelparams": {},
                            "epochs": 1,
                            "lr": 1.0e-3,
                            "itr_per_epoch": 1.0e+8,
                            "train": {"bs": 8, "gpus": [0],
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

csdi_transient_ptbxl_testtransient_40percent = {"modelname": "csdiv1",
                            "modeltype": 'csdi',
                            "annotate": "_transient_ptbxl",
                            "annotate_test": "_testtransient_40percent",
                            "data_name": "ptbxl",
                            "data_load": {"mode": True, "bounds": 1, "channels": [0]},
                            "modelparams": {},
                            "epochs": 1,
                            "lr": 1.0e-3,
                            "itr_per_epoch": 1.0e+8,
                            "train": {"bs": 8, "gpus": [0],
                                      "missingness_config": missingness_config(miss_transient={"wind": 5, "prob": .4}),
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

csdi_transient_ptbxl_testtransient_50percent = {"modelname": "csdiv1",
                            "modeltype": 'csdi',
                            "annotate": "_transient_ptbxl",
                            "annotate_test": "_testtransient_50percent",
                            "data_name": "ptbxl",
                            "data_load": {"mode": True, "bounds": 1, "channels": [0]},
                            "modelparams": {},
                            "epochs": 1,
                            "lr": 1.0e-3,
                            "itr_per_epoch": 1.0e+8,
                            "train": {"bs": 8, "gpus": [0],
                                      "missingness_config": missingness_config(miss_transient={"wind": 5, "prob": .5}),
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
