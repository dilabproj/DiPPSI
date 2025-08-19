
import os

lininterp_mimic_ppg_test = {'modelname':'lininterp', "annotate":"_mimic_ppg",  "annotate_test":"_test",
                'modeltype':'lininterp', 
                "modelparams":{},
                "data_name":"mimic_ppg","data_load": {"mean":True, "bounds":1, 
                                                      "train":False, "val":False, "test":True, "addmissing":True},
                "train":{"bs": 512, "gpus":[0]}}
mean_mimic_ppg_test = {'modelname':'mean', "annotate":"_mimic_ppg",  "annotate_test":"_test",
                'modeltype':'mean', 
                "modelparams":{},
                "data_name":"mimic_ppg","data_load": {"mean":True, "bounds":1, 
                                                      "train":False, "val":False, "test":True, "addmissing":True},
                "train":{"bs": 512, "gpus":[0]}}
fft_mimic_ppg_test = {'modelname':'fft', "annotate":"_mimic_ppg",  "annotate_test":"_test",
                'modeltype':'fft', 
                "modelparams":{},
                "data_name":"mimic_ppg","data_load": {"mean":True, "bounds":1, 
                                                      "train":False, "val":False, "test":True, "addmissing":True},
                "train":{"bs": 512, "gpus":[0]}}

lininterp_scientisst_ppg_test = {'modelname':'lininterp', "annotate":"_scientisst_ppg",  "annotate_test":"_test",
                'modeltype':'lininterp', 
                "modelparams":{},
                "data_name":"scientisst_ppg","data_load": {"mean":True, "bounds":1, 
                                                      "train":False, "val":False, "test":True, "addmissing":True},
                "train":{"bs": 512, "gpus":[0]}}
mean_scientisst_ppg_test = {'modelname':'mean', "annotate":"_scientisst_ppg",  "annotate_test":"_test",
                'modeltype':'mean', 
                "modelparams":{},
                "data_name":"scientisst_ppg","data_load": {"mean":True, "bounds":1, 
                                                      "train":False, "val":False, "test":True, "addmissing":True},
                "train":{"bs": 512, "gpus":[0]}}
fft_scientisst_ppg_test = {'modelname':'fft', "annotate":"_scientisst_ppg",  "annotate_test":"_test",
                'modeltype':'fft', 
                "modelparams":{},
                "data_name":"scientisst_ppg","data_load": {"mean":True, "bounds":1, 
                                                      "train":False, "val":False, "test":True, "addmissing":True},
                "train":{"bs": 512, "gpus":[0]}}

lininterp_wireless_ppg_test = {'modelname':'lininterp', "annotate":"_wireless_ppg",  "annotate_test":"_test",
                'modeltype':'lininterp', 
                "modelparams":{},
                "data_name":"wireless_ppg","data_load": {"mean":True, "bounds":1, 
                                                      "train":False, "val":False, "test":True, "addmissing":True},
                "train":{"bs": 512, "gpus":[0]}}
mean_wireless_ppg_test = {'modelname':'mean', "annotate":"_wireless_ppg",  "annotate_test":"_test",
                'modeltype':'mean', 
                "modelparams":{},
                "data_name":"wireless_ppg","data_load": {"mean":True, "bounds":1, 
                                                      "train":False, "val":False, "test":True, "addmissing":True},
                "train":{"bs": 512, "gpus":[0]}}
fft_wireless_ppg_test = {'modelname':'fft', "annotate":"_wireless_ppg",  "annotate_test":"_test",
                'modeltype':'fft', 
                "modelparams":{},
                "data_name":"wireless_ppg","data_load": {"mean":True, "bounds":1, 
                                                      "train":False, "val":False, "test":True, "addmissing":True},
                "train":{"bs": 512, "gpus":[0]}}

lininterp_scientisst_ppg_testtransient_30percent = {'modelname':'lininterp', "annotate":"_scientisst_ppg",  "annotate_test":"_test",
                'modeltype':'lininterp', 
                "modelparams":{},
                "data_name":"scientisst_ppg",
                "data_load": {"mode":True, "bounds":1,  "impute_transient":{"window":5, "prob":.20}, "channels":['ppg:wrist', 'acc_e4:z', 'acc_e4:x', 'acc_e4:y']},
                "train":{"bs": 128, "gpus":[0]}}
mean_scientisst_ppg_testtransient_30percent = {'modelname':'mean', "annotate":"_scientisst_ppg",  "annotate_test":"_test",
                'modeltype':'mean', 
                "modelparams":{},
                "data_name":"scientisst_ppg",
                "data_load": {"mode":True, "bounds":1,  "impute_transient":{"window":5, "prob":.20}, "channels":['ppg:wrist', 'acc_e4:z', 'acc:e4:x', 'acc_e4:y']},
                "train":{"bs": 128, "gpus":[0]}}
fft_scientisst_ppg_testtransient_30percent = {'modelname':'fft', "annotate":"_scientisst_ppg",  "annotate_test":"_test",
                'modeltype':'fft', 
                "modelparams":{},
                "data_name":"scientisst_ppg",
                "data_load": {"mode":True, "bounds":1,  "impute_transient":{"window":5, "prob":.20}, "channels":['ppg:wrist', 'acc_e4:z', 'acc:e4:x', 'acc_e4:y']},
                "train":{"bs": 128, "gpus":[0]}}

lininterp_wireless_ppg_testtransient_30percent = {'modelname':'lininterp', "annotate":"_wireless_ppg",  "annotate_test":"_test",
                'modeltype':'lininterp', 
                "modelparams":{},
                "data_name":"wireless_ppg",
                "data_load": {"mode":True, "bounds":1,  "impute_transient":{"window":5, "prob":.20}, "channels":['ppg:wrist', 'acc_e4:z', 'acc_e4:x', 'acc_e4:y']},
                "train":{"bs": 4, "gpus":[0]}}
mean_wireless_ppg_testtransient_30percent = {'modelname':'mean', "annotate":"_wireless_ppg",  "annotate_test":"_test",
                'modeltype':'mean', 
                "modelparams":{},
                "data_name":"wireless_ppg",
                "data_load": {"mode":True, "bounds":1,  "impute_transient":{"window":5, "prob":.20}, "channels":['ppg:wrist', 'acc_e4:z', 'acc:e4:x', 'acc_e4:y']},
                "train":{"bs": 128, "gpus":[0]}}
fft_wireless_ppg_testtransient_30percent = {'modelname':'fft', "annotate":"_wireless_ppg",  "annotate_test":"_test",
                'modeltype':'fft', 
                "modelparams":{},
                "data_name":"wireless_ppg",
                "data_load": {"mode":True, "bounds":1,  "impute_transient":{"window":5, "prob":.20}, "channels":['ppg:wrist', 'acc_e4:z', 'acc:e4:x', 'acc_e4:y']},
                "train":{"bs": 128, "gpus":[0]}}

lininterp_scientisst_ecg_testtransient_30percent = {'modelname':'lininterp', "annotate":"_scientisst_ecg",  "annotate_test":"_test",
                'modeltype':'lininterp', 
                "modelparams":{},
                "data_name":"scientisst_ecg",
                "data_load": {"mode":True, "bounds":1,  "impute_transient":{"window":50, "prob":.30}, "channels":['ecg:dry', 'acc_chest:z', 'acc_chest:x', 'acc_chest:y']},
                "train":{"bs": 128, "gpus":[0]}}
mean_scientisst_ecg_testtransient_30percent = {'modelname':'mean', "annotate":"_scientisst_ecg",  "annotate_test":"_test",
                'modeltype':'mean', 
                "modelparams":{},
                "data_name":"scientisst_ecg",
                "data_load": {"mode":True, "bounds":1,  "impute_transient":{"window":50, "prob":.30}, "channels":['ecg:dry', 'acc_chest:z', 'acc_chest:x', 'acc_chest:y']},
                "train":{"bs": 128, "gpus":[0]}}
fft_scientisst_ecg_testtransient_30percent = {'modelname':'fft', "annotate":"_scientisst_ecg",  "annotate_test":"_test",
                'modeltype':'fft', 
                "modelparams":{},
                "data_name":"scientisst_ecg",
                "data_load": {"mode":True, "bounds":1,  "impute_transient":{"window":50, "prob":.30}, "channels":['ecg:dry', 'acc_chest:z', 'acc_chest:x', 'acc_chest:y']},
                "train":{"bs": 128, "gpus":[0]}}

lininterp_mimic_ecg_test = {'modelname':'lininterp', "annotate":"_mimic_ecg",  "annotate_test":"_test",
                'modeltype':'lininterp', "data_name":"mimic_ecg",
                "modelparams":{},
                "data_load": {"train":False, "val":False, "test":True, "addmissing":True},
                "train":{"bs": 512, "gpus":[0]}}
mean_mimic_ecg_test = {'modelname':'mean', "annotate":"_mimic_ecg",  "annotate_test":"_test",
                'modeltype':'mean', "data_name":"mimic_ecg",
                "modelparams":{},
                "data_load": {"train":False, "val":False, "test":True, "addmissing":True},
                "train":{"bs": 512, "gpus":[0]}}
fft_mimic_ecg_test = {'modelname':'fft', "annotate":"_mimic_ecg",  "annotate_test":"_test",
                'modeltype':'fft', "data_name":"mimic_ecg",
                "modelparams":{},
                "data_load": {"train":False, "val":False, "test":True, "addmissing":True},
                "train":{"bs": 512, "gpus":[0]}}


lininterp_ptbxl_testextended_10percent = {'modelname':'lininterp', "annotate":"_ptbxl",  "annotate_test":"_testextended_10percent",
        'modeltype':'lininterp', "data_name":"ptbxl",
        "modelparams":{},
        "data_load": {"mode":True, "bounds":1, "impute_extended":100, "channels":[0]},
        "train":{"bs": 128, "gpus":[0]}}
lininterp_ptbxl_testextended_20percent = {'modelname':'lininterp', "annotate":"_ptbxl",  "annotate_test":"_testextended_20percent",
        'modeltype':'lininterp', "data_name":"ptbxl",
        "modelparams":{},
        "data_load": {"mode":True, "bounds":1, "impute_extended":200, "channels":[0]},
        "train":{"bs": 128, "gpus":[0]}}
lininterp_ptbxl_testextended_30percent = {'modelname':'lininterp', "annotate":"_ptbxl",  "annotate_test":"_testextended_30percent",
        'modeltype':'lininterp', "data_name":"ptbxl",
        "modelparams":{},
        "data_load": {"mode":True, "bounds":1, "impute_extended":300, "channels":[0]},
        "train":{"bs": 128, "gpus":[0]}}
lininterp_ptbxl_testextended_40percent = {'modelname':'lininterp', "annotate":"_ptbxl",  "annotate_test":"_testextended_40percent",
        'modeltype':'lininterp', "data_name":"ptbxl",
        "modelparams":{},
        "data_load": {"mode":True, "bounds":1, "impute_extended":400, "channels":[0]},
        "train":{"bs": 128, "gpus":[0]}}
lininterp_ptbxl_testextended_50percent = {'modelname':'lininterp', "annotate":"_ptbxl",  "annotate_test":"_testextended_50percent",
        'modeltype':'lininterp', "data_name":"ptbxl",
        "modelparams":{},
        "data_load": {"mode":True, "bounds":1, "impute_extended":500, "channels":[0]},
        "train":{"bs": 128, "gpus":[0]}}

lininterp_ptbxl_testtransient_10percent = {'modelname':'lininterp', "annotate":"_ptbxl",  "annotate_test":"_testtransient_10percent",
        'modeltype':'lininterp', "data_name":"ptbxl",
        "modelparams":{},
        "data_load": {"mode":True, "bounds":1,  "impute_transient":{"window":5, "prob":.10}, "channels":[0]},
        "train":{"bs": 128, "gpus":[0]}}
lininterp_ptbxl_testtransient_20percent = {'modelname':'lininterp', "annotate":"_ptbxl",  "annotate_test":"_testtransient_20percent",
        'modeltype':'lininterp', "data_name":"ptbxl",
        "modelparams":{},
        "data_load": {"mode":True, "bounds":1,  "impute_transient":{"window":5, "prob":.20}, "channels":[0]},
        "train":{"bs": 128, "gpus":[0]}}
lininterp_ptbxl_testtransient_30percent = {'modelname':'lininterp', "annotate":"_ptbxl",  "annotate_test":"_testtransient_30percent",
        'modeltype':'lininterp', "data_name":"ptbxl",
        "modelparams":{},
        "data_load": {"mode":True, "bounds":1,  "impute_transient":{"window":5, "prob":.30}, "channels":[0]},
        "train":{"bs": 128, "gpus":[0]}}
lininterp_ptbxl_testtransient_40percent = {'modelname':'lininterp', "annotate":"_ptbxl",  "annotate_test":"_testtransient_40percent",
        'modeltype':'lininterp', "data_name":"ptbxl",
        "modelparams":{},
        "data_load": {"mode":True, "bounds":1,  "impute_transient":{"window":5, "prob":.40}, "channels":[0]},
        "train":{"bs": 128, "gpus":[0]}}
lininterp_ptbxl_testtransient_50percent = {'modelname':'lininterp', "annotate":"_ptbxl",  "annotate_test":"_testtransient_50percent",
        'modeltype':'lininterp', "data_name":"ptbxl",
        "modelparams":{},
        "data_load": {"mode":True, "bounds":1,  "impute_transient":{"window":5, "prob":.50}, "channels":[0]},
        "train":{"bs": 128, "gpus":[0]}}




mean_ptbxl_testextended_10percent = {'modelname':'mean', "annotate":"_ptbxl",  "annotate_test":"_testextended_10percent",
        'modeltype':'mean', "data_name":"ptbxl",
        "modelparams":{},
        "data_load": {"mode":True, "bounds":1, "impute_extended":100, "channels":[0]},
        "train":{"bs": 128, "gpus":[0]}}
mean_ptbxl_testextended_20percent = {'modelname':'mean', "annotate":"_ptbxl",  "annotate_test":"_testextended_20percent",
        'modeltype':'mean', "data_name":"ptbxl",
        "modelparams":{},
        "data_load": {"mode":True, "bounds":1, "impute_extended":200, "channels":[0]},
        "train":{"bs": 128, "gpus":[0]}}
mean_ptbxl_testextended_30percent = {'modelname':'mean', "annotate":"_ptbxl",  "annotate_test":"_testextended_30percent",
        'modeltype':'mean', "data_name":"ptbxl",
        "modelparams":{},
        "data_load": {"mode":True, "bounds":1, "impute_extended":300, "channels":[0]},
        "train":{"bs": 128, "gpus":[0]}}
mean_ptbxl_testextended_40percent = {'modelname':'mean', "annotate":"_ptbxl",  "annotate_test":"_testextended_40percent",
        'modeltype':'mean', "data_name":"ptbxl",
        "modelparams":{},
        "data_load": {"mode":True, "bounds":1, "impute_extended":400, "channels":[0]},
        "train":{"bs": 128, "gpus":[0]}}
mean_ptbxl_testextended_50percent = {'modelname':'mean', "annotate":"_ptbxl",  "annotate_test":"_testextended_50percent",
        'modeltype':'mean', "data_name":"ptbxl",
        "modelparams":{},
        "data_load": {"mode":True, "bounds":1, "impute_extended":500, "channels":[0]},
        "train":{"bs": 128, "gpus":[0]}}

mean_ptbxl_testtransient_10percent = {'modelname':'mean', "annotate":"_ptbxl",  "annotate_test":"_testtransient_10percent",
        'modeltype':'mean', "data_name":"ptbxl",
        "modelparams":{},
        "data_load": {"mode":True, "bounds":1,  "impute_transient":{"window":5, "prob":.10}, "channels":[0]},
        "train":{"bs": 128, "gpus":[0]}}
mean_ptbxl_testtransient_20percent = {'modelname':'mean', "annotate":"_ptbxl",  "annotate_test":"_testtransient_20percent",
        'modeltype':'mean', "data_name":"ptbxl",
        "modelparams":{},
        "data_load": {"mode":True, "bounds":1,  "impute_transient":{"window":5, "prob":.20}, "channels":[0]},
        "train":{"bs": 128, "gpus":[0]}}
mean_ptbxl_testtransient_30percent = {'modelname':'mean', "annotate":"_ptbxl",  "annotate_test":"_testtransient_30percent",
        'modeltype':'mean', "data_name":"ptbxl",
        "modelparams":{},
        "data_load": {"mode":True, "bounds":1,  "impute_transient":{"window":5, "prob":.30}, "channels":[0]},
        "train":{"bs": 128, "gpus":[0]}}
mean_ptbxl_testtransient_40percent = {'modelname':'mean', "annotate":"_ptbxl",  "annotate_test":"_testtransient_40percent",
        'modeltype':'mean', "data_name":"ptbxl",
        "modelparams":{},
        "data_load": {"mode":True, "bounds":1,  "impute_transient":{"window":5, "prob":.40}, "channels":[0]},
        "train":{"bs": 128, "gpus":[0]}}
mean_ptbxl_testtransient_50percent = {'modelname':'mean', "annotate":"_ptbxl",  "annotate_test":"_testtransient_50percent",
        'modeltype':'mean', "data_name":"ptbxl",
        "modelparams":{},
        "data_load": {"mode":True, "bounds":1,  "impute_transient":{"window":5, "prob":.50}, "channels":[0]},
        "train":{"bs": 128, "gpus":[0]}}



fft_ptbxl_testextended_10percent = {'modelname':'fft', "annotate":"_ptbxl",  "annotate_test":"_testextended_10percent",
        'modeltype':'fft', "data_name":"ptbxl",
        "modelparams":{},
        "data_load": {"mode":True, "bounds":1, "impute_extended":100, "channels":[0]},
        "train":{"bs": 128, "gpus":[0]}}
fft_ptbxl_testextended_20percent = {'modelname':'fft', "annotate":"_ptbxl",  "annotate_test":"_testextended_20percent",
        'modeltype':'fft', "data_name":"ptbxl",
        "modelparams":{},
        "data_load": {"mode":True, "bounds":1, "impute_extended":200, "channels":[0]},
        "train":{"bs": 128, "gpus":[0]}}
fft_ptbxl_testextended_30percent = {'modelname':'fft', "annotate":"_ptbxl",  "annotate_test":"_testextended_30percent",
        'modeltype':'fft', "data_name":"ptbxl",
        "modelparams":{},
        "data_load": {"mode":True, "bounds":1, "impute_extended":300, "channels":[0]},
        "train":{"bs": 128, "gpus":[0]}}
fft_ptbxl_testextended_40percent = {'modelname':'fft', "annotate":"_ptbxl",  "annotate_test":"_testextended_40percent",
        'modeltype':'fft', "data_name":"ptbxl",
        "modelparams":{},
        "data_load": {"mode":True, "bounds":1, "impute_extended":400, "channels":[0]},
        "train":{"bs": 128, "gpus":[0]}}
fft_ptbxl_testextended_50percent = {'modelname':'fft', "annotate":"_ptbxl",  "annotate_test":"_testextended_50percent",
        'modeltype':'fft', "data_name":"ptbxl",
        "modelparams":{},
        "data_load": {"mode":True, "bounds":1, "impute_extended":500, "channels":[0]},
        "train":{"bs": 128, "gpus":[0]}}

fft_ptbxl_testtransient_10percent = {'modelname':'fft', "annotate":"_ptbxl",  "annotate_test":"_testtransient_10percent",
        'modeltype':'fft', "data_name":"ptbxl",
        "modelparams":{},
        "data_load": {"mode":True, "bounds":1,  "impute_transient":{"window":5, "prob":.10}, "channels":[0]},
        "train":{"bs": 128, "gpus":[0]}}
fft_ptbxl_testtransient_20percent = {'modelname':'fft', "annotate":"_ptbxl",  "annotate_test":"_testtransient_20percent",
        'modeltype':'fft', "data_name":"ptbxl",
        "modelparams":{},
        "data_load": {"mode":True, "bounds":1,  "impute_transient":{"window":5, "prob":.20}, "channels":[0]},
        "train":{"bs": 128, "gpus":[0]}}
fft_ptbxl_testtransient_30percent = {'modelname':'fft', "annotate":"_ptbxl",  "annotate_test":"_testtransient_30percent",
        'modeltype':'fft', "data_name":"ptbxl",
        "modelparams":{},
        "data_load": {"mode":True, "bounds":1,  "impute_transient":{"window":5, "prob":.30}, "channels":[0]},
        "train":{"bs": 128, "gpus":[0]}}
fft_ptbxl_testtransient_40percent = {'modelname':'fft', "annotate":"_ptbxl",  "annotate_test":"_testtransient_40percent",
        'modeltype':'fft', "data_name":"ptbxl",
        "modelparams":{},
        "data_load": {"mode":True, "bounds":1,  "impute_transient":{"window":5, "prob":.40}, "channels":[0]},
        "train":{"bs": 128, "gpus":[0]}}
fft_ptbxl_testtransient_50percent = {'modelname':'fft', "annotate":"_ptbxl",  "annotate_test":"_testtransient_50percent",
        'modeltype':'fft', "data_name":"ptbxl",
        "modelparams":{},
        "data_load": {"mode":True, "bounds":1,  "impute_transient":{"window":5, "prob":.50}, "channels":[0]},
        "train":{"bs": 128, "gpus":[0]}}
