# import argparse

# parser = argparse.ArgumentParser(description='manual to this script')
# parser.add_argument('--seed', type = int, default = 123456, help = 'random seed')
# args = parser.parse_args()
# print(args.seed)
import json
import numpy as np
with open("/root/lmy/GET/logs/get/log_results_Snopes_base/Fold_0/result_123756.txt_error_analysis_testing.json", "r") as f1:
    base0_test = json.load(f1)
    # print(type(base0_test))
    #claim_label  predicted_prob qid
base0_f_ids = []
base0_f_fakeids = []
base0_f_notfakeids = []
base0_t_ids = []
base0_t_fakeids = []
base0_t_notfakeids = []
for item in base0_test:
    claim_label = int(item["claim_label"])
    predicted_prob = np.array(item["predicted_prob"])
    pred_label = np.argmax(predicted_prob)
    qid = item["qid"]
    if claim_label == pred_label:
        base0_t_ids.append(qid)
        if claim_label == 0:
            base0_t_fakeids.append(qid)
        else:
            base0_t_notfakeids.append(qid)
    else:
        base0_f_ids.append(qid)
        if claim_label == 0:
            base0_f_fakeids.append(qid)
        else:
            base0_f_notfakeids.append(qid)
with open("/root/lmy/GET/logs/get/log_results_Snopes_exp/Fold_0/result_123756.txt_error_analysis_testing.json", "r") as f2:
    exp0_test = json.load(f2)
exp0_f_ids = []
exp0_f_fakeids = []
exp0_f_notfakeids = []
exp0_t_ids = []
exp0_t_fakeids = []
exp0_t_notfakeids = []
for item in exp0_test:
    claim_label = int(item["claim_label"])
    predicted_prob = np.array(item["predicted_prob"])
    pred_label = np.argmax(predicted_prob)
    qid = item["qid"]
    if claim_label == pred_label:
        exp0_t_ids.append(qid)
        if claim_label == 0:
            exp0_t_fakeids.append(qid)
        else:
            exp0_t_notfakeids.append(qid)
    else:
        exp0_f_ids.append(qid)
        if claim_label == 0:
            exp0_f_fakeids.append(qid)
        else:
            exp0_f_notfakeids.append(qid)

with open("/root/lmy/GET/logs/get/log_results_Snopes_base/Fold_1/result_123756.txt_error_analysis_testing.json", "r") as f1:
    base1_test = json.load(f1)
    # print(type(base0_test))
    #claim_label  predicted_prob qid
base1_f_ids = []
base1_f_fakeids = []
base1_f_notfakeids = []
base1_t_ids = []
base1_t_fakeids = []
base1_t_notfakeids = []
for item in base1_test:
    claim_label = int(item["claim_label"])
    predicted_prob = np.array(item["predicted_prob"])
    pred_label = np.argmax(predicted_prob)
    qid = item["qid"]
    if claim_label == pred_label:
        base1_t_ids.append(qid)
        if claim_label == 0:
            base1_t_fakeids.append(qid)
        else:
            base1_t_notfakeids.append(qid)
    else:
        base1_f_ids.append(qid)
        if claim_label == 0:
            base1_f_fakeids.append(qid)
        else:
            base1_f_notfakeids.append(qid)
with open("/root/lmy/GET/logs/get/log_results_Snopes_exp/Fold_1/result_123756.txt_error_analysis_testing.json", "r") as f2:
    exp1_test = json.load(f2)
exp1_f_ids = []
exp1_f_fakeids = []
exp1_f_notfakeids = []
exp1_t_ids = []
exp1_t_fakeids = []
exp1_t_notfakeids = []
for item in exp1_test:
    claim_label = int(item["claim_label"])
    predicted_prob = np.array(item["predicted_prob"])
    pred_label = np.argmax(predicted_prob)
    qid = item["qid"]
    if claim_label == pred_label:
        exp1_t_ids.append(qid)
        if claim_label == 0:
            exp1_t_fakeids.append(qid)
        else:
            exp1_t_notfakeids.append(qid)
    else:
        exp1_f_ids.append(qid)
        if claim_label == 0:
            exp1_f_fakeids.append(qid)
        else:
            exp1_f_notfakeids.append(qid)

with open("/root/lmy/GET/logs/get/log_results_Snopes_base/Fold_2/result_123756.txt_error_analysis_testing.json", "r") as f1:
    base2_test = json.load(f1)
    # print(type(base0_test))
    #claim_label  predicted_prob qid
base2_f_ids = []
base2_f_fakeids = []
base2_f_notfakeids = []
base2_t_ids = []
base2_t_fakeids = []
base2_t_notfakeids = []
for item in base2_test:
    claim_label = int(item["claim_label"])
    predicted_prob = np.array(item["predicted_prob"])
    pred_label = np.argmax(predicted_prob)
    qid = item["qid"]
    if claim_label == pred_label:
        base2_t_ids.append(qid)
        if claim_label == 0:
            base2_t_fakeids.append(qid)
        else:
            base2_t_notfakeids.append(qid)
    else:
        base2_f_ids.append(qid)
        if claim_label == 0:
            base2_f_fakeids.append(qid)
        else:
            base2_f_notfakeids.append(qid)
with open("/root/lmy/GET/logs/get/log_results_Snopes_exp/Fold_2/result_123756.txt_error_analysis_testing.json", "r") as f2:
    exp2_test = json.load(f2)
exp2_f_ids = []
exp2_f_fakeids = []
exp2_f_notfakeids = []
exp2_t_ids = []
exp2_t_fakeids = []
exp2_t_notfakeids = []
for item in exp2_test:
    claim_label = int(item["claim_label"])
    predicted_prob = np.array(item["predicted_prob"])
    pred_label = np.argmax(predicted_prob)
    qid = item["qid"]
    if claim_label == pred_label:
        exp2_t_ids.append(qid)
        if claim_label == 0:
            exp2_t_fakeids.append(qid)
        else:
            exp2_t_notfakeids.append(qid)
    else:
        exp2_f_ids.append(qid)
        if claim_label == 0:
            exp2_f_fakeids.append(qid)
        else:
            exp2_f_notfakeids.append(qid)

with open("/root/lmy/GET/logs/get/log_results_Snopes_base/Fold_3/result_123756.txt_error_analysis_testing.json", "r") as f1:
    base3_test = json.load(f1)
    # print(type(base0_test))
    #claim_label  predicted_prob qid
base3_f_ids = []
base3_f_fakeids = []
base3_f_notfakeids = []
base3_t_ids = []
base3_t_fakeids = []
base3_t_notfakeids = []
for item in base3_test:
    claim_label = int(item["claim_label"])
    predicted_prob = np.array(item["predicted_prob"])
    pred_label = np.argmax(predicted_prob)
    qid = item["qid"]
    if claim_label == pred_label:
        base3_t_ids.append(qid)
        if claim_label == 0:
            base3_t_fakeids.append(qid)
        else:
            base3_t_notfakeids.append(qid)
    else:
        base3_f_ids.append(qid)
        if claim_label == 0:
            base3_f_fakeids.append(qid)
        else:
            base3_f_notfakeids.append(qid)
with open("/root/lmy/GET/logs/get/log_results_Snopes_exp/Fold_3/result_123756.txt_error_analysis_testing.json", "r") as f2:
    exp3_test = json.load(f2)
exp3_f_ids = []
exp3_f_fakeids = []
exp3_f_notfakeids = []
exp3_t_ids = []
exp3_t_fakeids = []
exp3_t_notfakeids = []
for item in exp3_test:
    claim_label = int(item["claim_label"])
    predicted_prob = np.array(item["predicted_prob"])
    pred_label = np.argmax(predicted_prob)
    qid = item["qid"]
    if claim_label == pred_label:
        exp3_t_ids.append(qid)
        if claim_label == 0:
            exp3_t_fakeids.append(qid)
        else:
            exp3_t_notfakeids.append(qid)
    else:
        exp3_f_ids.append(qid)
        if claim_label == 0:
            exp3_f_fakeids.append(qid)
        else:
            exp3_f_notfakeids.append(qid)

with open("/root/lmy/GET/logs/get/log_results_Snopes_base/Fold_4/result_123756.txt_error_analysis_testing.json", "r") as f1:
    base4_test = json.load(f1)
    # print(type(base0_test))
    #claim_label  predicted_prob qid
base4_f_ids = []
base4_f_fakeids = []
base4_f_notfakeids = []
base4_t_ids = []
base4_t_fakeids = []
base4_t_notfakeids = []
for item in base4_test:
    claim_label = int(item["claim_label"])
    predicted_prob = np.array(item["predicted_prob"])
    pred_label = np.argmax(predicted_prob)
    qid = item["qid"]
    if claim_label == pred_label:
        base4_t_ids.append(qid)
        if claim_label == 0:
            base4_t_fakeids.append(qid)
        else:
            base4_t_notfakeids.append(qid)
    else:
        base4_f_ids.append(qid)
        if claim_label == 0:
            base4_f_fakeids.append(qid)
        else:
            base4_f_notfakeids.append(qid)
with open("/root/lmy/GET/logs/get/log_results_Snopes_exp/Fold_4/result_123756.txt_error_analysis_testing.json", "r") as f2:
    exp4_test = json.load(f2)
exp4_f_ids = []
exp4_f_fakeids = []
exp4_f_notfakeids = []
exp4_t_ids = []
exp4_t_fakeids = []
exp4_t_notfakeids = []
for item in exp4_test:
    claim_label = int(item["claim_label"])
    predicted_prob = np.array(item["predicted_prob"])
    pred_label = np.argmax(predicted_prob)
    qid = item["qid"]
    if claim_label == pred_label:
        exp4_t_ids.append(qid)
        if claim_label == 0:
            exp4_t_fakeids.append(qid)
        else:
            exp4_t_notfakeids.append(qid)
    else:
        exp4_f_ids.append(qid)
        if claim_label == 0:
            exp4_f_fakeids.append(qid)
        else:
            exp4_f_notfakeids.append(qid)

basef_expt_0 = list(set(base0_f_ids) & set(exp0_t_ids))
baset_expf_0 = list(set(base0_t_ids) & set(exp0_f_ids))
baset_expt_0 = list(set(base0_t_ids) & set(exp0_t_ids))
basef_expf_0 = list(set(base0_f_ids) & set(exp0_f_ids))
basef_expt_0fake = []
basef_expt_0notfake = []
baset_expf_0fake = []
baset_expf_0notfake = []
baset_expt_0fake = []
baset_expt_0notfake = []
basef_expf_0fake = []
basef_expf_0notfake = []
for item in basef_expt_0:
    if (item in base0_f_fakeids):
        basef_expt_0fake.append(item)
    else:
        basef_expt_0notfake.append(item)
for item in baset_expf_0:
    if (item in base0_t_fakeids):
        baset_expf_0fake.append(item)
    else:
        baset_expf_0notfake.append(item)
for item in baset_expt_0:
    if (item in base0_t_fakeids):
        baset_expt_0fake.append(item)
    else:
        baset_expt_0notfake.append(item)
for item in basef_expf_0:
    if (item in base0_f_fakeids):
        basef_expf_0fake.append(item)
    else:
        basef_expf_0notfake.append(item)



basef_expt_1 = list(set(base1_f_ids) & set(exp1_t_ids))
baset_expf_1 = list(set(base1_t_ids) & set(exp1_f_ids))
baset_expt_1 = list(set(base1_t_ids) & set(exp1_t_ids))
basef_expf_1 = list(set(base1_f_ids) & set(exp1_f_ids))
basef_expt_1fake = []
basef_expt_1notfake = []
baset_expf_1fake = []
baset_expf_1notfake = []
baset_expt_1fake = []
baset_expt_1notfake = []
basef_expf_1fake = []
basef_expf_1notfake = []
for item in basef_expt_1:
    if (item in base1_f_fakeids):
        basef_expt_1fake.append(item)
    else:
        basef_expt_1notfake.append(item)
for item in baset_expf_1:
    if (item in base1_t_fakeids):
        baset_expf_1fake.append(item)
    else:
        baset_expf_1notfake.append(item)
for item in baset_expt_1:
    if (item in base1_t_fakeids):
        baset_expt_1fake.append(item)
    else:
        baset_expt_1notfake.append(item)
for item in basef_expf_1:
    if (item in base1_f_fakeids):
        basef_expf_1fake.append(item)
    else:
        basef_expf_1notfake.append(item)

basef_expt_2 = list(set(base2_f_ids) & set(exp2_t_ids))
baset_expf_2 = list(set(base2_t_ids) & set(exp2_f_ids))
baset_expt_2 = list(set(base2_t_ids) & set(exp2_t_ids))
basef_expf_2 = list(set(base2_f_ids) & set(exp2_f_ids))
basef_expt_2fake = []
basef_expt_2notfake = []
baset_expf_2fake = []
baset_expf_2notfake = []
baset_expt_2fake = []
baset_expt_2notfake = []
basef_expf_2fake = []
basef_expf_2notfake = []
for item in basef_expt_2:
    if (item in base2_f_fakeids):
        basef_expt_2fake.append(item)
    else:
        basef_expt_2notfake.append(item)
for item in baset_expf_2:
    if (item in base2_t_fakeids):
        baset_expf_2fake.append(item)
    else:
        baset_expf_2notfake.append(item)
for item in baset_expt_2:
    if (item in base2_t_fakeids):
        baset_expt_2fake.append(item)
    else:
        baset_expt_2notfake.append(item)
for item in basef_expf_2:
    if (item in base2_f_fakeids):
        basef_expf_2fake.append(item)
    else:
        basef_expf_2notfake.append(item)

basef_expt_3 = list(set(base3_f_ids) & set(exp3_t_ids))
baset_expf_3 = list(set(base3_t_ids) & set(exp3_f_ids))
baset_expt_3 = list(set(base3_t_ids) & set(exp3_t_ids))
basef_expf_3 = list(set(base3_f_ids) & set(exp3_f_ids))
basef_expt_3fake = []
basef_expt_3notfake = []
baset_expf_3fake = []
baset_expf_3notfake = []
baset_expt_3fake = []
baset_expt_3notfake = []
basef_expf_3fake = []
basef_expf_3notfake = []
for item in basef_expt_3:
    if (item in base3_f_fakeids):
        basef_expt_3fake.append(item)
    else:
        basef_expt_3notfake.append(item)
for item in baset_expf_3:
    if (item in base3_t_fakeids):
        baset_expf_3fake.append(item)
    else:
        baset_expf_3notfake.append(item)
for item in baset_expt_3:
    if (item in base3_t_fakeids):
        baset_expt_3fake.append(item)
    else:
        baset_expt_3notfake.append(item)
for item in basef_expf_3:
    if (item in base3_f_fakeids):
        basef_expf_3fake.append(item)
    else:
        basef_expf_3notfake.append(item)

basef_expt_4 = list(set(base4_f_ids) & set(exp4_t_ids))
baset_expf_4 = list(set(base4_t_ids) & set(exp4_f_ids))
baset_expt_4 = list(set(base4_t_ids) & set(exp4_t_ids))
basef_expf_4 = list(set(base4_f_ids) & set(exp4_f_ids))
basef_expt_4fake = []
basef_expt_4notfake = []
baset_expf_4fake = []
baset_expf_4notfake = []
baset_expt_4fake = []
baset_expt_4notfake = []
basef_expf_4fake = []
basef_expf_4notfake = []
for item in basef_expt_4:
    if (item in base4_f_fakeids):
        basef_expt_4fake.append(item)
    else:
        basef_expt_4notfake.append(item)
for item in baset_expf_4:
    if (item in base4_t_fakeids):
        baset_expf_4fake.append(item)
    else:
        baset_expf_4notfake.append(item)
for item in baset_expt_4:
    if (item in base4_t_fakeids):
        baset_expt_4fake.append(item)
    else:
        baset_expt_4notfake.append(item)
for item in basef_expf_4:
    if (item in base4_f_fakeids):
        basef_expf_4fake.append(item)
    else:
        basef_expf_4notfake.append(item)

data = { 'fold0' : {
        'baset': str(base0_t_ids),
        'baset_len': len(base0_t_ids),
        'baset_fake':str(base0_t_fakeids),
        'baset_fake_len':len(base0_t_fakeids),
        'baset_notfake':str(base0_t_notfakeids),
        'baset_notfake_len':len(base0_t_notfakeids),
        'basef': str(base0_f_ids),
        'basef_len': len(base0_f_ids),
        'basef_fake':str(base0_f_fakeids),
        'basef_fake_len':len(base0_f_fakeids),
        'basef_notfake':str(base0_f_notfakeids),
        'basef_notfake_len':len(base0_f_notfakeids),
        'expt': str(exp0_t_ids),
        'expt_len': len(exp0_t_ids),
        'expt_fake':str(exp0_t_fakeids),
        'expt_fake_len':len(exp0_t_fakeids),
        'expt_notfake':str(exp0_t_notfakeids),
        'expt_notfake_len':len(exp0_t_notfakeids),
        'expf': str(exp0_f_ids),
        'expf_len': len(exp0_f_ids),
        'expf_fake':str(exp0_f_fakeids),
        'expf_fake_len':len(exp0_f_fakeids),
        'expf_notfake':str(exp0_f_notfakeids),
        'expf_notfake_len':len(exp0_f_notfakeids),
        'basef_expt': str(basef_expt_0),
        'basef_expt_len': len(basef_expt_0),
        'basef_expt_fake':str(basef_expt_0fake),
        'basef_expt_fake_len':len(basef_expt_0fake),
        'basef_expt_notfake':str(basef_expt_0notfake),
        'basef_expt_notfake_len':len(basef_expt_0notfake),
        'baset_expf': str(baset_expf_0),
        'baset_expf_len': len(baset_expf_0),
        'baset_expf_fake':str(baset_expf_0fake),
        'baset_expf_fake_len':len(baset_expf_0fake),
        'baset_expf_notfake':str(baset_expf_0notfake),
        'baset_expf_notfake_len':len(baset_expf_0notfake),
        'baset_expt': str(baset_expt_0),
        'baset_expt_len': len(baset_expt_0),
        'baset_expt_fake':str(baset_expt_0fake),
        'baset_expt_fake_len':len(baset_expt_0fake),
        'baset_expt_notfake':str(baset_expt_0notfake),
        'baset_expt_notfake_len':len(baset_expt_0notfake),
        'basef_expf': str(basef_expf_0),
        'basef_expf_len': len(basef_expf_0),
        'basef_expf_fake':str(basef_expf_0fake),
        'basef_expf_fake_len':len(basef_expf_0fake),
        'basef_expf_notfake':str(basef_expf_0notfake),
        'basef_expf_notfake_len':len(basef_expf_0notfake),
        }, 
        'fold1' : {
        'baset': str(base1_t_ids),
        'baset_len': len(base1_t_ids),
        'baset_fake':str(base1_t_fakeids),
        'baset_fake_len':len(base1_t_fakeids),
        'baset_notfake':str(base1_t_notfakeids),
        'baset_notfake_len':len(base1_t_notfakeids),
        'basef': str(base1_f_ids),
        'basef_len': len(base1_f_ids),
        'basef_fake':str(base1_f_fakeids),
        'basef_fake_len':len(base1_f_fakeids),
        'basef_notfake':str(base1_f_notfakeids),
        'basef_notfake_len':len(base1_f_notfakeids),
        'expt': str(exp1_t_ids),
        'expt_len': len(exp1_t_ids),
        'expt_fake':str(exp1_t_fakeids),
        'expt_fake_len':len(exp1_t_fakeids),
        'expt_notfake':str(exp1_t_notfakeids),
        'expt_notfake_len':len(exp1_t_notfakeids),
        'expf': str(exp1_f_ids),
        'expf_len': len(exp1_f_ids),
        'expf_fake':str(exp1_f_fakeids),
        'expf_fake_len':len(exp1_f_fakeids),
        'expf_notfake':str(exp1_f_notfakeids),
        'expf_notfake_len':len(exp1_f_notfakeids),
        'basef_expt': str(basef_expt_1),
        'basef_expt_len': len(basef_expt_1),
        'basef_expt_fake':str(basef_expt_1fake),
        'basef_expt_fake_len':len(basef_expt_1fake),
        'basef_expt_notfake':str(basef_expt_1notfake),
        'basef_expt_notfake_len':len(basef_expt_1notfake),
        'baset_expf': str(baset_expf_1),
        'baset_expf_len': len(baset_expf_1),
        'baset_expf_fake':str(baset_expf_1fake),
        'baset_expf_fake_len':len(baset_expf_1fake),
        'baset_expf_notfake':str(baset_expf_1notfake),
        'baset_expf_notfake_len':len(baset_expf_1notfake),
        'baset_expt': str(baset_expt_1),
        'baset_expt_len': len(baset_expt_1),
        'baset_expt_fake':str(baset_expt_1fake),
        'baset_expt_fake_len':len(baset_expt_1fake),
        'baset_expt_notfake':str(baset_expt_1notfake),
        'baset_expt_notfake_len':len(baset_expt_1notfake),
        'basef_expf': str(basef_expf_1),
        'basef_expf_len': len(basef_expf_1),
        'basef_expf_fake':str(basef_expf_1fake),
        'basef_expf_fake_len':len(basef_expf_1fake),
        'basef_expf_notfake':str(basef_expf_1notfake),
        'basef_expf_notfake_len':len(basef_expf_1notfake),
        }, 'fold2' : {
        'baset': str(base2_t_ids),
        'baset_len': len(base2_t_ids),
        'baset_fake':str(base2_t_fakeids),
        'baset_fake_len':len(base2_t_fakeids),
        'baset_notfake':str(base2_t_notfakeids),
        'baset_notfake_len':len(base2_t_notfakeids),
        'basef': str(base2_f_ids),
        'basef_len': len(base2_f_ids),
        'basef_fake':str(base2_f_fakeids),
        'basef_fake_len':len(base2_f_fakeids),
        'basef_notfake':str(base2_f_notfakeids),
        'basef_notfake_len':len(base2_f_notfakeids),
        'expt': str(exp2_t_ids),
        'expt_len': len(exp2_t_ids),
        'expt_fake':str(exp2_t_fakeids),
        'expt_fake_len':len(exp2_t_fakeids),
        'expt_notfake':str(exp2_t_notfakeids),
        'expt_notfake_len':len(exp2_t_notfakeids),
        'expf': str(exp2_f_ids),
        'expf_len': len(exp2_f_ids),
        'expf_fake':str(exp2_f_fakeids),
        'expf_fake_len':len(exp2_f_fakeids),
        'expf_notfake':str(exp2_f_notfakeids),
        'expf_notfake_len':len(exp2_f_notfakeids),
        'basef_expt': str(basef_expt_2),
        'basef_expt_len': len(basef_expt_2),
        'basef_expt_fake':str(basef_expt_2fake),
        'basef_expt_fake_len':len(basef_expt_2fake),
        'basef_expt_notfake':str(basef_expt_2notfake),
        'basef_expt_notfake_len':len(basef_expt_2notfake),
        'baset_expf': str(baset_expf_2),
        'baset_expf_len': len(baset_expf_2),
        'baset_expf_fake':str(baset_expf_2fake),
        'baset_expf_fake_len':len(baset_expf_2fake),
        'baset_expf_notfake':str(baset_expf_2notfake),
        'baset_expf_notfake_len':len(baset_expf_2notfake),
        'baset_expt': str(baset_expt_2),
        'baset_expt_len': len(baset_expt_2),
        'baset_expt_fake':str(baset_expt_2fake),
        'baset_expt_fake_len':len(baset_expt_2fake),
        'baset_expt_notfake':str(baset_expt_2notfake),
        'baset_expt_notfake_len':len(baset_expt_2notfake),
        'basef_expf': str(basef_expf_2),
        'basef_expf_len': len(basef_expf_2),
        'basef_expf_fake':str(basef_expf_2fake),
        'basef_expf_fake_len':len(basef_expf_2fake),
        'basef_expf_notfake':str(basef_expf_2notfake),
        'basef_expf_notfake_len':len(basef_expf_2notfake),
        }, 'fold3' : {
        'baset': str(base3_t_ids),
        'baset_len': len(base3_t_ids),
        'baset_fake':str(base3_t_fakeids),
        'baset_fake_len':len(base3_t_fakeids),
        'baset_notfake':str(base3_t_notfakeids),
        'baset_notfake_len':len(base3_t_notfakeids),
        'basef': str(base3_f_ids),
        'basef_len': len(base3_f_ids),
        'basef_fake':str(base3_f_fakeids),
        'basef_fake_len':len(base3_f_fakeids),
        'basef_notfake':str(base3_f_notfakeids),
        'basef_notfake_len':len(base3_f_notfakeids),
        'expt': str(exp3_t_ids),
        'expt_len': len(exp3_t_ids),
        'expt_fake':str(exp3_t_fakeids),
        'expt_fake_len':len(exp3_t_fakeids),
        'expt_notfake':str(exp3_t_notfakeids),
        'expt_notfake_len':len(exp3_t_notfakeids),
        'expf': str(exp3_f_ids),
        'expf_len': len(exp3_f_ids),
        'expf_fake':str(exp3_f_fakeids),
        'expf_fake_len':len(exp3_f_fakeids),
        'expf_notfake':str(exp3_f_notfakeids),
        'expf_notfake_len':len(exp3_f_notfakeids),
        'basef_expt': str(basef_expt_3),
        'basef_expt_len': len(basef_expt_3),
        'basef_expt_fake':str(basef_expt_3fake),
        'basef_expt_fake_len':len(basef_expt_3fake),
        'basef_expt_notfake':str(basef_expt_3notfake),
        'basef_expt_notfake_len':len(basef_expt_3notfake),
        'baset_expf': str(baset_expf_3),
        'baset_expf_len': len(baset_expf_3),
        'baset_expf_fake':str(baset_expf_3fake),
        'baset_expf_fake_len':len(baset_expf_3fake),
        'baset_expf_notfake':str(baset_expf_3notfake),
        'baset_expf_notfake_len':len(baset_expf_3notfake),
        'baset_expt': str(baset_expt_3),
        'baset_expt_len': len(baset_expt_3),
        'baset_expt_fake':str(baset_expt_3fake),
        'baset_expt_fake_len':len(baset_expt_3fake),
        'baset_expt_notfake':str(baset_expt_3notfake),
        'baset_expt_notfake_len':len(baset_expt_3notfake),
        'basef_expf': str(basef_expf_3),
        'basef_expf_len': len(basef_expf_3),
        'basef_expf_fake':str(basef_expf_3fake),
        'basef_expf_fake_len':len(basef_expf_3fake),
        'basef_expf_notfake':str(basef_expf_3notfake),
        'basef_expf_notfake_len':len(basef_expf_3notfake),
        }, 'fold4' : {
        'baset': str(base4_t_ids),
        'baset_len': len(base4_t_ids),
        'baset_fake':str(base4_t_fakeids),
        'baset_fake_len':len(base4_t_fakeids),
        'baset_notfake':str(base4_t_notfakeids),
        'baset_notfake_len':len(base4_t_notfakeids),
        'basef': str(base4_f_ids),
        'basef_len': len(base4_f_ids),
        'basef_fake':str(base4_f_fakeids),
        'basef_fake_len':len(base4_f_fakeids),
        'basef_notfake':str(base4_f_notfakeids),
        'basef_notfake_len':len(base4_f_notfakeids),
        'expt': str(exp4_t_ids),
        'expt_len': len(exp4_t_ids),
        'expt_fake':str(exp4_t_fakeids),
        'expt_fake_len':len(exp4_t_fakeids),
        'expt_notfake':str(exp4_t_notfakeids),
        'expt_notfake_len':len(exp4_t_notfakeids),
        'expf': str(exp4_f_ids),
        'expf_len': len(exp4_f_ids),
        'expf_fake':str(exp4_f_fakeids),
        'expf_fake_len':len(exp4_f_fakeids),
        'expf_notfake':str(exp4_f_notfakeids),
        'expf_notfake_len':len(exp4_f_notfakeids),
        'basef_expt': str(basef_expt_4),
        'basef_expt_len': len(basef_expt_4),
        'basef_expt_fake':str(basef_expt_4fake),
        'basef_expt_fake_len':len(basef_expt_4fake),
        'basef_expt_notfake':str(basef_expt_4notfake),
        'basef_expt_notfake_len':len(basef_expt_4notfake),
        'baset_expf': str(baset_expf_4),
        'baset_expf_len': len(baset_expf_4),
        'baset_expf_fake':str(baset_expf_4fake),
        'baset_expf_fake_len':len(baset_expf_4fake),
        'baset_expf_notfake':str(baset_expf_4notfake),
        'baset_expf_notfake_len':len(baset_expf_4notfake),
        'baset_expt': str(baset_expt_4),
        'baset_expt_len': len(baset_expt_4),
        'baset_expt_fake':str(baset_expt_4fake),
        'baset_expt_fake_len':len(baset_expt_4fake),
        'baset_expt_notfake':str(baset_expt_4notfake),
        'baset_expt_notfake_len':len(baset_expt_4notfake),
        'basef_expf': str(basef_expf_4),
        'basef_expf_len': len(basef_expf_4),
        'basef_expf_fake':str(basef_expf_4fake),
        'basef_expf_fake_len':len(basef_expf_4fake),
        'basef_expf_notfake':str(basef_expf_4notfake),
        'basef_expf_notfake_len':len(basef_expf_4notfake),
        } } 

data2 = json.dumps(data, indent=4, separators=(',', ': '))
with open("/root/lmy/GET/result_tf_fakeornot.json", 'w') as fin:
    fin.write(data2)