import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import seaborn as sns
import pandas as pd
import numpy as np
import json
from pathlib import Path

shots = [1, 5, 10, 20]

deepweeds_metrics = {
    'ViT-B16': {
        'Accuracy': [0.2789, 0.4164, 0.5197, 0.5971],
        'Accuracy_err': [0.0376, 0.0227, 0.0213, 0.0146],
        'Precision': [0.2925, 0.4514, 0.5289, 0.6018],
        'Precision_err': [0.0355, 0.0275, 0.0219, 0.0130],
        'Recall': [0.2776, 0.4158, 0.5199, 0.5974],
        'Recall_err': [0.0381, 0.0229, 0.0208, 0.0144],
    },
    'DenseNet121': {
        'Accuracy': [0.2364, 0.4380, 0.5444, 0.6143],
        'Accuracy_err': [0.0303, 0.0175, 0.0149, 0.0130],
        'Precision': [0.2601, 0.4527, 0.5504, 0.6168],
        'Precision_err': [0.0345, 0.0202, 0.0210, 0.0125],
        'Recall': [0.2377, 0.4389, 0.5452, 0.6146],
        'Recall_err': [0.0308, 0.0176, 0.0143, 0.0130],
    },
    'EfficientNet-B0': {
        'Accuracy': [0.2489, 0.4249, 0.5217, 0.5814],
        'Accuracy_err': [0.0295, 0.0272, 0.0211, 0.0165],
        'Precision': [0.2640, 0.4451, 0.5240, 0.5810],
        'Precision_err': [0.0342, 0.0272, 0.0236, 0.0153],
        'Recall': [0.2494, 0.4247, 0.5218, 0.5811],
        'Recall_err': [0.0301, 0.0273, 0.0204, 0.0167],
    },
    'VGG16': {
        'Accuracy': [0.2372, 0.3213, 0.3801, 0.4565],
        'Accuracy_err': [0.0273, 0.0280, 0.0240, 0.0183],
        'Precision': [0.2685, 0.3612, 0.4241, 0.4758],
        'Precision_err': [0.0615, 0.0612, 0.0432, 0.0143],
        'Recall': [0.2387, 0.3229, 0.3817, 0.4573],
        'Recall_err': [0.0276, 0.0287, 0.0242, 0.0186],
    },
    'ResNet18': {
        'Accuracy': [0.1991, 0.3058, 0.4407, 0.5316],
        'Accuracy_err': [0.0229, 0.0260, 0.0193, 0.0152],
        'Precision': [0.2345, 0.3590, 0.4521, 0.5357],
        'Precision_err': [0.0279, 0.0311, 0.0237, 0.0121],
        'Recall': [0.1998, 0.3078, 0.4413, 0.5322],
        'Recall_err': [0.0237, 0.0253, 0.0189, 0.0148],
    },
    'Fusion Baseline': {
        'Accuracy': [0.1767, 0.2518, 0.3035, 0.3647],
        'Accuracy_err': [0.0323, 0.0248, 0.0236, 0.0192],
        'Precision': [0.1926, 0.2642, 0.3127, 0.3712],
        'Precision_err': [0.0290, 0.0267, 0.0297, 0.0208],
        'Recall': [0.1767, 0.2518, 0.3035, 0.3647],
        'Recall_err': [0.0323, 0.0248, 0.0236, 0.0192],
    },
    'Dirichlet Attention': {
        'Accuracy': [0.3082, 0.5134, 0.6059, 0.6821],
        'Accuracy_err': [0.0387, 0.0311, 0.0233, 0.0169],
        'Precision': [0.3223, 0.5269, 0.6160, 0.6944],
        'Precision_err': [0.0328, 0.0316, 0.0252, 0.0137],
        'Recall': [0.3082, 0.5134, 0.6059, 0.6821],
        'Recall_err': [0.0387, 0.0311, 0.0233, 0.0169],
    },
}

deepweeds_rows = []
for model_name, vals in deepweeds_metrics.items():
    for i, shot in enumerate(shots):
        deepweeds_rows.append(
            {
                'Shot': shot,
                'Model': model_name,
                'Accuracy': vals['Accuracy'][i],
                'Accuracy_err': vals['Accuracy_err'][i],
                'Precision': vals['Precision'][i],
                'Precision_err': vals['Precision_err'][i],
                'Recall': vals['Recall'][i],
                'Recall_err': vals['Recall_err'][i],
            }
        )
deepweeds_df = pd.DataFrame(deepweeds_rows)

aptos_metrics = {
    'ViT-B16': {
        'Accuracy': [0.3239, 0.4811, 0.5042, 0.5427],
        'Accuracy_err': [0.1190, 0.0389, 0.0331, 0.0261],
        'Precision': [0.3079, 0.3640, 0.3890, 0.4226],
        'Precision_err': [0.0455, 0.0274, 0.0184, 0.0276],
        'Recall': [0.2915, 0.3881, 0.4102, 0.4183],
        'Recall_err': [0.0488, 0.0329, 0.0274, 0.0226],
    },
    'DenseNet121': {
        'Accuracy': [0.3665, 0.5454, 0.5958, 0.6387],
        'Accuracy_err': [0.1098, 0.0779, 0.0361, 0.0199],
        'Precision': [0.3016, 0.3947, 0.4491, 0.4808],
        'Precision_err': [0.0345, 0.0423, 0.0355, 0.0196],
        'Recall': [0.2878, 0.3805, 0.4691, 0.5111],
        'Recall_err': [0.0390, 0.0377, 0.0305, 0.0241],
    },
    'EfficientNet-B0': {
        'Accuracy': [0.4296, 0.5421, 0.5845, 0.6239],
        'Accuracy_err': [0.1065, 0.0441, 0.0540, 0.0224],
        'Precision': [0.2747, 0.3438, 0.4176, 0.4574],
        'Precision_err': [0.0567, 0.0481, 0.0288, 0.0210],
        'Recall': [0.2681, 0.3104, 0.4069, 0.4733],
        'Recall_err': [0.0425, 0.0375, 0.0259, 0.0199],
    },
    'ResNet18': {
        'Accuracy': [0.4059, 0.5761, 0.5521, 0.5926],
        'Accuracy_err': [0.1230, 0.0626, 0.0440, 0.0188],
        'Precision': [0.3154, 0.3704, 0.3982, 0.4216],
        'Precision_err': [0.0369, 0.0342, 0.0173, 0.0161],
        'Recall': [0.2953, 0.3481, 0.4138, 0.4304],
        'Recall_err': [0.0399, 0.0239, 0.0233, 0.0201],
    },
    'VGG16': {
        'Accuracy': [0.3669, 0.4350, 0.5827, 0.5573],
        'Accuracy_err': [0.1533, 0.1056, 0.0558, 0.0327],
        'Precision': [0.2604, 0.3565, 0.3801, 0.3660],
        'Precision_err': [0.0554, 0.0516, 0.0656, 0.0432],
        'Recall': [0.2710, 0.3249, 0.3684, 0.4015],
        'Recall_err': [0.0521, 0.0320, 0.0268, 0.0198],
    },
    'Fusion Baseline': {
        'Accuracy': [0.2439, 0.3346, 0.3801, 0.4686],
        'Accuracy_err': [0.0649, 0.0875, 0.0909, 0.0677],
        'Precision': [0.4515, 0.5055, 0.4912, 0.5663],
        'Precision_err': [0.1230, 0.0694, 0.0531, 0.0399],
        'Recall': [0.2439, 0.3346, 0.3801, 0.4686],
        'Recall_err': [0.0649, 0.0875, 0.0909, 0.0677],
    },
    'Dirichlet Attention': {
        'Accuracy': [0.3835, 0.5694, 0.5995, 0.6325],
        'Accuracy_err': [0.1228, 0.0548, 0.0454, 0.0449],
        'Precision': [0.5409, 0.6435, 0.6696, 0.7143],
        'Precision_err': [0.0748, 0.0248, 0.0260, 0.0208],
        'Recall': [0.3835, 0.5694, 0.5995, 0.6325],
        'Recall_err': [0.1228, 0.0548, 0.0454, 0.0449],
    },
}

aptos_rows = []
for model_name, vals in aptos_metrics.items():
    for i, shot in enumerate(shots):
        aptos_rows.append(
            {
                'Shot': shot,
                'Model': model_name,
                'Accuracy': vals['Accuracy'][i],
                'Accuracy_err': vals['Accuracy_err'][i],
                'Precision': vals['Precision'][i],
                'Precision_err': vals['Precision_err'][i],
                'Recall': vals['Recall'][i],
                'Recall_err': vals['Recall_err'][i],
            }
        )
aptos_df = pd.DataFrame(aptos_rows)

pathmnist_metrics = {
    'ViT-B16': {
        'Accuracy': [0.5086, 0.7205, 0.7670, 0.8287],
        'Accuracy_err': [0.0554, 0.0467, 0.0318, 0.0175],
        'Precision': [0.4941, 0.6951, 0.7443, 0.7997],
        'Precision_err': [0.0479, 0.0405, 0.0320, 0.0207],
        'Recall': [0.5114, 0.7045, 0.7560, 0.8028],
        'Recall_err': [0.0494, 0.0374, 0.0262, 0.0172],
    },
    'DenseNet121': {
        'Accuracy': [0.4928, 0.7452, 0.8350, 0.8850],
        'Accuracy_err': [0.0594, 0.0317, 0.0151, 0.0107],
        'Precision': [0.4654, 0.7019, 0.7936, 0.8549],
        'Precision_err': [0.0444, 0.0352, 0.0157, 0.0149],
        'Recall': [0.4570, 0.6878, 0.7936, 0.8425],
        'Recall_err': [0.0520, 0.0334, 0.0165, 0.0198],
    },
    'EfficientNet-B0': {
        'Accuracy': [0.3511, 0.6883, 0.7901, 0.8427],
        'Accuracy_err': [0.0606, 0.0302, 0.0271, 0.0144],
        'Precision': [0.3208, 0.6429, 0.7446, 0.8017],
        'Precision_err': [0.0444, 0.0327, 0.0277, 0.0155],
        'Recall': [0.2977, 0.6292, 0.7386, 0.7875],
        'Recall_err': [0.0450, 0.0312, 0.0233, 0.0149],
    },
    'ResNet18': {
        'Accuracy': [0.4081, 0.6364, 0.7751, 0.8364],
        'Accuracy_err': [0.0553, 0.0335, 0.0222, 0.0124],
        'Precision': [0.4379, 0.6530, 0.7345, 0.7897],
        'Precision_err': [0.0522, 0.0368, 0.0211, 0.0154],
        'Recall': [0.3597, 0.5900, 0.7220, 0.7884],
        'Recall_err': [0.0399, 0.0293, 0.0213, 0.0169],
    },
    'VGG16': {
        'Accuracy': [0.4645, 0.6108, 0.6661, 0.7323],
        'Accuracy_err': [0.0592, 0.0409, 0.0364, 0.0304],
        'Precision': [0.4726, 0.6107, 0.6664, 0.7220],
        'Precision_err': [0.0462, 0.0481, 0.0359, 0.0280],
        'Recall': [0.4286, 0.5438, 0.5961, 0.6831],
        'Recall_err': [0.0462, 0.0401, 0.0284, 0.0257],
    },
    'Fusion Baseline': {
        'Accuracy': [0.2606, 0.3918, 0.4790, 0.6398],
        'Accuracy_err': [0.0298, 0.0416, 0.0373, 0.0252],
        'Precision': [0.3660, 0.5132, 0.5437, 0.6776],
        'Precision_err': [0.0430, 0.0455, 0.0394, 0.0266],
        'Recall': [0.2606, 0.3918, 0.4790, 0.6398],
        'Recall_err': [0.0298, 0.0416, 0.0373, 0.0252],
    },
    'Dirichlet Attention': {
        'Accuracy': [0.5256, 0.7618, 0.8138, 0.8546],
        'Accuracy_err': [0.0479, 0.0423, 0.0354, 0.0148],
        'Precision': [0.5994, 0.7913, 0.8373, 0.8696],
        'Precision_err': [0.0603, 0.0303, 0.0302, 0.0129],
        'Recall': [0.5256, 0.7618, 0.8138, 0.8546],
        'Recall_err': [0.0479, 0.0423, 0.0354, 0.0148],
    },
}

pathmnist_rows = []
for model_name, vals in pathmnist_metrics.items():
    for i, shot in enumerate(shots):
        pathmnist_rows.append(
            {
                'Shot': shot,
                'Model': model_name,
                'Accuracy': vals['Accuracy'][i],
                'Accuracy_err': vals['Accuracy_err'][i],
                'Precision': vals['Precision'][i],
                'Precision_err': vals['Precision_err'][i],
                'Recall': vals['Recall'][i],
                'Recall_err': vals['Recall_err'][i],
            }
        )
pathmnist_df = pd.DataFrame(pathmnist_rows)

ninco_metrics = {
    'ViT-B16': {
        'Accuracy': [0.3961, 0.7359, 0.7935, 0.8336],
        'Accuracy_err': [0.0234, 0.0126, 0.0090, 0.0075],
        'Precision': [0.4519, 0.7501, 0.8019, 0.8369],
        'Precision_err': [0.0300, 0.0117, 0.0087, 0.0072],
        'Recall': [0.4003, 0.7425, 0.8005, 0.8406],
        'Recall_err': [0.0239, 0.0119, 0.0089, 0.0075],
    },
    'DenseNet121': {
        'Accuracy': [0.3933, 0.7460, 0.8085, 0.8417],
        'Accuracy_err': [0.0224, 0.0081, 0.0088, 0.0047],
        'Precision': [0.4233, 0.7607, 0.8167, 0.8475],
        'Precision_err': [0.0283, 0.0097, 0.0100, 0.0060],
        'Recall': [0.4014, 0.7531, 0.8147, 0.8480],
        'Recall_err': [0.0233, 0.0084, 0.0096, 0.0056],
    },
    'EfficientNet-B0': {
        'Accuracy': [0.3842, 0.7428, 0.8066, 0.8394],
        'Accuracy_err': [0.0232, 0.0121, 0.0080, 0.0069],
        'Precision': [0.4550, 0.7545, 0.8116, 0.8423],
        'Precision_err': [0.0300, 0.0126, 0.0081, 0.0069],
        'Recall': [0.3930, 0.7518, 0.8135, 0.8467],
        'Recall_err': [0.0256, 0.0115, 0.0073, 0.0066],
    },
    'ResNet18': {
        'Accuracy': [0.2759, 0.6753, 0.7541, 0.7987],
        'Accuracy_err': [0.0183, 0.0112, 0.0084, 0.0071],
        'Precision': [0.2989, 0.6861, 0.7611, 0.8090],
        'Precision_err': [0.0233, 0.0121, 0.0091, 0.0076],
        'Recall': [0.2812, 0.6864, 0.7626, 0.8078],
        'Recall_err': [0.0195, 0.0103, 0.0085, 0.0068],
    },
    'VGG16': {
        'Accuracy': [0.3667, 0.5835, 0.6515, 0.7035],
        'Accuracy_err': [0.0181, 0.0149, 0.0103, 0.0070],
        'Precision': [0.3962, 0.6072, 0.6693, 0.7184],
        'Precision_err': [0.0307, 0.0167, 0.0113, 0.0083],
        'Recall': [0.3789, 0.5994, 0.6653, 0.7155],
        'Recall_err': [0.0189, 0.0132, 0.0100, 0.0072],
    },
    'Fusion Baseline': {
        'Accuracy': [0.1578, 0.3173, 0.3880, 0.4976],
        'Accuracy_err': [0.0175, 0.0170, 0.0135, 0.0168],
        'Precision': [0.1621, 0.2958, 0.3906, 0.5182],
        'Precision_err': [0.0202, 0.0286, 0.0244, 0.0289],
        'Recall': [0.1578, 0.3173, 0.3880, 0.4976],
        'Recall_err': [0.0175, 0.0170, 0.0135, 0.0168],
    },
    'Dirichlet Attention': {
        'Accuracy': [0.3684, 0.7482, 0.8207, 0.8381],
        'Accuracy_err': [0.0222, 0.0124, 0.0095, 0.0186],
        'Precision': [0.4055, 0.7715, 0.8335, 0.8549],
        'Precision_err': [0.0324, 0.0109, 0.0096, 0.0137],
        'Recall': [0.3684, 0.7482, 0.8207, 0.8381],
        'Recall_err': [0.0222, 0.0124, 0.0095, 0.0186],
    },
}

ninco_rows = []
for model_name, vals in ninco_metrics.items():
    for i, shot in enumerate(shots):
        ninco_rows.append(
            {
                'Shot': shot,
                'Model': model_name,
                'Accuracy': vals['Accuracy'][i],
                'Accuracy_err': vals['Accuracy_err'][i],
                'Precision': vals['Precision'][i],
                'Precision_err': vals['Precision_err'][i],
                'Recall': vals['Recall'][i],
                'Recall_err': vals['Recall_err'][i],
            }
        )
ninco_df = pd.DataFrame(ninco_rows)

plantvillage_metrics = {
    'ViT-B16': {
        'Accuracy': [0.2243, 0.6244, 0.7339, 0.8179],
        'Accuracy_err': [0.0273, 0.0178, 0.0152, 0.0078],
        'Precision': [0.3610, 0.5866, 0.6861, 0.7700],
        'Precision_err': [0.0290, 0.0114, 0.0117, 0.0086],
        'Recall': [0.2481, 0.6376, 0.7437, 0.8223],
        'Recall_err': [0.0189, 0.0106, 0.0097, 0.0072],
    },
    'DenseNet121': {
        'Accuracy': [0.3386, 0.7460, 0.8303, 0.8891],
        'Accuracy_err': [0.0248, 0.0153, 0.0085, 0.0052],
        'Precision': [0.4220, 0.7012, 0.7832, 0.8476],
        'Precision_err': [0.0184, 0.0115, 0.0092, 0.0060],
        'Recall': [0.3411, 0.7697, 0.8426, 0.8953],
        'Recall_err': [0.0192, 0.0100, 0.0079, 0.0043],
    },
    'EfficientNet-B0': {
        'Accuracy': [0.2907, 0.6902, 0.7812, 0.8444],
        'Accuracy_err': [0.0223, 0.0236, 0.0083, 0.0058],
        'Precision': [0.3635, 0.6460, 0.7284, 0.7932],
        'Precision_err': [0.0208, 0.0176, 0.0072, 0.0071],
        'Recall': [0.2935, 0.7159, 0.7940, 0.8504],
        'Recall_err': [0.0150, 0.0168, 0.0068, 0.0045],
    },
    'ResNet18': {
        'Accuracy': [0.1348, 0.6620, 0.7711, 0.8386],
        'Accuracy_err': [0.0187, 0.0168, 0.0094, 0.0067],
        'Precision': [0.3212, 0.6088, 0.7260, 0.7919],
        'Precision_err': [0.0366, 0.0143, 0.0083, 0.0085],
        'Recall': [0.1485, 0.6756, 0.7802, 0.8445],
        'Recall_err': [0.0118, 0.0101, 0.0081, 0.0063],
    },
    'VGG16': {
        'Accuracy': [0.0676, 0.5297, 0.6823, 0.7664],
        'Accuracy_err': [0.0099, 0.0248, 0.0166, 0.0069],
        'Precision': [0.1639, 0.5696, 0.6595, 0.7230],
        'Precision_err': [0.0468, 0.0152, 0.0095, 0.0076],
        'Recall': [0.1062, 0.5505, 0.7045, 0.7784],
        'Recall_err': [0.0125, 0.0171, 0.0125, 0.0062],
    },
    'Fusion Baseline': {
        'Accuracy': [0.0663, 0.1723, 0.2558, 0.3757],
        'Accuracy_err': [0.0177, 0.0352, 0.0412, 0.0315],
        'Precision': [0.1148, 0.2652, 0.3589, 0.4938],
        'Precision_err': [0.0314, 0.0272, 0.0341, 0.0276],
        'Recall': [0.0663, 0.1723, 0.2558, 0.3757],
        'Recall_err': [0.0177, 0.0352, 0.0412, 0.0315],
    },
    'Dirichlet Attention': {
        'Accuracy': [0.2611, 0.6860, 0.8115, 0.8480],
        'Accuracy_err': [0.0281, 0.0197, 0.0092, 0.0270],
        'Precision': [0.4321, 0.7637, 0.8437, 0.8769],
        'Precision_err': [0.0544, 0.0107, 0.0068, 0.0163],
        'Recall': [0.2611, 0.6860, 0.8115, 0.8480],
        'Recall_err': [0.0281, 0.0197, 0.0092, 0.0270],
    },
}

plantvillage_rows = []
for model_name, vals in plantvillage_metrics.items():
    for i, shot in enumerate(shots):
        plantvillage_rows.append(
            {
                'Shot': shot,
                'Model': model_name,
                'Accuracy': vals['Accuracy'][i],
                'Accuracy_err': vals['Accuracy_err'][i],
                'Precision': vals['Precision'][i],
                'Precision_err': vals['Precision_err'][i],
                'Recall': vals['Recall'][i],
                'Recall_err': vals['Recall_err'][i],
            }
        )
plantvillage_df = pd.DataFrame(plantvillage_rows)

cassava_metrics = {
    'ViT-B16': {
        'Accuracy': [0.2526, 0.3528, 0.4133, 0.4795],
        'Accuracy_err': [0.0826, 0.0475, 0.0317, 0.0370],
        'Precision': [0.2392, 0.2897, 0.3224, 0.3629],
        'Precision_err': [0.0249, 0.0162, 0.0113, 0.0142],
        'Recall': [0.2413, 0.3145, 0.3662, 0.4034],
        'Recall_err': [0.0334, 0.0257, 0.0163, 0.0149],
    },
    'DenseNet121': {
        'Accuracy': [0.2009, 0.2863, 0.3980, 0.5187],
        'Accuracy_err': [0.0608, 0.0846, 0.0453, 0.0341],
        'Precision': [0.2299, 0.2882, 0.3370, 0.3831],
        'Precision_err': [0.0261, 0.0177, 0.0164, 0.0128],
        'Recall': [0.2295, 0.3067, 0.3852, 0.4340],
        'Recall_err': [0.0350, 0.0178, 0.0205, 0.0148],
    },
    'EfficientNet-B0': {
        'Accuracy': [0.2324, 0.2840, 0.4236, 0.5314],
        'Accuracy_err': [0.0804, 0.0661, 0.0558, 0.0320],
        'Precision': [0.2248, 0.2858, 0.3320, 0.3769],
        'Precision_err': [0.0327, 0.0264, 0.0196, 0.0126],
        'Recall': [0.2287, 0.2894, 0.3661, 0.4198],
        'Recall_err': [0.0255, 0.0245, 0.0199, 0.0142],
    },
    'ResNet18': {
        'Accuracy': [0.3253, 0.3469, 0.3566, 0.4150],
        'Accuracy_err': [0.0898, 0.0545, 0.0354, 0.0355],
        'Precision': [0.2284, 0.2573, 0.2845, 0.3280],
        'Precision_err': [0.0183, 0.0095, 0.0097, 0.0121],
        'Recall': [0.2190, 0.2571, 0.2996, 0.3461],
        'Recall_err': [0.0243, 0.0145, 0.0160, 0.0162],
    },
    'VGG16': {
        'Accuracy': [0.2234, 0.3343, 0.2949, 0.4656],
        'Accuracy_err': [0.1254, 0.0951, 0.0520, 0.0339],
        'Precision': [0.2731, 0.2940, 0.3084, 0.3677],
        'Precision_err': [0.0583, 0.0199, 0.0128, 0.0231],
        'Recall': [0.2550, 0.3074, 0.3343, 0.3781],
        'Recall_err': [0.0388, 0.0159, 0.0160, 0.0150],
    },
    'Fusion Baseline': {
        'Accuracy': [0.2034, 0.2294, 0.2741, 0.2705],
        'Accuracy_err': [0.0717, 0.0605, 0.0733, 0.0514],
        'Precision': [0.4199, 0.4602, 0.4906, 0.5340],
        'Precision_err': [0.0492, 0.0357, 0.0350, 0.0349],
        'Recall': [0.2034, 0.2294, 0.2741, 0.2705],
        'Recall_err': [0.0717, 0.0605, 0.0733, 0.0514],
    },
    'Dirichlet Attention': {
        'Accuracy': [0.2800, 0.4095, 0.4136, 0.4508],
        'Accuracy_err': [0.0969, 0.0789, 0.0609, 0.0525],
        'Precision': [0.4858, 0.5689, 0.6010, 0.6342],
        'Precision_err': [0.0572, 0.0285, 0.0320, 0.0178],
        'Recall': [0.2800, 0.4095, 0.4136, 0.4508],
        'Recall_err': [0.0969, 0.0789, 0.0609, 0.0525],
    },
}

cassava_rows = []
for model_name, vals in cassava_metrics.items():
    for i, shot in enumerate(shots):
        cassava_rows.append(
            {
                'Shot': shot,
                'Model': model_name,
                'Accuracy': vals['Accuracy'][i],
                'Accuracy_err': vals['Accuracy_err'][i],
                'Precision': vals['Precision'][i],
                'Precision_err': vals['Precision_err'][i],
                'Recall': vals['Recall'][i],
                'Recall_err': vals['Recall_err'][i],
            }
        )
cassava_df = pd.DataFrame(cassava_rows)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
FUSION_RESULTS_ROOT = PROJECT_ROOT / "kshot" / "results"


def fusion_result_dir(source_name, dataset_key):
    return FUSION_RESULTS_ROOT / source_name / dataset_key


def apply_fusion_results_from_kshot():
    """Override hard-coded fusion rows with repo-local k-shot fusion summaries.

    The single-backbone rows are already macro in the current pipeline. This
    replaces only the two fusion methods for datasets whose historical rows
    are defined directly in this script. The primary source is:
    ddcomb/kshot/results/<stage_a_source>/<dataset>/.
    """
    source_to_model = {
        "img100_stagea_softmax_linear_resnet_order_v1": "Fusion Baseline",
        "img100_dirichlet_imagenet_1_acc85": "Dirichlet Attention",
    }
    dataset_to_df = {
        "deepweeds_no_bg_mergeval": deepweeds_df,
        "aptos2019_split": aptos_df,
        "pathmnist_224_imagefolder_mergeval": pathmnist_df,
        "cassava_leaf_disease_split": cassava_df,
    }

    patched = []
    for source_name, model_name in source_to_model.items():
        for dataset_key, df in dataset_to_df.items():
            result_dir = fusion_result_dir(source_name, dataset_key)
            for shot in shots:
                summary_path = result_dir / f"{shot}_shot_summary.json"
                if not summary_path.is_file():
                    continue
                payload = json.loads(summary_path.read_text())
                summary = payload["summary"]
                mask = (df["Model"] == model_name) & (df["Shot"] == shot)
                if not mask.any():
                    continue

                df.loc[mask, "Accuracy"] = float(summary["accuracy_mean"])
                df.loc[mask, "Accuracy_err"] = float(summary["accuracy_std"])
                df.loc[mask, "Precision"] = float(summary["precision_mean"])
                df.loc[mask, "Precision_err"] = float(summary["precision_std"])
                df.loc[mask, "Recall"] = float(summary["recall_mean"])
                df.loc[mask, "Recall_err"] = float(summary["recall_std"])
                patched.append((dataset_key, model_name, shot))

    print(f"Applied fusion summaries from {FUSION_RESULTS_ROOT}: {len(patched)} rows")


apply_fusion_results_from_kshot()


def load_summary_metric_row(summary_path, model_name, shot):
    payload = json.loads(summary_path.read_text())
    summary = payload["summary"]
    return {
        'Shot': shot,
        'Model': model_name,
        'Accuracy': float(summary['accuracy_mean']),
        'Accuracy_err': float(summary['accuracy_std']),
        'Precision': float(summary['precision_mean']),
        'Precision_err': float(summary['precision_std']),
        'Recall': float(summary['recall_mean']),
        'Recall_err': float(summary['recall_std']),
    }


def build_summary_df(dataset_key, dataset_label):
    rows = []
    single_backbone_sources = {
        'ViT-B16': PROJECT_ROOT / "kshot" / "results_single_backbone" / dataset_key / "vit_b16" / "head_only_basic",
        'DenseNet121': PROJECT_ROOT / "kshot" / "results_single_backbone" / dataset_key / "densenet121" / "head_only_basic",
        'EfficientNet-B0': PROJECT_ROOT / "kshot" / "results_single_backbone" / dataset_key / "efficientnet_b0" / "head_only_basic",
        'ResNet18': PROJECT_ROOT / "kshot" / "results_single_backbone" / dataset_key / "resnet18" / "head_only_basic",
        'VGG16': PROJECT_ROOT / "kshot" / "results_single_backbone" / dataset_key / "vgg16" / "head_only_basic",
    }
    fusion_sources = {
        'Fusion Baseline': fusion_result_dir("img100_stagea_softmax_linear_resnet_order_v1", dataset_key),
        'Dirichlet Attention': fusion_result_dir("img100_dirichlet_imagenet_1_acc85", dataset_key),
    }

    for model_name, result_dir in {**single_backbone_sources, **fusion_sources}.items():
        for shot in shots:
            summary_path = result_dir / f"{shot}_shot_summary.json"
            if not summary_path.is_file():
                raise FileNotFoundError(f"Missing {dataset_label} summary: {summary_path}")
            rows.append(load_summary_metric_row(summary_path, model_name, shot))
    return pd.DataFrame(rows)


def build_summary_df_or_keep_hardcoded(dataset_key, dataset_label, fallback_df):
    try:
        return build_summary_df(dataset_key, dataset_label)
    except (FileNotFoundError, json.JSONDecodeError) as exc:
        print(f"Keeping hardcoded {dataset_label} rows because kshot summaries are incomplete: {exc}")
        return fallback_df


def build_summary_df_or_none(dataset_key, dataset_label):
    try:
        return build_summary_df(dataset_key, dataset_label)
    except (FileNotFoundError, json.JSONDecodeError) as exc:
        print(f"Skipping {dataset_label} because kshot summaries are incomplete: {exc}")
        return None


deepweeds_df = build_summary_df_or_keep_hardcoded("deepweeds_no_bg_mergeval", "DeepWeeds", deepweeds_df)
aptos_df = build_summary_df_or_keep_hardcoded("aptos2019_split", "APTOS 2019", aptos_df)
leafsugar_df = build_summary_df_or_none("leafsugar_disease_split", "LeafSugar")
plant_seedlings_df = build_summary_df("plant_seedlings_v2_split", "Plant Seedlings V2")
sar_cld_df = build_summary_df_or_none("sar_cld_original_split", "CottonLeaf")
bloodmnist_df = build_summary_df("bloodmnist_224_imagefolder_mergeval", "BloodMNIST")


def add_f1_columns(df):
    precision = df['Precision'].to_numpy(dtype=float)
    recall = df['Recall'].to_numpy(dtype=float)
    precision_err = df['Precision_err'].to_numpy(dtype=float)
    recall_err = df['Recall_err'].to_numpy(dtype=float)

    denom = precision + recall
    f1 = np.divide(
        2.0 * precision * recall,
        denom,
        out=np.zeros_like(denom),
        where=denom != 0,
    )

    # First-order error propagation for F1(P, R) = 2PR / (P + R).
    d_f1_d_precision = np.divide(
        2.0 * recall**2,
        denom**2,
        out=np.zeros_like(denom),
        where=denom != 0,
    )
    d_f1_d_recall = np.divide(
        2.0 * precision**2,
        denom**2,
        out=np.zeros_like(denom),
        where=denom != 0,
    )
    f1_err = np.sqrt(
        (d_f1_d_precision * precision_err) ** 2
        + (d_f1_d_recall * recall_err) ** 2
    )

    df['F1-Score'] = f1
    df['F1-Score_err'] = f1_err
    return df


for metric_df in (
    deepweeds_df,
    plant_seedlings_df,
    bloodmnist_df,
    aptos_df,
):
    if metric_df is None:
        continue
    add_f1_columns(metric_df)


datasets = {
    'DeepWeeds': {
        'models': ['ViT-B16', 'DenseNet121', 'EfficientNet-B0', 'ResNet18', 'VGG16', 'Fusion Baseline', 'Dirichlet Attention'],
        'df': deepweeds_df,
    },
    'Plant Seedlings V2': {
        'models': ['ViT-B16', 'DenseNet121', 'EfficientNet-B0', 'ResNet18', 'VGG16', 'Fusion Baseline', 'Dirichlet Attention'],
        'df': plant_seedlings_df,
    },
    'BloodMNIST': {
        'models': ['ViT-B16', 'DenseNet121', 'EfficientNet-B0', 'ResNet18', 'VGG16', 'Fusion Baseline', 'Dirichlet Attention'],
        'df': bloodmnist_df,
    },
    'APTOS 2019': {
        'models': ['ViT-B16', 'DenseNet121', 'EfficientNet-B0', 'ResNet18', 'VGG16', 'Fusion Baseline', 'Dirichlet Attention'],
        'df': aptos_df,
    },
}
datasets = {name: data for name, data in datasets.items() if data['df'] is not None}

sns.set_theme(style="darkgrid", palette="muted")
metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
STANDARDIZE_Y_AXIS = True
Y_AXIS_MIN = 0.0
Y_AXIS_MAX = 1.0
Y_TICK_STEP = 0.1

model_styles = {
    'Dirichlet Attention': {'color': '#DD8452', 'marker': 's'},
    'ViT-B16': {'color': '#4C72B0', 'marker': 'o'},
    'DenseNet121': {'color': '#55A868', 'marker': '^'},
    'EfficientNet-B0': {'color': '#C44E52', 'marker': 'D'},
    'VGG16': {'color': '#8172B3', 'marker': 'P'},
    'Fusion Baseline': {'color': '#64B5CD', 'marker': 'v'},
    'ResNet18': {'color': '#937860', 'marker': 'X'},
}

full_dataset_best = {}

for vals in full_dataset_best.values():
    precision = vals['Precision']
    recall = vals['Recall']
    vals['F1-Score'] = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)

num_rows = len(datasets)
fig, axes = plt.subplots(num_rows, len(metrics), figsize=(20, 4 * num_rows + 1), sharey=False)
if num_rows == 1:
    axes = np.expand_dims(axes, axis=0)

for row_idx, (dataset_name, data) in enumerate(datasets.items()):
    df = data['df']
    model_list = data['models']
    
    for col_idx, metric in enumerate(metrics):
        ax = axes[row_idx][col_idx]
        
        for model in model_list:
            sub = df[df['Model'] == model].sort_values('Shot')
            x = np.array(sub['Shot'])
            y = sub[metric].values
            err = sub[f'{metric}_err'].values
            style = model_styles.get(model, {'color': '#4C72B0', 'marker': 'o'})
            color = style['color']
            marker = style['marker']
            
            ax.errorbar(x, y, yerr=err, label=model,
                        color=color, marker=marker,
                        linewidth=2, markersize=7, capsize=4, capthick=1.5,
                        linestyle='-', alpha=0.9)

        if dataset_name in full_dataset_best:
            ref_y = full_dataset_best[dataset_name][metric]
            ref_model = full_dataset_best[dataset_name]['best_model']
            ref_color = model_styles.get(ref_model, {'color': '#2F2F2F'})['color']
            ax.axhline(
                y=ref_y,
                color=ref_color,
                linestyle='--',
                linewidth=2.1,
                alpha=0.98,
                zorder=5,
                clip_on=False,
            )
            # Keep a little headroom so dashed full-data references stay visible.
            y_min, y_max = ax.get_ylim()
            ax.set_ylim(y_min, max(y_max, ref_y + 0.03))
        
        if col_idx == 0:
            ax.set_ylabel(dataset_name, fontsize=12, fontweight='bold', labelpad=10)
        if row_idx == 0:
            ax.set_title(metric, fontsize=13, fontweight='bold', pad=10)
        
        ax.set_xlabel('Shots', fontsize=10)
        ax.set_xscale('log')
        ax.set_xticks([1, 5, 10, 20])
        ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
        if STANDARDIZE_Y_AXIS:
            ax.set_ylim(Y_AXIS_MIN, Y_AXIS_MAX)
            ax.set_yticks(np.arange(Y_AXIS_MIN, Y_AXIS_MAX + 1e-9, Y_TICK_STEP))

# Shared legend at bottom (collect unique labels across all subplots)
legend_map = {}
for ax_row in axes:
    for ax in ax_row:
        handles, labels = ax.get_legend_handles_labels()
        for h, l in zip(handles, labels):
            legend_map[l] = h
fig.legend(
    list(legend_map.values()),
    list(legend_map.keys()),
    loc='lower center',
    ncol=4,
    fontsize=11,
    bbox_to_anchor=(0.5, -0.01),
    frameon=True,
    title='K-shot Models',
)

best_handles = {}
for vals in full_dataset_best.values():
    model_name = vals['best_model']
    if model_name in best_handles:
        continue
    model_color = model_styles.get(model_name, {'color': '#2F2F2F'})['color']
    best_handles[model_name] = Line2D(
        [0],
        [0],
        color=model_color,
        linestyle='--',
        linewidth=2.0,
        label=model_name,
    )
if best_handles:
    fig.legend(
        handles=list(best_handles.values()),
        loc='lower center',
        ncol=2,
        fontsize=10,
        bbox_to_anchor=(0.5, -0.085),
        frameon=True,
        title='Full-data Reference (dashed)',
    )

fig.suptitle('Few-Shot Benchmark Results: Macro Fusion Precision/Recall', fontsize=15, fontweight='bold', y=1.01)
plt.tight_layout(rect=[0, 0.12, 1, 0.98])
plt.savefig('benchmark_plots_macro.png', dpi=150, bbox_inches='tight')
plt.close(fig)


selected_dataset_names = (
    'DeepWeeds',
    'Plant Seedlings V2',
    'BloodMNIST',
    'APTOS 2019',
)
selected_datasets = {
    name: datasets[name]
    for name in selected_dataset_names
}

num_rows = len(selected_datasets)
fig, axes = plt.subplots(num_rows, len(metrics), figsize=(20, 4 * num_rows + 1), sharey=False)
if num_rows == 1:
    axes = np.expand_dims(axes, axis=0)

for row_idx, (dataset_name, data) in enumerate(selected_datasets.items()):
    df = data['df']
    model_list = data['models']

    for col_idx, metric in enumerate(metrics):
        ax = axes[row_idx][col_idx]

        for model in model_list:
            sub = df[df['Model'] == model].sort_values('Shot')
            x = np.array(sub['Shot'])
            y = sub[metric].values
            err = sub[f'{metric}_err'].values
            style = model_styles.get(model, {'color': '#4C72B0', 'marker': 'o'})

            ax.errorbar(
                x,
                y,
                yerr=err,
                label=model,
                color=style['color'],
                marker=style['marker'],
                linewidth=2,
                markersize=7,
                capsize=4,
                capthick=1.5,
                linestyle='-',
                alpha=0.9,
            )

        if col_idx == 0:
            ax.set_ylabel(dataset_name, fontsize=12, fontweight='bold', labelpad=10)
        if row_idx == 0:
            ax.set_title(metric, fontsize=13, fontweight='bold', pad=10)

        ax.set_xlabel('Shots', fontsize=10)
        ax.set_xscale('log')
        ax.set_xticks([1, 5, 10, 20])
        ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
        if STANDARDIZE_Y_AXIS:
            ax.set_ylim(Y_AXIS_MIN, Y_AXIS_MAX)
            ax.set_yticks(np.arange(Y_AXIS_MIN, Y_AXIS_MAX + 1e-9, Y_TICK_STEP))

legend_map = {}
for ax_row in axes:
    for ax in ax_row:
        handles, labels = ax.get_legend_handles_labels()
        for h, l in zip(handles, labels):
            legend_map[l] = h

fig.legend(
    list(legend_map.values()),
    list(legend_map.keys()),
    loc='lower center',
    ncol=4,
    fontsize=11,
    bbox_to_anchor=(0.5, -0.01),
    frameon=True,
    title='K-shot Models',
)

fig.suptitle('Few-Shot Benchmark Results: Selected Datasets', fontsize=15, fontweight='bold', y=1.01)
plt.tight_layout(rect=[0, 0.08, 1, 0.98])
plt.savefig('benchmark_plots_macro_selected.png', dpi=150, bbox_inches='tight')
plt.savefig('benchmark_plots_macro_selected.pdf', bbox_inches='tight')
plt.close(fig)
