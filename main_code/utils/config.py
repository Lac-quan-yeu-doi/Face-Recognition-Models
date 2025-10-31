DATASET_PATH = "/root/dfs/callmePhineas/DACN/dataset"
WORKING_PATH = "/root/dfs/callmePhineas/DACN/working"

# DATASET_PATH = "/kvm/callmePhineas/DACN/dataset"
# WORKING_PATH = "/kvm/callmePhineas/DACN/result"


DATASET_PATH = "/home/phatvo/callmePhineas/DACN/working/dataset"
WORKING_PATH = "/home/phatvo/callmePhineas/DACN/working/result"


FEATURE_DIM = 512
LAMBDA_G = 0.0

# SphereFace
M_sphere = 2
S_sphere = 20.0

# CosFace
M_cos = 0.35
S_cos = 64.0

# ArcFace
M_arc = 0.5
S_arc = 64.0

# CurricularFace
M_curricular = 0.5
S_curricular = 64.0
MOMENTUM_curricular = 0.01

# MV-Softmax
M_mv        = 0.35
WEIGHT_mv   = 1.12
S_mv        = 32.0
MARGIN_TYPE_mv = 'am' # 'arc' for MV-Arc; 'am' for MV-Cos

# AdaFace
S_ada        = 64.0
M_ada        = 0.4
H_ada        = 0.333
T_ALPHA_ada  = 1.0

# ElasticArcFace
S_elastic_arc   = 64.0
M_elastic_arc   = 0.50
STD_elastic_arc = 0.0125
PLUS_elastic_arc = False

# ElasticCosFace
S_elastic_cos   = 64.0
M_elastic_cos   = 0.35
STD_elastic_cos = 0.0125
PLUS_elastic_cos = False

# MagFace
S_mag = 64.0
M_l_mag = 0.45
M_u_mag = 0.8
A_l_mag = 10.0
A_u_mag = 110.0
LAMBDA_g_mag = 35.0


