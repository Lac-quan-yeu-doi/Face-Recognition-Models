DATASET_PATH = "/root/dfs/callmePhineas/DACN/dataset"
WORKING_PATH = "/root/dfs/callmePhineas/DACN/working"

# DATASET_PATH = "/kvm/callmePhineas/DACN/dataset"
# WORKING_PATH = "/kvm/callmePhineas/DACN/result"


DATASET_PATH = "/home/phatvo/callmePhineas/DACN/working/dataset"
WORKING_PATH = "/home/phatvo/callmePhineas/DACN/working/result"

BACKBONE = 'resnet50'

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

# MV-Softmax
M_mv        = 0.35
WEIGHT_mv   = 1.12
S_mv        = 32.0
MARGIN_TYPE_mv = 'am' # 'arc' for MV-Arc; 'am' for MV-Cos

# CurricularFace
M_curricular = 0.5
S_curricular = 64.0
MOMENTUM_curricular = 0.01

# VPL-ArcFace
S_vpl           = 64.0
M_vpl           = 0.50
EASY_MARGIN_vpl = True
LAMDA_vpl       = 0.15
DELTA_vpl       = 100

# AdaFace
S_ada        = 64.0
M_ada        = 0.4
H_ada        = 0.333
T_ALPHA_ada  = 0.99

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
S_mag           = 64.0
EASY_MARGIN_mag = True
L_MARGIN_mag    = 0.40
U_MARGIN_mag    = 0.8
L_A_mag         = 10.0
U_A_mag         = 110.0

# SphereFace2
LAMBDA_sf2 = 0.7          # Balance between positive/negative pairs (λ)
R_sf2 = 40.0             # Scale factor for logits
M_sf2 = 0.4              # Angular margin
T_sf2 = 3.0              # Power for g(cos θ) transformation
LW_sf2 = 50.0            # Loss weight multiplier

# UniFace
M_uniface = 0.4          # Margin for positive samples
S_uniface = 64.0         # Scale factor (temperature)
L_uniface = 1.0          # Weight for negative loss (lambda/balance)
R_uniface = 1.0          # Bias initialization multiplier

# UnitFace
M_units = 0.4              # margin for normalized softmax
S_units = 64               # scale
L_units = 1.0            # λ for combining the two losses
R_units = 1.0            # regularization or balancing

# QAFace
S_qa            = 64.0
M_qa            = 0.50
EASY_MARGIN_qa  = True
DELTA_qa        = 1000      # memory lifetime (steps)
TTO_qa          = 2.0       # threshold for injection (in std space)
ALPHA_qa        = 0.99      # EMA coefficient for magnitude stats
GAMMA_qa        = 0.99      # momentum coefficient for mbackbone

# QMagFace
S_qmag            = 64.0
EASY_MARGIN_qmag  = True
L_MARGIN_qmag     = 0.45
U_MARGIN_qmag     = 0.80
L_A_qmag           = 10.0
U_A_qmag           = 110.0
ALPHA_18_qmag     = 0.092861
BETA_18_qmag      = 0.135311
ALPHA_50_qmag     = 0.065984
BETA_50_qmag      = 0.103799
ALPHA_100_qmag = 0.077428
BETA_100_qmag     = 0.125926



