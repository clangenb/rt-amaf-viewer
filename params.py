SMILExtract = '../opensmile-2.3.0/inst/bin/SMILExtract'
smile_config = 'smileconfig/live_ComParE_2016_reduced.conf'

config_str_a = 'run2_A_size300_step100_bs16_sl10_nl2_ss80_no10_std_0.9'
config_str_v = 'V_size300_step100_bs16_sl30_nl2_ss80_no10'

model_a = 'golden_models/{}/model.ckpt-{}'.format(config_str_a, 60)
model_v = 'golden_models/{}/model.ckpt-{}'.format(config_str_v, 80)