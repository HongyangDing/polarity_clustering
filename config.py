fsta = "./input/xue.sta"
ftemp = "./input/temp.pha"
fdetected = "./input/detected.pha"

# =========================================================
# 项目级配置说明
# 下列目录筛选阈值和波形窗口配置用于当前工程主线, 不是 Shelly (2016) 文中明确给出的参数:
# - template_min_magnitude
# - template_min_station_count
# - P_t_before / P_t_after / S_t_before / S_t_after
# - freq_min / freq_max
# =========================================================
# Optional source catalog used by step0 event/template selection.
source_catalog = "./input/source.pha"
template_min_magnitude = 1.0
template_min_station_count = 8

data_path = "/home/changning/dhy_pre/processed"

output_detected = "./output/detected_waveforms_tmp"
output_temp = "./output/temp_waveforms_tmp"

channel_list = ["BHN", "BHE", "BHZ"]
phase_ids = [0, 1]

P_t_before = 0.25  # s
P_t_after = 2.5 - P_t_before  # s
S_t_before = 0.25  # s
S_t_after = 4.0 - S_t_before  # s

t_before = [P_t_before, S_t_before]
t_after = [P_t_after, S_t_after]

t_before_for_filter = 20  # s
t_after_for_filter = 20  # s
freq_min = 2
freq_max = 15
sampling_rate = 100
