fsta = "./input/xue.sta"  # 台站文件
ftemp = "./input/temp.pha"  # 模板目录
fdetected = "./input/detected.pha"  # 待分类目录


# =========================================================
# Step 0: 模板目录与待分类目录构建
# 由 sel_temp.py 使用
#
# 说明:
# 这一步是当前工程为了构建模板目录而加入的预处理步骤,
# 不是 Shelly (2016) 文中固定给出的标准流程。
# =========================================================
source_catalog = "./input/source.pha"  # Step 0 读取的总目录
template_min_magnitude = 1.0  # 模板最小震级
template_min_station_count = 8  # 模板最少台站数


# =========================================================
# Step 1: 波形切片与矩阵生成
# 由 cut_waveforms.py 和 lib/cut.py 使用
#
# 说明:
# 下列时间窗和滤波参数属于当前工程实现参数,
# 不是 Shelly (2016) 文中直接给出的固定值。
# =========================================================
data_path = "/home/changning/dhy_pre/processed"  # 连续波形根目录

output_detected = "./output/detected_waveforms_tmp"  # Step 1 detected 波形矩阵输出目录
output_temp = "./output/temp_waveforms_tmp"  # Step 1 template 波形矩阵输出目录

channel_list = ["BHN", "BHE", "BHZ"]  # 参与计算的分量
phase_ids = [0, 1]  # 0=P, 1=S

P_t_before = 0.25  # P 相截窗前长度, 单位 s
P_t_after = 2.5 - P_t_before  # P 相截窗后长度, 单位 s
S_t_before = 0.25  # S 相截窗前长度, 单位 s
S_t_after = 4.0 - S_t_before  # S 相截窗后长度, 单位 s

t_before = [P_t_before, S_t_before]  # 供 phase_id 索引的前窗
t_after = [P_t_after, S_t_after]  # 供 phase_id 索引的后窗

t_before_for_filter = 20  # 滤波前额外多取的前窗, 单位 s
t_after_for_filter = 20  # 滤波前额外多取的后窗, 单位 s
freq_min = 2  # 带通滤波低频, Hz
freq_max = 15  # 带通滤波高频, Hz
sampling_rate = 100  # 目标采样率, Hz


# =========================================================
# Step 2: 相对极性特征构建
# 由 cc_cal.py 使用
#
# 说明:
# 默认采用一条 Shelly (2016) 主线:
# 1. 在允许时滞窗内寻找绝对值最大的相关峰
# 2. 再寻找与主峰相隔一定距离的次强峰
# 3. 用 sign(primary) * (|primary| - |secondary|) 构造相对极性测量
# 4. 对主峰 |CC| 过低的测量置零
# 5. 用 SVD 第一左奇异向量构造事件特征
#
# 其中 |CC| 阈值用于近似 Shelly (2016) 中
# "只有模板及其相关 detected event 才有有效测量" 的约束。
#
# 下列参数不是 Shelly (2016) 文中直接给出的固定值:
# - step2_shelly2016_abs_cc_min
# =========================================================
step2_shelly2016_abs_cc_min = 0.5  # 主峰 |CC| 低于该值时直接置零
step2_shelly2016_min_peak_sep_sec = 0.03  # 主峰和次峰的最小时间间隔, s

step2_lag_max_sec = [0.5, 0.8]  # P/S 相允许搜索的最大时滞, s

step2_template_batch_size = 4  # 一次送入 GPU 的模板条数
step2_enable_cudnn_benchmark = True  # CUDA 下是否启用 cuDNN 自动调优

step2_feature_matrix_out = "combined_RR_all_2col.npy"  # Step 2 主输出


# =========================================================
# Step 3: 聚类与可视化
# 由 cluster.py 使用
#
# 说明:
# 当前主干尽量贴近 Shelly (2016):
# 事件极性向量 + cosine distance + hierarchical clustering
#
# 但下列参数属于实现选择或展示参数,
# 不是 Shelly (2016) 文中已在本工程中被唯一确认的固定值:
# - step3_linkage_method
# - step3_cluster_cut_mode
# - step3_distance_cut_threshold
# - step3_maxclust_n_clusters
# - step3_top_n_show
# - step3_export_html
# =========================================================
step3_matplotlib_backend = None  # 如需强制无头绘图可设为 "Agg"
step3_mplconfig_dir = ".mplconfig"  # Matplotlib 缓存目录

step3_linkage_method = "average"  # 层次聚类 linkage 方法
step3_cluster_cut_mode = "maxclust"  # Shelly (2016) 文中示例切到 100 个 clusters

step3_distance_cut_threshold = 0.40  # distance 模式切树阈值
step3_maxclust_n_clusters = 100  # Shelly (2016) 示例中的总簇数

step3_top_n_show = 5  # 图件和 HTML 重点展示的最大簇数
step3_show_noise = False  # 静态图中是否显示 labels=0 的事件
step3_save_fig = True  # 是否保存静态图
step3_fig_out = "cluster_map_profiles.png"  # 静态图输出路径
step3_point_size = 5  # 簇内事件点大小
step3_bg_point_size = 2.5  # 背景全部事件点大小

step3_export_html = True  # 是否导出 HTML 时间滑条
step3_html_out = "detected_3d_slider_by_cluster.html"  # HTML 输出路径
step3_html_marker_size = 1.2  # HTML 基础点大小
step3_html_slider_nframes = 120  # 时间滑条帧数
step3_other_cluster_color = "#B0B0B0"  # 非重点簇的颜色
step3_mag_highlight = 3.0  # 大震级事件开始放大的阈值
step3_mag_highlight_mult = 4.0  # 大震级事件的放大倍数
step3_future_alpha = 0.20  # HTML 中未来事件的透明度
step3_past_alpha = 0.90  # HTML 中已发生事件的透明度
