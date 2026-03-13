import os

import config
from lib.cut import read_fpha_list, read_pha_simplified


# =========================================================
# 说明:
# 这一步是当前工程的模板/待分类目录构建步骤,
# 不属于 Shelly (2016) 文中固定给出的标准处理流程。
# 相关阈值定义在 config.py 开头。
# =========================================================
def select_template_event_ids(event_dic, min_magnitude, min_station_count):
    template_ids = []
    for evid, (event_loc, phase_dic) in event_dic.items():
        mag = event_loc[4]
        sta_num = len(phase_dic)
        if mag >= min_magnitude and sta_num >= min_station_count:
            template_ids.append(evid)
    return set(template_ids)


def write_template_and_detected_catalogs(event_lines, template_ids, template_out, detected_out):
    with open(template_out, "w", encoding="utf-8") as ftemp, open(
        detected_out, "w", encoding="utf-8"
    ) as fdetected:
        for evid, lines in event_lines.items():
            for line in lines:
                fdetected.write(line)
                if evid in template_ids:
                    ftemp.write(line)


def main():
    if not os.path.exists(config.source_catalog):
        raise FileNotFoundError(
            f"source catalog not found: {config.source_catalog}\n"
            "请先把待筛选的总目录放到 config.source_catalog。"
        )

    event_dic = read_fpha_list(config.source_catalog)
    event_lines = read_pha_simplified(config.source_catalog)

    template_ids = select_template_event_ids(
        event_dic=event_dic,
        min_magnitude=config.template_min_magnitude,
        min_station_count=config.template_min_station_count,
    )

    write_template_and_detected_catalogs(
        event_lines=event_lines,
        template_ids=template_ids,
        template_out=config.ftemp,
        detected_out=config.fdetected,
    )

    print(f"[INFO] Source events         : {len(event_dic)}")
    print(f"[INFO] Selected templates    : {len(template_ids)}")
    print(f"[INFO] Template output       : {config.ftemp}")
    print(f"[INFO] Detected output       : {config.fdetected}")
    print(f"[INFO] Template mag cutoff   : {config.template_min_magnitude}")
    print(f"[INFO] Template sta cutoff   : {config.template_min_station_count}")


if __name__ == "__main__":
    main()
