import os
from collections import OrderedDict

import numpy as np
from obspy.core import UTCDateTime, read

import config


def read_fsta(fsta):
    with open(fsta, encoding="utf-8") as f:
        lines = f.readlines()

    sta_dict = {}
    for line in lines:
        codes = line.split(",")
        net_sta = codes[0]
        lat, lon, ele = [float(code) for code in codes[1:4]]
        sta_dict[net_sta] = [lon, lat, ele]
    return sta_dict


def read_fpha_list(fpha):
    with open(fpha, encoding="utf-8") as f:
        flist = f.readlines()

    event_dic = {}
    for line in flist:
        if line.startswith("20"):
            time, lat, lon, dep, mag, evid = line.strip().split(",")
            event_dic[evid] = [
                [UTCDateTime(time), float(lat), float(lon), float(dep), float(mag), evid],
                {},
            ]
            continue

        net_sta, tp, ts, *_ = line.strip().split(",")
        tp = UTCDateTime(tp) if tp != "-1" else -1
        ts = UTCDateTime(ts) if ts != "-1" else -1
        event_dic[evid][1][net_sta] = [tp, ts]

    return event_dic


def read_pha_simplified(fpha):
    with open(fpha, encoding="utf-8") as f:
        flist = f.readlines()

    event_dic = {}
    for line in flist:
        if line.startswith("20"):
            _, _, _, _, _, evid = line.strip().split(",")
            event_dic[evid] = [line]
            continue
        event_dic[evid].append(line)
    return event_dic


def iter_event_waveform_jobs(sta_names):
    for sta_name in sta_names:
        for channel in config.channel_list:
            for phase_id in config.phase_ids:
                yield sta_name, channel, int(phase_id)


def build_waveform_groups(event_dic, event_ids, data_path, sta_name, channel, phase_id):
    groups = OrderedDict()

    for row, evid in enumerate(event_ids):
        phase_dic = event_dic[evid][1]
        sta_phase = phase_dic.get(sta_name)
        if sta_phase is None:
            continue

        phase_time = sta_phase[phase_id]
        if phase_time == -1:
            continue

        data_dir = phase_time.strftime("%Y%m%d")
        mseed_name = f"{sta_name}.{data_dir}.{channel}.mseed"
        waveform_path = os.path.join(data_path, data_dir, mseed_name)
        s_time = sta_phase[1] if phase_id == 0 else -1

        groups.setdefault(waveform_path, []).append(
            {
                "row": row,
                "evid": evid,
                "phase_time": phase_time,
                "s_time": s_time,
            }
        )

    return groups


def load_waveforms(data_path, event_dic, event_ids, sta_name="XX.C038", channel="BHZ", phase_id=1):
    t_before = config.t_before[phase_id]
    t_after = config.t_after[phase_id]
    width = int(config.sampling_rate * (t_before + t_after)) + 1

    mat = np.zeros((len(event_ids), width), dtype=np.float32)
    have_data = 0
    waveform_groups = build_waveform_groups(
        event_dic=event_dic,
        event_ids=event_ids,
        data_path=data_path,
        sta_name=sta_name,
        channel=channel,
        phase_id=phase_id,
    )

    for waveform_path, records in waveform_groups.items():
        try:
            cached_stream = read(waveform_path)
        except Exception as e:
            print(
                f"[load_waveforms] path={waveform_path} n_records={len(records)} "
                f"err={type(e).__name__}: {e}"
            )
            continue

        for record in records:
            row = record["row"]
            evid = record["evid"]
            phase_time = record["phase_time"]
            s_time = record["s_time"]
            try:
                start_time = phase_time - config.t_before_for_filter
                end_time = phase_time + config.t_after_for_filter
                st_tmp = cached_stream.slice(starttime=start_time, endtime=end_time).copy()
                if len(st_tmp) == 0:
                    continue
                st_tmp.merge(method=1, fill_value=0.0)
                st_tmp.detrend("demean")
                st_tmp.detrend("linear")
                st_tmp.taper(max_percentage=0.05, type="cosine")
                st_tmp.filter(
                    "bandpass",
                    freqmin=config.freq_min,
                    freqmax=config.freq_max,
                    corners=2,
                    zerophase=False,
                )

                if phase_id == 0:
                    if s_time != -1:
                        ps_delta = s_time - phase_time - t_before
                        if ps_delta < t_after:
                            st_tmp = st_tmp.trim(
                                starttime=phase_time - t_before,
                                endtime=phase_time + ps_delta,
                            )
                        else:
                            st_tmp = st_tmp.trim(
                                starttime=phase_time - t_before,
                                endtime=phase_time + t_after,
                            )
                    else:
                        st_tmp = st_tmp.trim(
                            starttime=phase_time - t_before,
                            endtime=phase_time + t_after,
                        )
                elif phase_id == 1:
                    st_tmp = st_tmp.trim(
                        starttime=phase_time - t_before,
                        endtime=phase_time + t_after,
                    )
                else:
                    raise ValueError(f"Unsupported phase id: {phase_id}")

                data = st_tmp[0].data
                if data.size < width:
                    print(f"warning: data truncated,{data.size}/{width}")
                    print(f"evid {evid} at sta {sta_name} chn {channel}")
                    data = np.pad(
                        data,
                        (0, width - data.size),
                        mode="constant",
                        constant_values=0.0,
                    )
                else:
                    data = data[:width]

                mat[row, :] = np.asarray(data, dtype=np.float32)
                have_data += 1
            except Exception as e:
                print(
                    f"[load_waveforms] evid={evid} sta={sta_name} ch={channel} "
                    f"err={type(e).__name__}: {e}"
                )

    if have_data == 0:
        print(f"{sta_name}_{channel}_{phase_id} warning! data not found")
    return mat
