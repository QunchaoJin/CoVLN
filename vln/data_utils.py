import os
import json
import random
from collections import defaultdict


def load_instr_datasets(anno_dir, dataset, splits, tokenizer, is_test=True):
    data = []
    for split in splits:
        if 'sample' in split:
            filepath = os.path.join(anno_dir, split)
            with open(filepath) as f:
                new_data = json.load(f)

        elif "/" not in split:    # the official splits
            if tokenizer == 'bert':
                filepath = os.path.join(anno_dir, '%s_%s_enc.json' % (dataset.upper(), split))
            elif tokenizer == 'xlm':
                filepath = os.path.join(anno_dir, '%s_%s_enc_xlmr.json' % (dataset.upper(), split))
            else:
                raise NotImplementedError('unsupported tokenizer %s' % tokenizer)

            with open(filepath) as f:
                new_data = json.load(f)

            if split == 'val_train_seen':
                new_data = new_data[:50]

        # Join
        data += new_data
    return data


def pair_episodes(data, pairing='prior', seed=22):
    """Split a flat episode list into two aligned lists (main agent, aux agent).

    Episodes at the same index share a scan and navigate concurrently.
    pairing='prior'  -> two_list_aux: maximise ground-truth path overlap within a scan.
    pairing='random' -> random_multi_list_split: random pairing within a scan.
    """
    if pairing == 'prior':
        data1, data2 = two_list_aux(data, seed=seed)
    elif pairing == 'random':
        data1, data2 = random_multi_list_split(data, n_splits=2, seed=seed)
    else:
        raise ValueError('unknown pairing strategy %s' % pairing)
    assert len(data1) == len(data2)
    return data1, data2


def construct_instrs(anno_dir, dataset, splits, tokenizer, max_instr_len=512, is_test=True, pairing='prior'):
    data = []
    for i, item in enumerate(load_instr_datasets(anno_dir, dataset, splits, tokenizer, is_test=is_test)):
        # Split multiple instructions into separate entries
        for j, instr in enumerate(item['instructions']):
            new_item = dict(item)
            new_item['instr_id'] = '%s_%d' % (item['path_id'], j)
            new_item['instruction'] = instr
            new_item['instr_encoding'] = item['instr_encodings'][j][:max_instr_len]
            del new_item['instructions']
            del new_item['instr_encodings']
            data.append(new_item)

    return pair_episodes(data, pairing=pairing)


def load_obj2vps(bbox_file):
    obj2vps = {}
    bbox_data = json.load(open(bbox_file))
    for scanvp, value in bbox_data.items():
        scan, vp = scanvp.split('_')
        for objid, objinfo in value.items():
            if objinfo['visible_pos']:
                obj2vps.setdefault(scan+'_'+objid, [])
                obj2vps[scan+'_'+objid].append(vp)
    return obj2vps


def two_list_aux(ori_list, seed=22):
    """
    Prior-based pairing. Split the episode list into two equal-length lists such that:
      1. episodes at the same index belong to the same scan;
      2. they have different path_id and different start viewpoint;
      3. among valid candidates, the one whose ground-truth path overlaps most
         with the anchor episode is chosen.
    An episode that cannot be paired (odd count, or no valid candidate left)
    is placed at the same index in both lists (self-pair).

    Args:
        ori_list (list): list of episode dicts.
        seed (int): random seed for the in-scan shuffle, for reproducibility.

    Returns:
        tuple: (list_1, list_2) of equal length.
    """
    # 1. fix the seed
    random.seed(seed)

    # 2. group episodes by scan: {scan_id: [item1, item2, ...]}
    scan_groups = {}
    for item in ori_list:
        scan_id = item['scan']
        if scan_id not in scan_groups:
            scan_groups[scan_id] = []
        scan_groups[scan_id].append(item)

    list_1 = []
    list_2 = []

    # sort scan keys so the result is identical across platforms / runs
    sorted_scan_ids = sorted(scan_groups.keys())

    for scan in sorted_scan_ids:
        items = scan_groups[scan]

        # shuffle within the scan
        random.shuffle(items)

        # pair up
        while items:
            # take the first element as the anchor for list_1
            item_a = items.pop(0)

            # nothing left in this scan (odd count): self-pair
            if not items:
                list_1.append(item_a)
                list_2.append(item_a)
                break

            # find the best partner item_b among the remaining items
            best_match_idx = -1
            max_overlap = -1
            candidates = []

            for i, item_b in enumerate(items):
                # condition 2: different path_id and different start viewpoint
                if item_a['path_id'] != item_b['path_id'] and item_a['path'][0] != item_b['path'][0]:
                    # condition 3: prefer the largest ground-truth path overlap
                    set_a = set(item_a['path'])
                    set_b = set(item_b['path'])
                    overlap = len(set_a.intersection(set_b))
                    candidates.append((i, overlap))

            if candidates:
                # sort by overlap (descending) and take the best
                candidates.sort(key=lambda x: x[1], reverse=True)
                best_match_idx = candidates[0][0]

                item_b = items.pop(best_match_idx)

                list_1.append(item_a)
                list_2.append(item_b)
            else:
                # no valid partner (all remaining episodes share path_id / start
                # with item_a): fall back to a self-pair
                list_1.append(item_a)
                list_2.append(item_a)

    return list_1, list_2


def random_multi_list_split(ori_list, n_splits=3, seed=22):
    """
    Random pairing. Split the episode list into n_splits equal-length lists such that:
      1. episodes at the same index belong to the same scan;
      2. episodes at the same index have different path_id where possible;
      3. leftover episodes (count not divisible by n_splits) are repeated in all lists.
    """
    if not ori_list:
        return [[] for _ in range(n_splits)]

    random.seed(seed)

    # 1. group by scan
    scan_groups = defaultdict(list)
    for item in ori_list:
        scan_groups[item.get('scan')].append(item)

    result_tuples = []  # (item_1, item_2, ..., item_n)
    leftovers = []      # episodes in a scan that could not fill a full group

    # 2. build groups within each scan
    scan_keys = list(scan_groups.keys())
    random.shuffle(scan_keys)

    for scan in scan_keys:
        items = scan_groups[scan]
        random.shuffle(items)

        while len(items) >= n_splits:
            # try to pick n_splits episodes with distinct path_id
            path_map = defaultdict(list)
            for itm in items:
                path_map[itm.get('path_id')].append(itm)

            if len(path_map) >= n_splits:
                selected_group = []
                distinct_paths = list(path_map.keys())[:n_splits]
                for p_id in distinct_paths:
                    # remove one episode with this path_id from items
                    for idx, itm in enumerate(items):
                        if itm.get('path_id') == p_id:
                            selected_group.append(items.pop(idx))
                            break
                result_tuples.append(tuple(selected_group))
            else:
                # not enough distinct paths left in this scan: take the first
                # n_splits episodes regardless of path_id
                fallback_group = [items.pop(0) for _ in range(n_splits)]
                result_tuples.append(tuple(fallback_group))

        # fewer than n_splits episodes remain in this scan
        leftovers.extend(items)

    # 3. group cross-scan leftovers so that all lists have the same length.
    #    Note: these groups break the same-scan guarantee.
    while len(leftovers) >= n_splits:
        result_tuples.append(tuple(leftovers[:n_splits]))
        leftovers = leftovers[n_splits:]

    # 4. shuffle the groups globally
    random.shuffle(result_tuples)

    # 5. unzip into n lists
    final_lists = [list(lst) for lst in zip(*result_tuples)]

    # 6. remaining leftovers are repeated in every list (condition 3)
    if leftovers:
        for extra in leftovers:
            for i in range(n_splits):
                final_lists[i].append(extra)

    return final_lists
