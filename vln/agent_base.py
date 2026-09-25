import json
import os
import time
from utils.logger import write_to_record_file


class BaseAgent(object):
    ''' Base class for an agent to generate and save trajectories. '''

    def __init__(self, env, env_au):
        self.env = env
        self.env_au = env_au
        self.results = {}

    def get_results(self, detailed_output=False):
        output = []
        for k, v in self.results.items():
            output.append({'instr_id': k, 'trajectory': v['path'], 'a_t': v['a_t']})
            if detailed_output:
                output[-1]['details'] = v['details']
        return output
    
    def get_results_aux(self, detailed_output=False):
        output = []
        for k, v in self.results_aux.items():
            output.append({'instr_id': k, 'trajectory': v['path'], 'a_t': v['a_t']})
            if detailed_output:
                output[-1]['details'] = v['details']
        return output

    def rollout(self, **args):
        raise NotImplementedError

    def test(self, iters=None, args=None, **kwargs):
        self.env.reset_epoch(shuffle=(iters is not None))   # If iters is not none, shuffle the env batch
        self.env_au.reset_epoch(shuffle=(iters is not None))   # If iters is not none, shuffle the env batch
        
        self.results = {}
        self.results_aux = {}
        looped = False

        # The epoch is finished once a (main, aux) pair repeats. Keying on the pair
        # rather than on the main instr_id alone is required for self-pairs, where
        # main and aux share the same instr_id.
        seen_pairs = set()
        while True:
            trajs, traj_auxs = self.rollout(**kwargs)
            pair_key = (trajs[0]['instr_id'], traj_auxs[0]['instr_id'])
            if pair_key in seen_pairs:
                looped = True
            else:
                seen_pairs.add(pair_key)
                for traj in trajs:
                    self.results[traj['instr_id']] = traj
                for traj_aux in traj_auxs:
                    self.results_aux[traj_aux['instr_id']] = traj_aux

            if looped:
                break

            preds = self.get_results(detailed_output=args.detailed_output)
            preds_aux = self.get_results_aux(detailed_output=args.detailed_output)
            current_pred = [preds[-1]]
            current_pred_aux = [preds_aux[-1]]

            # evaluating current case
            score_summary, current_metrics = self.env.eval_metrics(current_pred, args.dataset)
            score_summary_aux, current_metrics_aux = self.env_au.eval_metrics(current_pred_aux, args.dataset)   

            loss_str = "Current case  -"
            loss_str_aux = "Current case  aux-"
            for metric, val in score_summary.items():
                loss_str += '  %s: %.2f' % (metric, val)
            print(loss_str)

            for metric, val in score_summary_aux.items():
                loss_str_aux += '  %s: %.2f' % (metric, val)
            print(loss_str_aux)

            # add evaluation result
            instr_id = preds[-1]['instr_id']
            scan, gt_traj = self.env.gt_trajs[instr_id]

            preds[-1]['scan'] = scan
            preds[-1]['gt_traj'] = gt_traj
            preds[-1]['evaluation'] = current_metrics

            instr_id_aux = preds_aux[-1]['instr_id']
            scan, gt_traj = self.env_au.gt_trajs[instr_id_aux]
            preds_aux[-1]['scan'] = scan
            preds_aux[-1]['gt_traj'] = gt_traj
            preds_aux[-1]['evaluation'] = current_metrics_aux

            if args.save_pred:
                json.dump(
                    preds[-1],
                    open(os.path.join(args.pred_dir, "case_InstrID_%s.json" % instr_id), 'w'),
                    sort_keys=True, indent=4, separators=(',', ': ')
                )

                json.dump(
                    preds_aux[-1],
                    open(os.path.join(args.pred_dir, "case_aux_InstrID_%s.json" % instr_id_aux), 'w'),
                    sort_keys=True, indent=4, separators=(',', ': ')
                )

        # evaluating all cases
        score_summary, _ = self.env.eval_metrics(preds, args.dataset)
        score_summary_aux, _ = self.env_au.eval_metrics(preds_aux, args.dataset)

        loss_str = "All cases  -"
        loss_str_aux = "All cases  -"
        for metric, val in score_summary.items():
            loss_str += '  %s: %.2f' % (metric, val)

        for metric, val in score_summary_aux.items():
            loss_str_aux += '  %s: %.2f' % (metric, val)

        record_file = os.path.join(args.log_dir, 'valid.txt')
        write_to_record_file(loss_str + '\n', record_file)
        write_to_record_file(loss_str_aux + '\n', record_file)





