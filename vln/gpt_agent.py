import sys
import numpy as np
from collections import defaultdict
from GPT.one_stage_prompt_manager import OneStagePromptManager
from .agent_base import BaseAgent
from GPT.api import gpt_infer
import json
import os
from utils.logger import write_to_record_file


class GPTNavAgent(BaseAgent):
    env_actions = {
        'left': (0, -1, 0),  # left
        'right': (0, 1, 0),  # right
        'up': (0, 0, 1),  # up
        'down': (0, 0, -1),  # down
        'forward': (1, 0, 0),  # forward
        '<end>': (0, 0, 0),  # <end>
        '<start>': (0, 0, 0),  # <start>
        '<ignore>': (0, 0, 0)  # <ignore>
    }
    for k, v in env_actions.items():
        env_actions[k] = [[vx] for vx in v]

    def __init__(self, args, env, env_au, rank=0):
        super().__init__(env, env_au)
        self.args = args

        self._build_prompt_manager()

        self.record_file = os.path.join(args.log_dir, 'process.txt')

        # Logs
        sys.stdout.flush()
        self.logs = defaultdict(list)
    
    def _build_prompt_manager(self):
        self.prompt_manager = OneStagePromptManager(self.args)
        self.prompt_manager_aux = OneStagePromptManager(self.args)
        print('Model version:', self.args.llm)

    def make_equiv_action(self, a_t, obs, traj=None, mode='primary'):

        def take_action(i, name):
            if type(name) is int:       # Go to the next viewpoint
                if mode == 'primary':
                    self.env.env.sims[i].makeAction([name], [0], [0])
                elif mode == 'auxiliary':
                    self.env_au.env.sims[i].makeAction([name], [0], [0])
            else:                       # Adjust
                if mode == 'primary':
                    self.env.env.sims[i].makeAction(*self.env_actions[name])
                elif mode == 'auxiliary':
                    self.env_au.env.sims[i].makeAction(*self.env_actions[name])

        for i, ob in enumerate(obs):
            action = a_t[i]
            if action != -1:            # -1 is the <stop> action
                select_candidate = ob['candidate'][action]
                src_point = ob['viewIndex']
                trg_point = select_candidate['pointId']
                src_level = (src_point ) // 12  # The point idx started from 0
                trg_level = (trg_point ) // 12
                while src_level < trg_level:    # Tune up
                    take_action(i, 'up')
                    src_level += 1
                while src_level > trg_level:    # Tune down
                    take_action(i, 'down')
                    src_level -= 1
                
                if mode == 'primary':
                    while self.env.env.sims[i].getState()[0].viewIndex != trg_point:    # Turn right until the target
                        take_action(i, 'right')
                elif mode == 'auxiliary':
                    while self.env_au.env.sims[i].getState()[0].viewIndex != trg_point:    # Turn right until the target
                        take_action(i, 'right')
                if mode == 'primary':
                    assert select_candidate['viewpointId'] == \
                           self.env.env.sims[i].getState()[0].navigableLocations[select_candidate['idx']].viewpointId
                elif mode == 'auxiliary':
                    assert select_candidate['viewpointId'] == \
                           self.env_au.env.sims[i].getState()[0].navigableLocations[select_candidate['idx']].viewpointId
                take_action(i, select_candidate['idx']) # j+1: idx for navigable location

                if mode == 'primary':
                    state = self.env.env.sims[i].getState()[0]
                elif mode == 'auxiliary':
                    state = self.env_au.env.sims[i].getState()[0]

                if traj is not None:
                    traj[i]['path'].append([state.location.viewpointId])
     

    def rollout(self, train_ml=None, train_rl=False, reset=True):
        if reset:  # Reset env
            obs = self.env.reset()
            obs_aux = self.env_au.reset()
        else:
            obs = self.env._get_obs()
            obs_aux = self.env_au._get_obs()

        batch_size = len(obs)

        # Record the navigation path
        traj = [{
            'instr_id': ob['instr_id'],
            'path': [[ob['viewpoint']]],
            'details': {},
            'a_t': {},
        } for ob in obs]

        traj_aux = [{
            'instr_id': ob['instr_id'],
            'path': [[ob['viewpoint']]],
            'details': {},
            'a_t': {},
        } for ob in obs_aux]

        # Initialization the tracking state
        ended = np.array([False] * batch_size)
        just_ended = np.array([False] * batch_size)

        ended_aux = np.array([False] * batch_size)
        just_ended_aux = np.array([False] * batch_size)

        previous_angle = [{'heading': ob['heading'],
                               'elevation': ob['elevation']} for ob in obs]
        previous_angle_aux = [{'heading': ob['heading'],
                               'elevation': ob['elevation']} for ob in obs_aux]

        self.prompt_manager.history = ['' for _ in range(self.args.batch_size)]
        self.prompt_manager.nodes_list = [[] for _ in range(self.args.batch_size)]
        self.prompt_manager.node_imgs = [[] for _ in range(self.args.batch_size)]
        self.prompt_manager.graph = [{} for _ in range(self.args.batch_size)]
        self.prompt_manager.trajectory = [[] for _ in range(self.args.batch_size)]
        self.prompt_manager.planning = [["Navigation has just started, with no planning yet."] for _ in range(self.args.batch_size)]

        self.prompt_manager_aux.history = ['' for _ in range(self.args.batch_size)]
        self.prompt_manager_aux.nodes_list = [[] for _ in range(self.args.batch_size)]
        self.prompt_manager_aux.node_imgs = [[] for _ in range(self.args.batch_size)]
        self.prompt_manager_aux.graph = [{} for _ in range(self.args.batch_size)]
        self.prompt_manager_aux.trajectory = [[] for _ in range(self.args.batch_size)]
        self.prompt_manager_aux.planning = [["Navigation has just started, with no planning yet."] for _ in range(self.args.batch_size)]

        for t in range(self.args.max_action_len):
            main_node_list = self.prompt_manager.nodes_list[0]
            aux_node_list = self.prompt_manager_aux.nodes_list[0]

            # Vision-sharing is enabled once the two agents' maps share a common node.
            if set(main_node_list) & set(aux_node_list):
                self.prompt_manager.merge_flag = True
                self.prompt_manager_aux.merge_flag = True

            cand_inputs = self.prompt_manager.make_action_prompt(obs, previous_angle)
            cand_inputs_aux = self.prompt_manager_aux.make_action_prompt(obs_aux, previous_angle_aux)

            nav_input = self.prompt_manager.make_r2r_json_prompts(cand_inputs=cand_inputs, obs=obs, t=t, other_prompt=self.prompt_manager_aux)
            nav_input_aux = self.prompt_manager_aux.make_r2r_json_prompts(cand_inputs=cand_inputs_aux, obs=obs_aux, t=t, other_prompt=self.prompt_manager)

            image_list = self.prompt_manager.node_imgs[0]
            image_list_aux = self.prompt_manager_aux.node_imgs[0]

            environment_prompts = nav_input["prompts"][0]
            environment_prompts_aux = nav_input_aux["prompts"][0]

            write_to_record_file('-------------------- Main Environment Prompts --------------------\n', self.record_file)
            write_to_record_file(environment_prompts, self.record_file)

            write_to_record_file('-------------------- Main other_index --------------------\n', self.record_file)
            write_to_record_file(str(nav_input["other_index"]), self.record_file)

            ## feed into mllm
            if np.all(ended):
                a_t = [0] * batch_size
                write_to_record_file('Main_Already stopped, skip inference.', self.record_file)
            elif len(image_list) > 40:
                a_t = [0]
                write_to_record_file('Main_Exceed image limit and stop!', self.record_file)
            else:
                json_output = None
                nav_output = ""
                for attempt in range(5):
                    nav_output, tokens = gpt_infer(nav_input["task_description"], nav_input["other_index"], environment_prompts, image_list, image_list_aux,
                                                    self.args.llm, self.args.max_tokens, response_format={"type": "json_object"})
                    try:
                        json_output = json.loads(nav_output)
                        break
                    except json.JSONDecodeError:
                        write_to_record_file(f'Main_Invalid JSON response, retry {attempt + 1}.\n', self.record_file)

                if json_output is None:
                    write_to_record_file('Main_Invalid JSON response, use default.\n', self.record_file)
                    only_options = nav_input["only_options"][0]
                    default_action = only_options[0] if len(only_options) > 0 else "A"
                    json_output = {
                        "Thought": "",
                        "New Planning": "No plans currently.",
                        "Action": default_action
                    }

                a_t = self.prompt_manager.parse_json_action(json_output, nav_input["only_options"], t)
                self.prompt_manager.parse_json_planning(json_output)

                write_to_record_file('-------------------- Main Output --------------------\n', self.record_file)
                write_to_record_file(nav_output, self.record_file)

            if np.any(ended):
                a_t = [0 if ended[i] else a_t[i] for i in range(batch_size)]

            write_to_record_file('##################################################################\n', self.record_file)

            write_to_record_file('-------------------- Aux Environment Prompts --------------------\n', self.record_file)
            write_to_record_file(environment_prompts_aux, self.record_file)
            
            write_to_record_file('-------------------- Aux other_index --------------------\n', self.record_file)
            write_to_record_file(str(nav_input_aux["other_index"]), self.record_file)
            
            ## feed into mllm for aux task
            if np.all(ended_aux):
                a_t_aux = [0] * batch_size
                write_to_record_file('Aux_Already stopped, skip inference.', self.record_file)
            elif len(image_list_aux) > 40:
                a_t_aux = [0]
                write_to_record_file('Aux_Exceed image limit and stop!', self.record_file)
            else:
                json_output_aux = None
                nav_output_aux = ""
                for attempt in range(5):
                    nav_output_aux, tokens_aux = gpt_infer(nav_input_aux["task_description"], nav_input_aux["other_index"], environment_prompts_aux, image_list_aux, image_list,
                                                    self.args.llm, self.args.max_tokens, response_format={"type": "json_object"})
                    try:
                        json_output_aux = json.loads(nav_output_aux)
                        break
                    except json.JSONDecodeError:
                        write_to_record_file(f'Aux_Invalid JSON response, retry {attempt + 1}.\n', self.record_file)

                if json_output_aux is None:
                    write_to_record_file('Aux_Invalid JSON response, use default.\n', self.record_file)
                    only_options = nav_input_aux["only_options"][0]
                    default_action = only_options[0] if len(only_options) > 0 else "A"
                    json_output_aux = {
                        "Thought": "",
                        "New Planning": "No plans currently.",
                        "Action": default_action
                    }

                a_t_aux = self.prompt_manager_aux.parse_json_action(json_output_aux, nav_input_aux["only_options"], t)
                self.prompt_manager_aux.parse_json_planning(json_output_aux)

                write_to_record_file('-------------------- Aux Output --------------------\n', self.record_file)
                write_to_record_file(nav_output_aux, self.record_file)

            if np.any(ended_aux):
                a_t_aux = [0 if ended_aux[i] else a_t_aux[i] for i in range(batch_size)]

            for i in range(batch_size):
                traj[i]['a_t'][t] = a_t[i]
                traj_aux[i]['a_t'][t] = a_t_aux[i]

            # Determine stop actions
            a_t_stop = [a_t_i == 0 for a_t_i in a_t]
            a_t_stop_aux = [a_t_i == 0 for a_t_i in a_t_aux]

            ended = np.logical_or(ended, np.array(a_t_stop))
            ended_aux = np.logical_or(ended_aux, np.array(a_t_stop_aux))

            # Prepare environment action
            cpu_a_t = []
            cpu_a_t_aux = []

            for i in range(batch_size):
                if a_t_stop[i] or ended[i]:
                    cpu_a_t.append(-1)
                    just_ended[i] = True
                else:
                    cpu_a_t.append(a_t[i] - 1)

                if a_t_stop_aux[i] or ended_aux[i]:
                    cpu_a_t_aux.append(-1)
                    just_ended_aux[i] = True
                else:
                    cpu_a_t_aux.append(a_t_aux[i] - 1)

            self.make_equiv_action(cpu_a_t, obs, traj, mode = 'primary')
            self.make_equiv_action(cpu_a_t_aux, obs_aux, traj_aux, mode = 'auxiliary')

            obs = self.env._get_obs()
            obs_aux = self.env_au._get_obs()

            previous_angle = [{'heading': ob['heading'],
                               'elevation': ob['elevation']} for ob in obs]

            previous_angle_aux = [{'heading': ob['heading'],
                               'elevation': ob['elevation']} for ob in obs_aux]

            # we only implement batch_size=1
            if a_t[0] == 0 and a_t_aux[0] == 0:
                break

            self.prompt_manager.make_history(a_t, nav_input, t)
            self.prompt_manager_aux.make_history(a_t_aux, nav_input_aux, t)

        return traj, traj_aux
