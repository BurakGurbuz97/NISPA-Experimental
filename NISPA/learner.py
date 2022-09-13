from argparse import Namespace
from avalanche.benchmarks import GenericCLScenario, TCLExperience
import torch
import copy


from  NISPA.architecture import random_prune, Network, get_device
from  NISPA.selection import get_tau_schedule, select_candidate_stable_units, remove_units_without_incoming
from  NISPA.train_eval import task_training, test
from  NISPA.track import TaskLog, SequenceLog, PhaseLog, acc_prev_tasks
from  NISPA.rewire import drop_connections_plastic_to_stable, Growth, fix_plastic_units_without_outgoing
from  NISPA.extra_utils import get_data_loaders, get_plastic_units, get_loss, PhaseVars



class Learner():

    def __init__(self, args: Namespace, network: Network, scenario: GenericCLScenario, log_params: dict):
        self.args = args
        self.optim_obj = getattr(torch.optim, args.optimizer)
        self.pruned_network = random_prune(network.to(get_device()), args.prune_perc)
        print("Model: \n", self.pruned_network)
        self.original_scenario = scenario
        self.tau_schedule = get_tau_schedule(args)
        self.log_params = log_params

    def learn_next_task(self, task_index: int, train_task: TCLExperience, val_task: TCLExperience, test_task: TCLExperience) -> Network:
        print("****** Learning Task-{}   Classes: {} ******".format(task_index + 1, train_task.classes_in_this_experience))
        self.pruned_network.add_seen_classes(train_task.classes_in_this_experience)
        train_loader, val_loader, test_loader = get_data_loaders(self.args, train_task, val_task, test_task)
        phase_vars = PhaseVars(self.pruned_network)

        while(True):
            print("Sparsity phase-{}: {:.2f}".format(phase_vars.phase_index, self.pruned_network.compute_weight_sparsity()))
            loss = get_loss(self.args)()
            optimizer = self.optim_obj(self.pruned_network.parameters(), lr= self.args.learning_rate, weight_decay= 0)

            # Phase Training
            self.pruned_network = task_training(self.pruned_network, loss, optimizer, train_loader, self.args, train_task.classes_in_this_experience)
            acc_after_phase = test(self.pruned_network, val_loader, train_task.classes_in_this_experience)
            print("Accuracy after phase-{}: {:.2f}.".format(phase_vars.phase_index, acc_after_phase))

            stable_selection_perc = self.tau_schedule(phase_vars.phase_index) * 100

            # Should we stop or continue next phase? Phase level early stopping.
            if ((acc_after_phase < phase_vars.best_phase_acc - self.args.recovery_perc) and phase_vars.phase_index > 2) or stable_selection_perc <= 5:
                print('Reverting to the end of previou phase.')
                self.pruned_network = copy.deepcopy(phase_vars.previous_model)
                self.pruned_network.add_new_stable_units(phase_vars.prev_phase_stable_and_candidate_stable_units)
                print('Freezing connections.')
                self.pruned_network.freeze_stable_to_stable()
                if  self.args.reinit:
                    self.pruned_network.re_initialize_not_frozen()
                    break

            # Save network after task learning
            phase_vars.best_phase_acc = max(phase_vars.best_phase_acc, acc_after_phase)
            phase_vars.previous_model = copy.deepcopy(self.pruned_network)
            phase_vars.prev_phase_stable_and_candidate_stable_units = self.pruned_network.current_stable_and_candidate_units
            phase_logger = PhaseLog(self.log_params, phase_vars.phase_index, train_task.classes_in_this_experience, task_index + 1)
            phase_logger.archive(self.pruned_network, train_loader, val_loader, test_loader)


            print('Selecting for capturing {:.2f}% activations.'.format(stable_selection_perc))
            stable_and_candidate_stable_units, candidate_stable_units = select_candidate_stable_units(self.pruned_network, train_loader, stable_selection_perc)
            stable_and_candidate_stable_units = remove_units_without_incoming(stable_and_candidate_stable_units, self.pruned_network)
            self.pruned_network.current_stable_and_candidate_units = stable_and_candidate_stable_units
            

            print('Dropping connections.')
            self.pruned_network, connection_quota = drop_connections_plastic_to_stable(stable_and_candidate_stable_units, self.pruned_network)

            self.pruned_network, connection_quota = fix_plastic_units_without_outgoing(self.pruned_network, stable_and_candidate_stable_units, connection_quota)


            print('Growing connections.')
            connection_grower = Growth(self.pruned_network.list_stable_units, candidate_stable_units,
                                       get_plastic_units(self.pruned_network, stable_and_candidate_stable_units))
            self.pruned_network, connection_quota = connection_grower.grow(self.pruned_network, connection_quota,
                                                                           grow_algo=self.args.grow_method, grow_init=self.args.grow_init)
            if sum(connection_quota) != 0:
                print("Warning: Cannot accomodate all connection growth request. Density will decrease.")

            phase_vars.phase_index = phase_vars.phase_index + 1

        task_logger = TaskLog(self.args, self.log_params, task_index + 1, self.original_scenario)
        task_logger.archive(self.pruned_network)
        return self.pruned_network


    def learn_tasks(self) -> Network:
        for task_index, (train_task, val_task, test_task) in enumerate(zip(self.original_scenario.train_stream,
                                                                           self.original_scenario.val_stream,  # type: ignore
                                                                           self.original_scenario.test_stream)):
            network = self.learn_next_task(task_index, train_task, val_task, test_task) 
        sequence_logger = SequenceLog(self.args, self.log_params, self.original_scenario)
        sequence_logger.archive(network)  # type: ignore
        return network  # type: ignore
