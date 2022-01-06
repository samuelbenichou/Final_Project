from Model.Simulation import data_feature_sub
from Model.Simulation.experiment import Experiment
from Model.OFS.ofs_ac import OnlineFeatureSelectionAC
from Model.OL.ol_ac import OnlineLearningAC
from Model.Simulation.parse import Parse




class Controller:
    OFS_CONTROLLER = {
        'MCNN': 'MCNN',
        'ABFS': 'ABFS',
        'Fires': 'fires',
        'None': None
    }

    OL_CONTROLLER = {
        'K-NN': 'KNN',
        'Perceptron Mask (ANN)': 'NeuralNetwork',
        'Random Forest': 'RandomForest',
        'Naive-Bayes': 'NB'
    }


    @classmethod
    def get_relevant_ofs_algorithms(cls, chosen_ofs):
        ofs_instances = []
        ofs_algos = OnlineFeatureSelectionAC.get_all_ofs_algo()
        for ofs_name, ofs_params in chosen_ofs.items():
            if not Controller.OFS_CONTROLLER.get(ofs_name):
                ofs_instances.append(None)  # case without ofs
            else:
                ofs_instance = ofs_algos.get(Controller.OFS_CONTROLLER.get(ofs_name))()
                ofs_instance.set_algorithm_parameters(**ofs_params)
                ofs_instances.append(ofs_instance)
        return ofs_instances

    @classmethod
    def get_relevant_ol_models(cls, chosen_ol):
        ol_instances = []
        ol_models = OnlineLearningAC.get_all_ol_algo()
        for ol_name, ol_params in chosen_ol.items():
            ol_instance = ol_models.get(Controller.OL_CONTROLLER.get(ol_name))()
            ol_instance.set_algorithm_parameters(**ol_params)
            ol_instances.append(ol_instance)
        return ol_instances

    @classmethod

    def run_multi_experiments(cls, file_path, file_name, export_path, data_type, chunk_size, feature_set_type, feature_percentage, ofs_algos, ol_models, window_instance, file_target_index=-1, batch_size=500):
        ol_models = cls.get_relevant_ol_models(ol_models)
        ofs_algos = cls.get_relevant_ofs_algorithms(ofs_algos)
        X, y, classes = Parse.read_ds(file_path, target_index=file_target_index)

        feature_count = data_feature_sub.get_number_of_features_by_percentage(X, feature_percentage)

        ds_exps = []
        for ofs_instance in ofs_algos:
            for ol_instance in ol_models:
                ol_instance.set_algorithm_fit_parameters(classes=classes)

                for i in range(chunk_size, len(X), chunk_size):
                    sub_data = data_feature_sub.select_sub_data_from_data(X, i)
                    if feature_set_type == 'Varying':
                        sub_data = data_feature_sub.select_feature_set_from_data(sub_data, feature_count)
                    elif feature_set_type == 'Trapezoidal':
                        sub_data = data_feature_sub.expand_feature_set_from_data(sub_data, feature_count)

                experiment = Experiment(ofs=ofs_instance, ol=ol_instance, window_size=batch_size, X=sub_data, y=y,
                                        ds_name=file_name, transform_binary=False, special_name='multi')
                ds_exps.append(experiment)
                experiment.run()
                experiment.save(path=export_path)
                window_instance.increase_pb()
        Experiment.save_graphs(ds_exps)

