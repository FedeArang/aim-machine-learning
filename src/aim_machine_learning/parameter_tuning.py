import numpy as np
from aim_machine_learning.model_evaluator import ModelEvaluator
import matplotlib.pyplot as plt

class ParametersTuner():

    def __init__(self, model_class, X, y, supported_eval_types, **output_path):

        self.model_class=model_class
        self.X=X
        self.y=y
        self.supported_eval_types=supported_eval_types
        self.output_path=output_path
    
    def tune_parameters(self, param_dict, eval_type, eval_obj, **params):
        
        param_keys=list(param_dict.keys())

        neighbors=False
        
        if len(param_keys)==1:
            my_k=param_keys[0]
            neighbors=True


        self.eval_type=eval_type
    
        if eval_type not in self.supported_eval_types:
            raise NameError('{} eval type not supported'.format(eval_type))
        
        min_value=float('inf')

        results=[]
        
        if neighbors:

            best_params={my_k:-1}

            for k in param_dict[my_k]:

                mod_eval=ModelEvaluator(self.model_class, {my_k:k}, self.X, self.y)

                if eval_type=='ttsplit':

                    MSE=mod_eval.train_test_split_eval(eval_obj, params['test_proportion'])
                
                else:

                    MSE=mod_eval.kfold_cv_eval(eval_obj, params['K'])
                
                result=MSE['mean']+MSE['std']
                
                results.append(result)

                if result<min_value:
                    min_value=result
                    best_params[my_k]=k
            
        
        else:

            best_params={param_keys[0]:np.zeros(len(param_keys[0])), param_keys[1]:0}

            for a in param_dict[param_keys[0]]:
                for b in param_dict[param_keys[1]]:

                    mod_eval=ModelEvaluator(self.model_class, {param_keys[0]:a, param_keys[1]:b}, self.X, self.y)

                    if eval_type=='ttsplit':

                        MSE=mod_eval.train_test_split_eval(eval_obj, params['test_proportion'])
                
                    else:

                        MSE=mod_eval.kfold_cv_eval(eval_obj, params['K'])
                
                    result=MSE['mean']+MSE['std']
                
                    results.append(result)

                    if result<min_value:
                        min_value=result
                        best_params[param_keys[0]]=a
                        best_params[param_keys[1]]=b
            

        prova=params.get('fig_name')
        
            
        if len(self.output_path)>0 and prova is not None:

            plt.figure()
            if neighbors:
                plt.plot(np.array(param_dict[my_k]), np.array(results))
                plt.xlabel('k')
            else:
                plt.plot(param_dict[param_keys[0]], np.array(results))
                plt.xlabel('a')
            plt.title(params['fig_name'])
            plt.ylabel('Upper bound MSE')

            plt.savefig('{} {}'.format((self.output_path)['output_path'], params['fig_name']))
        
        

        return best_params
            



        
        